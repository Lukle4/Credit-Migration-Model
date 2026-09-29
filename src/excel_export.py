"""
Generates the Excel scenario calculator (credit_migration_scenario_calculator.xlsx)
from model_weights.json -- the Excel counterpart of sheets_export.py. Same
motivation: a hand-maintained workbook can silently ship a stale fit, and
Excel's own formula engine can't be executed here to check it, so the
generator is disciplined instead -- weights are read straight from the JSON,
row positions are computed (never hand-typed), and the result is read back
and compared to the source before being trusted.

Rebuilt from scratch after several confirmed bugs in the previous, hand-built
version and in this module's own earlier rebuilds -- each found by actually
opening the file in real Excel or inspecting its raw XML, not guessed:

1. The heatmap conditional formatting used Excel's x14 extended CF format
   (openpyxl warns it can't even read this back) with formulas anchored to
   FIXED, absolute Calc! cells -- e.g. the cumulative grid's rule referenced
   Calc!B288, which is past the last populated cumulative block and did not
   correspond to any horizon. It could not have coloured correctly for any
   input.
2. A later rebuild switched the displayed matrix cells to plain numeric
   formulas with `number_format` set directly (so a `ColorScaleRule` could
   colour them by their own value), which read back as "0.00%" correctly
   immediately after generation -- but a real Excel/OneDrive AutoSave
   round-trip was confirmed (by writing the file, having it opened and
   autosaved, then re-reading it) to silently strip that format back to
   General on this class of formula-driven cell. Nothing on the writing
   side can prevent that round-trip.
3. The cumulative chain's first block only copied the M matrix's top row
   into all four of its own rows, leaving the other three genuinely blank;
   every later MMULT then propagated #VALUE! forward from block 2 on.
4. The tranche-row column placement was hardcoded ("the upgrade always
   lands in column B"), which only happened to be correct for one tranche
   and silently misplaced another's upgrade probability and left its
   deepest-fall column unwritten.

Current design: the displayed cells hold `=TEXT(...,"0.00%")` -- the "%" is
baked into the returned string, so nothing can strip it -- and, since a text
cell can no longer be coloured by its own value, the heatmap is a standard
(non-x14) `FormulaRule`, three tiers (red/yellow/green) per grid, whose
criteria formula reads the SAME hidden Calc-sheet numeric cell the display
text was derived from via `INDEX(...)` and `ROW()`/`COLUMN()` -- computed
explicitly per cell rather than relying on Excel's implicit
relative-reference shifting across a multi-cell CF range, which is closer to
what bug (1)'s x14 rule was attempting and getting wrong. Tranche count and
matrix dimension are derived from `model_weights["tranches"]`
(`len(tranches)+1`), not a literal 4, so this scales correctly however many
tranches are declared.

What IS and ISN'T verified: there is no Excel/LibreOffice in this
environment, so, unlike sheets_export.py (checked by literally running the
JS under Node), the regenerated formulas here are never executed by a real
spreadsheet engine from this side. What verify_workbook DOES check: the
embedded weights match model_weights.json exactly; the Calc-sheet formula
chain's structure (MMULT block count, row ranges, every M-block/block-1 cell
populated, no reference past the sheet's own data) is internally
consistent; the display cells use TEXT(...,"0.00%"); and the conditional
formatting rules genuinely reference the same Calc range the display
formula does, not a stale or hand-typed one. tests/stress_suite.py
additionally (a) replays a scenario's values that were cached in an OLD
workbook by a real copy of Excel and confirms the Python engine still
reproduces them, and (b) symbolically evaluates the Calc sheet's own
cascade-logit formulas across in-range and outlier scenarios and
cross-checks every tranche's row against the trusted Python engine, with a
self-mutation test proving that check still catches the historical
column-placement bug on every run. Bug (2) above -- the AutoSave format
strip -- was found only by a human opening the file in real Excel; nothing
automated here could have caught it, since the file was provably correct at
the moment this module wrote it.
"""

import json
import math
import os

from openpyxl import Workbook, load_workbook
from openpyxl.formatting.rule import FormulaRule
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
from openpyxl.utils import get_column_letter
from openpyxl.worksheet.datavalidation import DataValidation
from openpyxl.worksheet.formula import ArrayFormula

DEFAULT_N_QUARTERS = 60

HEADER_FONT = Font(bold=True)
TITLE_FONT = Font(bold=True, size=16)
SECTION_FONT = Font(bold=True)
INPUT_FILL = PatternFill("solid", fgColor="FFF3CD")
INPUT_BORDER = Border(*(Side(style="thin"),) * 4)
GREEN, YELLOW, RED = "1A9850", "FFFFBF", "D73027"  # same hex values as sheets/Code.gs's gradient, for a consistent look


def _col(i: int) -> str:
    return get_column_letter(i)


def _scenario_input_cells() -> dict:
    """
    Where Scenario's three input cells live. The ONE place this is decided --
    Calc's raw-X-vector formulas (`=Scenario!...`) and _write_scenario_sheet's
    own input-writing code both read it from here, instead of each
    independently assuming the same row numbers and risking silent
    divergence if the layout above them ever changes.
    """
    return {"earnings": "C4", "yield": "C5", "horizon": "C6"}


def _validate_weights(w: dict) -> None:
    """Raises ValueError for anything this template can't handle correctly. Deliberately the same rules as sheets_export._validate_weights -- both engines require the identical 4-tranche/3-2-1-cascade structure."""
    from sheets_export import _validate_weights as _shared_validate

    _shared_validate(w)


# ---------------------------------------------------------------------------
# Calc sheet: cascade-logit formula blocks, one per non-Default tranche.
# _cascade_rows returns (rows, final) where rows is [(label, {0: formula}), ...]
# (column offset 0 = column B) and final maps k=0 (stay), 1 (fall 1 notch),
# ... up to max_fall, to an index into `rows` -- used by the caller to write
# the tranche's summary row once real row numbers are known.
# ---------------------------------------------------------------------------

def _stable_sigmoid_formula(logit_cell: str) -> str:
    return f"IF({logit_cell}>=0,1/(1+EXP(-{logit_cell})),EXP({logit_cell})/(1+EXP({logit_cell})))"


def _cascade_rows(tranche: str, stages: list, beta_row_of: dict, x_range: str) -> tuple:
    max_fall = len(stages)
    if max_fall not in (1, 2, 3):
        raise ValueError(f"{tranche}: cascade with {max_fall} stages is not supported (only 1, 2 or 3 are)")

    rows = []  # (label, {0: formula_no_leading_eq})

    def add(label, formula):
        rows.append((label, {0: formula}))
        return len(rows) - 1  # index into `rows`, resolved to a real row number by the caller

    beta_A = beta_row_of[(tranche, stages[0])]
    i_logit_A = add("logit_A", f"SUMPRODUCT({x_range}, Model_Weights!C{beta_A}:F{beta_A})")
    i_p_ge1 = add("p(fall=1)" if max_fall == 1 else "p(fall>=1)", "__SIGMOID__" + str(i_logit_A))
    i_p_stay = add("p(stay)", "__1MINUS__" + str(i_p_ge1))
    final = {}

    if max_fall == 1:
        final[0] = i_p_stay
        final[1] = i_p_ge1
    else:
        beta_B = beta_row_of[(tranche, stages[1])]
        i_logit_B = add("logit_B", f"SUMPRODUCT({x_range}, Model_Weights!C{beta_B}:F{beta_B})")
        b_label = "p(fall=2 | fall>=1)" if max_fall == 2 else "p(fall>=2 | fall>=1)"
        i_p_B_given = add(b_label, "__SIGMOID__" + str(i_logit_B))
        i_p_eq1 = add("p(fall=1)", f"__MUL1MINUS__{i_p_ge1}__{i_p_B_given}")
        i_p_ge2 = add("p(fall=2)" if max_fall == 2 else "p(fall>=2)", f"__MUL__{i_p_ge1}__{i_p_B_given}")
        final[0] = i_p_stay
        final[1] = i_p_eq1

        if max_fall == 2:
            final[2] = i_p_ge2
        else:
            beta_C = beta_row_of[(tranche, stages[2])]
            i_logit_C = add("logit_C", f"SUMPRODUCT({x_range}, Model_Weights!C{beta_C}:F{beta_C})")
            i_p_C_given = add("p(fall=3 | fall>=2)", "__SIGMOID__" + str(i_logit_C))
            i_p_eq2 = add("p(fall=2)", f"__MUL1MINUS__{i_p_ge2}__{i_p_C_given}")
            i_p_eq3 = add("p(fall=3)", f"__MUL__{i_p_ge2}__{i_p_C_given}")
            final[2] = i_p_eq2
            final[3] = i_p_eq3

    return rows, final


def _resolve_placeholder(formula: str, row_addr: list) -> str:
    """row_addr[i] = 'B<row>' for the i-th row emitted by _cascade_rows -- turns the __SIGMOID__3 etc. placeholders into real formulas now that row numbers are known."""
    if formula.startswith("__SIGMOID__"):
        return "=" + _stable_sigmoid_formula(row_addr[int(formula[len("__SIGMOID__"):])])
    if formula.startswith("__1MINUS__"):
        return f"=1-{row_addr[int(formula[len('__1MINUS__'):])]}"
    if formula.startswith("__MUL1MINUS__"):
        a, b = formula[len("__MUL1MINUS__"):].split("__")
        return f"={row_addr[int(a)]}*(1-{row_addr[int(b)]})"
    if formula.startswith("__MUL__"):
        a, b = formula[len("__MUL__"):].split("__")
        return f"={row_addr[int(a)]}*{row_addr[int(b)]}"
    return "=" + formula


# ---------------------------------------------------------------------------
# Workbook assembly
# ---------------------------------------------------------------------------

def _write_model_weights_sheet(ws, w: dict, macro_features: list, tranches: list) -> dict:
    """Returns a dict of row lookups the Calc/Scenario sheets need: beta_row[(tranche,stage)], upgrade_row[tranche], bounds_row[feature], cap_row[feature]."""
    ws["A1"] = "Model weights (from model_weights.json)"
    ws["A1"].font = TITLE_FONT

    r = 3
    ws[f"A{r}"] = "beta_hat -- fitted macro-sensitivity coefficients per tranche/cascade stage"
    ws[f"A{r}"].font = SECTION_FONT
    r += 1
    for c, label in enumerate(["Tranche", "Stage", "Intercept (X0)", "Delta Earnings (X1)", "Delta Yield (X2)", "Interaction (X3)"]):
        cell = ws.cell(row=r, column=1 + c, value=label)
        cell.font = HEADER_FONT
    r += 1
    beta_row = {}
    for tranche in tranches:
        for stage in w["cascade_stages"][tranche]:
            ws.cell(row=r, column=1, value=tranche)
            ws.cell(row=r, column=2, value=stage)
            for c, v in enumerate(w["beta_hat"][tranche][stage]):
                ws.cell(row=r, column=3 + c, value=float(v))
            beta_row[(tranche, stage)] = r
            r += 1

    r += 1
    ws[f"A{r}"] = "Upgrade probabilities (fixed, non-macro-sensitive -- see README.md point 3)"
    ws[f"A{r}"].font = SECTION_FONT
    r += 1
    ws.cell(row=r, column=1, value="Tranche").font = HEADER_FONT
    ws.cell(row=r, column=2, value="Upgrade Prob").font = HEADER_FONT
    r += 1
    upgrade_row = {}
    for tranche in tranches:
        ws.cell(row=r, column=1, value=tranche)
        ws.cell(row=r, column=2, value=float(w["upgrade_prob"][tranche]))
        upgrade_row[tranche] = r
        r += 1

    def bounds_table(title, block_key, note=None):
        nonlocal r
        r += 1
        ws[f"A{r}"] = title
        ws[f"A{r}"].font = SECTION_FONT
        r += 1
        ws.cell(row=r, column=1, value="Feature").font = HEADER_FONT
        ws.cell(row=r, column=2, value="Min" if "CAP" not in title else "Cap Min").font = HEADER_FONT
        ws.cell(row=r, column=3, value="Max" if "CAP" not in title else "Cap Max").font = HEADER_FONT
        r += 1
        row_of = {}
        display = {"X1_delta_earnings": "Delta Earnings (X1)", "X2_delta_yield": "Delta Yield (X2)", "X3_interaction": "Interaction (X3)"}
        for feat in macro_features:
            lo, hi = w[block_key][feat]
            ws.cell(row=r, column=1, value=display.get(feat, feat))
            ws.cell(row=r, column=2, value=float(lo))
            ws.cell(row=r, column=3, value=float(hi))
            row_of[feat] = r
            r += 1
        if note:
            ws[f"A{r}"] = note
            r += 1
        return row_of

    bounds_row = bounds_table("Observed X_train bounds (extrapolation-warning reference)", "x_train_bounds")
    cap_row = bounds_table(
        "EXTRAPOLATION CAP BOUNDS (per feature -- output becomes static at/beyond these values)",
        "extrapolation_cap_bounds",
        note=("Note: per-feature, not one shared multiplier -- earnings and yield saturate at very different multiples "
              "of their own training range. The interaction (X3) bound applies to the recomputed X1(capped)*X2(capped) "
              "value, so a COMBINED scenario's plateau can begin before either X1 or X2 alone reaches its own bound above."))

    ws.column_dimensions["A"].width = 10
    ws.column_dimensions["B"].width = 8
    for col in "CDEF":
        ws.column_dimensions[col].width = 16
    return {"beta_row": beta_row, "upgrade_row": upgrade_row, "bounds_row": bounds_row, "cap_row": cap_row}


def _write_calc_sheet(ws, w: dict, lookup: dict, n_quarters: int, tranches: list) -> dict:
    beta_row, upgrade_row, cap_row = lookup["beta_row"], lookup["upgrade_row"], lookup["cap_row"]
    macro_feature_names = ["X1_delta_earnings", "X2_delta_yield", "X3_interaction"]

    ws["A1"] = "X vector: row 2 = CAPPED (drives all math below), row 3 = RAW (for the range-check warning only)"
    ws["A1"].font = HEADER_FONT
    for c, label in enumerate(["X0", "X1 (Earn)", "X2 (Yield)", "X3 (Inter.)"]):
        ws.cell(row=1, column=2 + c, value=label).font = HEADER_FONT

    x1cap, x2cap, x3cap = (cap_row[f] for f in macro_feature_names)
    ws["B2"] = 1
    ws["C2"] = f"=MIN(MAX(C3,Model_Weights!B{x1cap}),Model_Weights!C{x1cap})"
    ws["D2"] = f"=MIN(MAX(D3,Model_Weights!B{x2cap}),Model_Weights!C{x2cap})"
    ws["E2"] = f"=MIN(MAX(C2*D2,Model_Weights!B{x3cap}),Model_Weights!C{x3cap})"
    input_cells = _scenario_input_cells()
    ws["A3"] = "X vector (RAW, uncapped)"
    ws["B3"], ws["C3"], ws["D3"], ws["E3"] = 1, f"=Scenario!{input_cells['earnings']}", f"=Scenario!{input_cells['yield']}", "=C3*D3"
    ws["A4"] = "Capped? (text, blank if not)"
    ws["C4"] = '=IF(C2<>C3,"Delta Earnings: "&TEXT(C3,"0.000")&" -> "&TEXT(C2,"0.000"),"")'
    ws["D4"] = '=IF(D2<>D3,"Delta Yield: "&TEXT(D3,"0.000")&" -> "&TEXT(D2,"0.000"),"")'
    ws["E4"] = '=IF(E2<>E3,"Interaction: "&TEXT(E3,"0.000")&" -> "&TEXT(E2,"0.000"),"")'

    r = 7
    final_row_of_tranche = {}  # tranche -> row number of its "[->IG,...,->Default]" summary row
    for idx, tranche in enumerate(tranches):
        ws.cell(row=r, column=1, value=tranche).font = HEADER_FONT
        r += 1
        stages = w["cascade_stages"][tranche]
        rows, final = _cascade_rows(tranche, stages, beta_row, "$B$2:$E$2")
        row_addr = []  # row_addr[i] = "B<row>" for the i-th emitted row
        base = r
        for i, (label, colmap) in enumerate(rows):
            row_addr.append(f"B{base + i}")
        for i, (label, colmap) in enumerate(rows):
            ws.cell(row=base + i, column=1, value=label)
            ws.cell(row=base + i, column=2, value=_resolve_placeholder(colmap[0], row_addr))
        r = base + len(rows)

        u = float(w["upgrade_prob"][tranche])
        max_fall = len(stages)
        if u > 0:
            u_row = upgrade_row[tranche]
            ws.cell(row=r, column=1, value="u (upgrade prob)")
            ws.cell(row=r, column=2, value=f"=Model_Weights!B{u_row}")
            u_cell = f"B{r}"
            r += 1
        else:
            u_cell = None

        ws.cell(row=r, column=1, value=f"{tranche} row [{', '.join('->' + t for t in tranches + ['Default'])}]")
        # Column layout is absolute, by TARGET tranche index, not by how many columns precede the
        # tranche's own block: column 2+idx is always this tranche's own diagonal (stay), and any
        # upgrade -- one tier up -- always lands at column 2+(idx-1), wherever that falls. Previously
        # this used a hardcoded "the upgrade is always column B, data starts at column C", which is
        # only right for idx==1 (NonIG) and silently misplaced Junk's (idx==2) upgrade into the ->IG
        # column while leaving ->Default (column E) never written at all.
        diagonal_col = 2 + idx
        for c in range(2, 6):  # explicitly zero every column first so none is ever left unwritten
            ws.cell(row=r, column=c, value=0)
        if u_cell:
            ws.cell(row=r, column=diagonal_col - 1, value=f"={u_cell}")
        for k in range(0, max_fall + 1):
            src = row_addr[final[k]]
            ws.cell(row=r, column=diagonal_col + k, value=f"=(1-{u_cell})*{src}" if u_cell else f"={src}")
        final_row_of_tranche[tranche] = r
        r += 2

    # n = the transition matrix's actual dimension (every non-Default tranche, plus Default itself) --
    # driven by len(tranches), not a literal 4, so the M block and cumulative chain's row/column math
    # below scales to however many tranches model_weights actually declares.
    n = len(tranches) + 1
    last_col = _col(1 + n)

    m_start = r
    ws.cell(row=r - 1, column=1, value=f"M (Next-Quarter Transition Matrix) [{', '.join(tranches + ['Default'])}]").font = HEADER_FONT
    for idx, tranche in enumerate(tranches):
        src = final_row_of_tranche[tranche]
        for c in range(n):
            ws.cell(row=m_start + idx, column=2 + c, value=f"={_col(2 + c)}{src}")
    for c in range(n):  # Default row: absorbing, [0,...,0,1]
        ws.cell(row=m_start + n - 1, column=2 + c, value=1 if c == n - 1 else 0)

    chain_start = m_start + n + 1
    ws.cell(row=chain_start - 1, column=1,
            value=f"Cumulative transition matrix chain, quarters 1-{n_quarters} (each block = MMULT(previous block, M)). Block 1 = M itself.").font = HEADER_FONT
    m_range = f"$B${m_start}:${last_col}${m_start + n - 1}"
    prev_range = None
    for q in range(1, n_quarters + 1):
        block_row = chain_start + (q - 1) * n
        if q == 1:
            for r_off in range(n):  # block 1 is a plain copy of M -- every row, not just the top one
                for c in range(n):
                    ws.cell(row=block_row + r_off, column=2 + c, value=f"={_col(2 + c)}{m_start + r_off}")
        else:
            ref = f"{prev_range},{m_range}"
            cell = ws.cell(row=block_row, column=2, value=ArrayFormula(f"B{block_row}:{last_col}{block_row + n - 1}", f"=MMULT({ref})"))
        prev_range = f"$B${block_row}:${last_col}${block_row + n - 1}"
    last_row = chain_start + n_quarters * n - 1

    ws.column_dimensions["A"].width = 34
    return {"m_start": m_start, "chain_start": chain_start, "last_row": last_row, "bounds_row": lookup["bounds_row"], "n": n, "last_col": last_col}


def _write_scenario_sheet(ws, calc: dict, n_quarters: int, tranches: list) -> None:
    ws["A1"] = "Credit Migration Scenario Calculator"
    ws["A1"].font = TITLE_FONT

    input_cells = _scenario_input_cells()  # single source of truth, shared with Calc's raw-X-vector formulas
    r = 3
    ws.cell(row=r, column=1, value="Inputs (percentage points, e.g. 0.5 = +0.5%)").font = SECTION_FONT
    r += 1
    for text, default, cell_ref in (("Delta Earnings Growth", 0, input_cells["earnings"]),
                                     ("Delta 10Y Yield", 0, input_cells["yield"]),
                                     (f"Cumulative Horizon (quarters, 1-{n_quarters})", min(10, n_quarters), input_cells["horizon"])):
        if cell_ref != f"C{r}":
            raise ValueError(f"_scenario_input_cells()[...]={cell_ref!r} no longer matches this section's actual row ({r}) -- update it to match")
        ws.cell(row=r, column=1, value=text)
        cell = ws[cell_ref]
        cell.value = default
        cell.font = HEADER_FONT
        cell.fill = INPUT_FILL
        cell.border = INPUT_BORDER
        r += 1
    horizon_cell = input_cells["horizon"]

    b1, b2, b3 = (calc["bounds_row"][f] for f in ("X1_delta_earnings", "X2_delta_yield", "X3_interaction"))
    ws.cell(row=r, column=1, value="Scenario check")
    check_cell = ws["C" + str(r)]  # display column: same fixed column as the input cells (a layout choice, not data-dependent)
    check_cell.value = (
        f'=IF(OR(Calc!C3<Model_Weights!B{b1}, Calc!C3>Model_Weights!C{b1}, Calc!D3<Model_Weights!B{b2}, Calc!D3>Model_Weights!C{b2}, '
        f'Calc!E3<Model_Weights!B{b3}, Calc!E3>Model_Weights!C{b3}), "EXTRAPOLATION WARNING: input(s) outside observed training range" & '
        f'IF(OR(Calc!C4<>"",Calc!D4<>"",Calc!E4<>""), " | INPUT CAPPED (result reflects capped value): " & TRIM(Calc!C4&" "&Calc!D4&" "&Calc!E4), ""), '
        f'"OK - within training range")')
    check_cell.alignment = Alignment(wrap_text=True)
    r += 2

    targets = tranches + ["Default"]
    headers = [""] + [f"-> {t}" for t in targets]
    row_labels = [f"{t} ->" for t in targets]
    cf_rules = []  # (range, value_expr_for_that_range's_top-left-relative_cell)

    # Display cells use TEXT(...,"0.00%") rather than a numeric cell + number_format: confirmed by
    # directly inspecting the file after a real Excel/OneDrive AutoSave round-trip that the number
    # format (correct in the freshly generated file) gets silently stripped back to General by that
    # round-trip, on THIS specific class of formula-driven cell -- not something this environment can
    # prevent from the writing side. TEXT() bakes the "%" into the returned string itself, so nothing
    # downstream can strip it. That makes the displayed cells text, not numbers, so the heatmap can no
    # longer colour them by their own value (openpyxl ColorScaleRule needs numeric cells) -- instead,
    # each FormulaRule below reads the SAME hidden Calc-sheet numeric cell the display text was
    # derived from, computed via ROW()/COLUMN() rather than relying on Excel's implicit
    # relative-reference shifting across a CF range (the mechanism that broke the original hand-built
    # file for exactly this kind of block).
    def matrix_block(title, calc_range, row_index_of, row_index_of_cf):
        nonlocal r
        ws.cell(row=r, column=1, value=title).font = SECTION_FONT
        r += 1
        for c, h in enumerate(headers):
            if h:
                ws.cell(row=r, column=1 + c, value=h).font = HEADER_FONT
        r += 1
        first_data_row = r
        for i, rl in enumerate(row_labels):
            ws.cell(row=r, column=1, value=rl).font = HEADER_FONT
            for c in range(len(targets)):
                value_expr = f"INDEX({calc_range}, {row_index_of(i)}, {c + 1})"
                ws.cell(row=r, column=2 + c, value=f'=TEXT({value_expr},"0.00%")')
            r += 1

        cf_range = f"{_col(2)}{first_data_row}:{_col(1 + len(targets))}{first_data_row + len(targets) - 1}"
        cf_value_expr = f"INDEX({calc_range}, {row_index_of_cf(first_data_row)}, COLUMN()-1)"
        cf_rules.append((cf_range, cf_value_expr))
        r += 1

    m_range = f"Calc!$B${calc['m_start']}:${calc['last_col']}${calc['m_start'] + calc['n'] - 1}"
    matrix_block("Next-Quarter Transition Matrix", m_range,
                 row_index_of=lambda i: str(i + 1),
                 row_index_of_cf=lambda first_data_row: f"ROW()-{first_data_row}+1")

    horizon_abs = "$" + horizon_cell[0] + "$" + horizon_cell[1:]  # e.g. "C6" -> "$C$6"
    chain_range = f"Calc!$B${calc['chain_start']}:${calc['last_col']}${calc['last_row']}"

    def cum_display(i):  # wraps the whole TEXT() call, not just the INDEX -- "Invalid horizon" must stay a plain string
        return f"({horizon_abs}-1)*{calc['n']}+{i + 1}"

    def wrap_cum_cell_invalid_guard(formula_with_eq: str) -> str:
        inner = formula_with_eq[1:]  # strip the leading "="
        return f'=IF(OR({horizon_abs}<1,{horizon_abs}<>ROUND({horizon_abs},0),{horizon_abs}>{n_quarters}),"Invalid horizon",{inner})'

    matrix_block("N-Quarter Cumulative Transition Matrix (sustained scenario)", chain_range,
                 row_index_of=cum_display,
                 row_index_of_cf=lambda first_data_row: f"({horizon_abs}-1)*{calc['n']}+ROW()-{first_data_row}+1")

    # retrofit the invalid-horizon guard onto the cumulative block's display cells (matrix_block above
    # doesn't know which block is which, so this stays a small, explicit post-pass rather than another
    # branch parameter)
    cum_first_row = cf_rules[1][0].split(":")[0][1:]  # e.g. "B18:E21" -> "18"
    cum_first_row = int("".join(ch for ch in cum_first_row if ch.isdigit()))
    for rr in range(cum_first_row, cum_first_row + len(targets)):
        for cc in range(2, 2 + len(targets)):
            cell = ws.cell(row=rr, column=cc)
            cell.value = wrap_cum_cell_invalid_guard(cell.value)

    GREEN_FILL, YELLOW_FILL, RED_FILL = (PatternFill("solid", fgColor=c) for c in (GREEN, YELLOW, RED))
    for cf_range, value_expr in cf_rules:
        ws.conditional_formatting.add(cf_range, FormulaRule(formula=[f"{value_expr}>=0.66"], fill=RED_FILL))
        ws.conditional_formatting.add(cf_range, FormulaRule(formula=[f"AND({value_expr}>=0.33,{value_expr}<0.66)"], fill=YELLOW_FILL))
        ws.conditional_formatting.add(cf_range, FormulaRule(formula=[f"{value_expr}<0.33"], fill=GREEN_FILL))

    ws.cell(row=r, column=1, value="Note: output becomes static (flat) beyond these input bounds:")
    r += 1
    ws.cell(row=r, column=1, value="See the Model_Weights tab (EXTRAPOLATION CAP BOUNDS) for the exact current values and why.")

    dv = DataValidation(type="whole", formula1=1, formula2=n_quarters, showErrorMessage=True,
                         errorTitle="Invalid horizon", error=f"Enter a whole number of quarters from 1 to {n_quarters}.")
    ws.add_data_validation(dv)
    dv.add(horizon_cell)

    ws.column_dimensions["A"].width = 24
    ws.column_dimensions["B"].width = 12


def _write_readme_sheet(ws, w: dict, macro_features: list) -> None:
    x1c, x2c, x3c = (w["extrapolation_cap_bounds"][f] for f in macro_features)
    lines = [
        ("Credit Migration Scenario Calculator", True),
        ("", False),
        ("What this is", True),
        ("Pure-formula Excel port of the cascade-logit Markov engine from src/markov_dgp.py and src/markov_engine.py "
         "-- no macros, no add-ins, opens anywhere. Generated by src/excel_export.py from model_weights.json -- do not "
         "edit the Calc or Model_Weights tabs by hand, they will be overwritten on the next regeneration.", False),
        ("", False),
        ("How to use it", True),
        ("1. Go to the 'Scenario' tab.", False),
        ("2. Edit the two yellow input cells: Delta Earnings Growth and Delta 10Y Yield, both in PERCENTAGE POINTS "
         "(type 0.5 for +0.5%, i.e. +50bp -- not 0.005). This matches exactly how the Python model was fit.", False),
        (f"3. Edit Cumulative Horizon (quarters, 1-{DEFAULT_N_QUARTERS}) to choose how many quarters to compound the scenario over. "
         "If this is ever invalid (0, negative, non-integer), the Cumulative grid shows 'Invalid horizon' instead of an error.", False),
        ("4. The 'Scenario check' cell tells you if your inputs are within the range the model was actually fit on. "
         "If they're beyond the per-feature cap (see the Model_Weights tab), it also shows an INPUT CAPPED note -- "
         "inputs that extreme are clamped before the matrix math runs, so the sigmoid can't saturate into a literal, "
         'historically-unprecedented 0%/100% (e.g. "100% of IG demotes in one quarter"). The displayed matrix reflects '
         "the CAPPED value whenever that note appears, not your literal typed input.", False),
        ("5. The two 4x4 grids below show the next-quarter and N-quarter-cumulative transition matrices as real "
         "percentages, colour-scaled green (low probability) to red (high probability).", False),
        ("6. If a grid ever looks stuck (doesn't change when you edit the inputs), press Ctrl+Alt+F9 to force a full recalculation.", False),
        ("", False),
        ("IMPORTANT: where the output becomes static (flat)", True),
        (f"Delta Earnings Growth is clamped to [{x1c[0]:.3f}, {x1c[1]:.3f}]. Any input at or beyond that boundary computes "
         "the SAME result as the boundary value itself -- this is not a bug.", False),
        (f"Delta 10Y Yield is clamped to [{x2c[0]:.3f}, {x2c[1]:.3f}] -- an EXPLICIT range, not a multiplier of the training "
         "range. Yield's fitted sensitivity is smaller than earnings', so a small multiplier leaves it needlessly "
         "restricted while a large one overshoots into saturation -- this range was chosen from where the model is "
         "actually still informative, not derived mechanically.", False),
        (f"The interaction term (Delta Earnings x Delta Yield) is capped to [{x3c[0]:.3f}, {x3c[1]:.3f}] AFTER being "
         "recomputed from the already-capped values above -- capping the two factors alone doesn't sufficiently bound "
         "their product. This means a COMBINED scenario's plateau can begin before either input alone reaches its own boundary.", False),
        ("Past these boundaries the tool can no longer distinguish \"moderately extreme\" from \"wildly extreme\" -- that's "
         "the deliberate tradeoff of preventing saturation-driven nonsense, not a defect.", False),
        ("", False),
        ("Other tabs", True),
        ("Model_Weights -- the fitted beta_hat coefficients, observed training-data bounds, and the explicit per-feature "
         "capped-input boundaries, generated from model_weights.json. Reference only, not meant to be edited.", False),
        ("Calc (hidden) -- intermediate cascade-logit calculations and the pre-computed cumulative matrix chain. Unhide "
         "it (right-click any tab) if you want to see the full working.", False),
        ("", False),
        ("Source of truth", True),
        ("All beta_hat values here come directly from model_weights.json, generated by python run_pipeline.py. "
         "Re-run that (which now also regenerates this file) to refresh the numbers.", False),
    ]
    for i, (text, bold) in enumerate(lines, start=1):
        cell = ws.cell(row=i, column=1, value=text)
        cell.font = TITLE_FONT if i == 1 else (SECTION_FONT if bold else Font(bold=False))
    ws.column_dimensions["A"].width = 110


def build_workbook(model_weights: dict, n_quarters: int = DEFAULT_N_QUARTERS) -> Workbook:
    _validate_weights(model_weights)
    if not (isinstance(n_quarters, int) and n_quarters >= 1):
        raise ValueError(f"n_quarters must be a positive integer, got {n_quarters!r}")
    macro_features = list(model_weights["feature_names"][1:])
    tranches = list(model_weights["tranches"][:-1])  # every tranche but Default, in order -- validated above to be exactly ["IG","NonIG","Junk"]

    wb = Workbook()
    wb.remove(wb.active)
    ws_readme = wb.create_sheet("README")
    ws_scenario = wb.create_sheet("Scenario")
    ws_weights = wb.create_sheet("Model_Weights")
    ws_calc = wb.create_sheet("Calc")

    lookup = _write_model_weights_sheet(ws_weights, model_weights, macro_features, tranches)
    calc = _write_calc_sheet(ws_calc, model_weights, lookup, n_quarters, tranches)
    _write_scenario_sheet(ws_scenario, calc, n_quarters, tranches)
    _write_readme_sheet(ws_readme, model_weights, macro_features)

    ws_calc.sheet_state = "hidden"
    wb.active = wb.sheetnames.index("Scenario")
    from openpyxl.workbook.properties import CalcProperties
    wb.calculation = CalcProperties(fullCalcOnLoad=True)

    verify_workbook(wb, model_weights)
    return wb


def verify_workbook(wb_or_path, model_weights: dict) -> bool:
    """Raises ValueError unless the Model_Weights sheet of `wb_or_path` holds exactly `model_weights`'s numbers, and the Calc-sheet cumulative chain is structurally sound (no block referencing past the sheet's own data -- the exact class of bug this generator replaced)."""
    wb = load_workbook(wb_or_path, data_only=False) if isinstance(wb_or_path, str) else wb_or_path
    ws_weights, ws_calc = wb["Model_Weights"], wb["Calc"]
    macro_features = list(model_weights["feature_names"][1:])
    tranches = list(model_weights["tranches"][:-1])

    lookup = {
        "beta_row": {(t, s): r for t in tranches for s, r in
                     [(row[1], i) for i, row in enumerate(ws_weights.iter_rows(min_row=5, max_col=2, values_only=True), start=5) if row[0] == t]},
    }
    for tranche in tranches:
        for stage in model_weights["cascade_stages"][tranche]:
            r = lookup["beta_row"].get((tranche, stage))
            if r is None:
                raise ValueError(f"Model_Weights sheet is missing a row for {tranche}/{stage}")
            want = [float(v) for v in model_weights["beta_hat"][tranche][stage]]
            got = [ws_weights.cell(row=r, column=3 + c).value for c in range(len(want))]
            if any(g is None or abs(g - w) > 1e-12 for g, w in zip(got, want)):
                raise ValueError(f"Model_Weights beta row for {tranche}/{stage} = {got}, expected {want}")

    for r in range(1, ws_weights.max_row + 1):
        label = ws_weights.cell(row=r, column=1).value
        if label == "Tranche" and ws_weights.cell(row=r, column=2).value == "Upgrade Prob":
            for i, tranche in enumerate(tranches, start=r + 1):
                got_t, got_u = ws_weights.cell(row=i, column=1).value, ws_weights.cell(row=i, column=2).value
                want_u = float(model_weights["upgrade_prob"][got_t])
                if got_t != tranche or abs(got_u - want_u) > 1e-12:
                    raise ValueError(f"Model_Weights upgrade_prob row {i} = ({got_t}, {got_u}), expected ({tranche}, {want_u})")
            break

    for block_key in ("x_train_bounds", "extrapolation_cap_bounds"):
        header = "Min" if block_key == "x_train_bounds" else "Cap Min"
        for r in range(1, ws_weights.max_row + 1):
            if ws_weights.cell(row=r, column=2).value == header:
                for i, feat in enumerate(macro_features, start=r + 1):
                    lo, hi = float(ws_weights.cell(row=i, column=2).value), float(ws_weights.cell(row=i, column=3).value)
                    want_lo, want_hi = (float(v) for v in model_weights[block_key][feat])
                    if abs(lo - want_lo) > 1e-12 or abs(hi - want_hi) > 1e-12:
                        raise ValueError(f"Model_Weights {block_key} row for {feat} = [{lo},{hi}], expected [{want_lo},{want_hi}]")
                break

    _verify_cumulative_chain_bounds(wb)
    return True


def _verify_cumulative_chain_bounds(wb) -> None:
    """
    Regression test for exactly the historical bug (a heatmap rule pointed at
    Calc!B288, past the last real cumulative-matrix block, and never tracked
    the selected horizon): recovers the intended chain length from the
    Scenario sheet's own C6 data-validation bound (not a re-derived guess),
    locates the Calc sheet's cascade-block/chain markers by their labels, and
    checks every reference -- the chain's own internal MMULT formulas AND the
    Scenario sheet's cumulative-grid INDEX formula -- points at rows that
    actually exist and matches the chain's true, computed end row exactly.
    """
    import re

    ws_calc, ws_scenario = wb["Calc"], wb["Scenario"]
    horizon_cell = _scenario_input_cells()["horizon"]
    dv = next(iter(ws_scenario.data_validations.dataValidation), None)
    if dv is None or dv.sqref is None or horizon_cell not in str(dv.sqref):
        raise ValueError(f"Scenario!{horizon_cell} has no data validation -- can't recover the intended cumulative horizon to check against")
    n_quarters = int(dv.formula2)

    # Locate the two displayed matrix blocks by their own title labels, rather than assuming their
    # rows -- so this stays correct however many rows the input section above ends up using. Width
    # (n_targets = every tranche plus Default) comes from how many columns the next-quarter grid's
    # own header row actually filled in, not a literal 4 -- both the Calc-sheet block size below and
    # the completeness check reuse this SAME discovered value.
    next_q_title_row = cum_title_row = None
    for row in ws_scenario.iter_rows(min_col=1, max_col=1):
        for cell in row:
            if not isinstance(cell.value, str):
                continue
            if cell.value.startswith("Next-Quarter Transition Matrix"):
                next_q_title_row = cell.row
            elif cell.value.startswith("N-Quarter Cumulative Transition Matrix"):
                cum_title_row = cell.row
    if next_q_title_row is None or cum_title_row is None:
        raise ValueError("Scenario sheet is missing its 'Next-Quarter Transition Matrix' or 'N-Quarter Cumulative Transition Matrix' title label")
    next_q_first_row, cum_first_row = next_q_title_row + 2, cum_title_row + 2
    header_row = next_q_title_row + 1
    n_targets = sum(1 for c in range(2, ws_scenario.max_column + 1) if ws_scenario.cell(row=header_row, column=c).value)
    if n_targets == 0:
        raise ValueError("Next-Quarter Transition Matrix has no header columns to measure its width from")
    last_col = _col(1 + n_targets)

    label_row = {}
    for row in ws_calc.iter_rows(min_col=1, max_col=1):
        for cell in row:
            if isinstance(cell.value, str):
                if cell.value.startswith("M (Next-Quarter"):
                    label_row["m_header"] = cell.row
                elif cell.value.startswith("Cumulative transition matrix chain"):
                    label_row["chain_header"] = cell.row
    if "m_header" not in label_row or "chain_header" not in label_row:
        raise ValueError("Calc sheet is missing its 'M (Next-Quarter...' or 'Cumulative transition matrix chain...' label row")
    m_start = label_row["m_header"] + 1
    chain_start = label_row["chain_header"] + 1
    expected_last_row = chain_start + n_quarters * n_targets - 1

    # Regression test for a second, more basic bug caught by opening the file in real Excel: block 1
    # (the cumulative chain's first block, a plain copy of M) once only had its top row written -- the
    # other rows were silently blank, which every later MMULT then propagated forward as #VALUE! the
    # whole chain. Every cell in both the M block and block 1 must hold a real formula/value.
    for label, top_row in (("M block", m_start), ("cumulative block 1", chain_start)):
        for r_off in range(n_targets):
            for c_off in range(n_targets):
                cell = ws_calc.cell(row=top_row + r_off, column=2 + c_off)
                if cell.value is None:
                    raise ValueError(f"{label} is missing a value at {cell.coordinate} -- a downstream MMULT would silently propagate #VALUE! from here")

    array_refs = []
    for row in ws_calc.iter_rows():
        for cell in row:
            if isinstance(cell.value, ArrayFormula):
                array_refs.append((cell.row, cell.value.ref, cell.value.text))
    # n_quarters == 1 is a valid edge case (the horizon input allows it) with no MMULT at all -- block 1
    # is the whole chain, a plain copy of M, nothing to multiply. Only n_quarters > 1 requires blocks 2+.
    if not array_refs:
        if n_quarters > 1:
            raise ValueError(f"Calc sheet has no MMULT array formulas, but n_quarters={n_quarters} > 1 -- the chain beyond block 1 was not written")
    else:
        last_top_row, last_ref, _ = max(array_refs, key=lambda t: t[0])
        ref_end_row = int(re.search(r":\$?[A-Z]+\$?(\d+)", last_ref).group(1))
        if ref_end_row != expected_last_row:
            raise ValueError(f"last cumulative block's range ends at row {ref_end_row}, expected {expected_last_row} "
                              f"(n_quarters={n_quarters}, chain_start={chain_start})")

    for top_row, ref, text in array_refs:
        for m in re.finditer(r"\$?[A-Z]+\$(\d+)", text):
            referenced = int(m.group(1))
            if referenced > expected_last_row or referenced < m_start:
                raise ValueError(f"Calc!B{top_row}'s formula ({text}) references row {referenced}, outside [{m_start}, {expected_last_row}]")

    expected_calc_ref = f"Calc!$B${chain_start}:${last_col}${expected_last_row}"
    cum_cell = f"B{cum_first_row}"
    cum_formula = ws_scenario[cum_cell].value or ""
    if expected_calc_ref not in cum_formula:
        raise ValueError(f"Scenario!{cum_cell}'s cumulative-grid formula does not reference {expected_calc_ref} (found: {cum_formula!r})")

    # Display cells deliberately use TEXT(...,"0.00%") -- confirmed by inspecting a real file after an
    # Excel/OneDrive AutoSave round-trip that a plain numeric cell + number_format (the earlier design)
    # gets its format silently stripped back to General by that round-trip on this class of
    # formula-driven cell. TEXT() bakes the "%" into the returned string, immune to that.
    for coord in (f"B{next_q_first_row}", cum_cell):
        formula = ws_scenario[coord].value or ""
        if 'TEXT(' not in formula or '"0.00%"' not in formula:
            raise ValueError(f"Scenario!{coord} does not wrap its result in TEXT(...,\"0.00%\") -- required so the display survives an Excel/OneDrive AutoSave round-trip: {formula!r}")

    # Since the displayed cells are now text, the heatmap can't colour them by their own value -- it
    # must read the same hidden Calc-sheet numeric source the display text was derived from. Confirms
    # every colour tier (red/yellow/green) is present and its formula actually references an INDEX
    # into the SAME Calc range the display formula uses (calc_range/chain_range respectively) -- not a
    # hand-typed range that could silently drift from it, and specifically not the historical bug's
    # fixed absolute cell.
    next_q_calc_ref = f"Calc!$B${m_start}:${last_col}${m_start + n_targets - 1}"
    cum_calc_ref = f"Calc!$B${chain_start}:${last_col}${expected_last_row}"
    expected_cf = {
        f"B{next_q_first_row}:{last_col}{next_q_first_row + n_targets - 1}": next_q_calc_ref,
        f"B{cum_first_row}:{last_col}{cum_first_row + n_targets - 1}": cum_calc_ref,
    }
    got_ranges_by_sqref: dict = {}
    for cf in ws_scenario.conditional_formatting:
        got_ranges_by_sqref.setdefault(str(cf.sqref), []).extend(cf.rules)
    if set(got_ranges_by_sqref) != set(expected_cf):
        raise ValueError(f"Scenario sheet's conditional-formatting ranges are {set(got_ranges_by_sqref)}, expected {set(expected_cf)}")
    for rng, calc_ref in expected_cf.items():
        rules = got_ranges_by_sqref[rng]
        if len(rules) < 3:
            raise ValueError(f"Scenario!{rng} has only {len(rules)} conditional-format rule(s), expected 3 (red/yellow/green tiers)")
        formulas = " | ".join((r.formula[0] if r.formula else "") for r in rules)
        if "INDEX(" not in formulas or calc_ref not in formulas:
            raise ValueError(f"Scenario!{rng}'s conditional-format rules don't reference {calc_ref} (found: {formulas!r})")


def export_workbook(weights_path: str, out_path: str, n_quarters: int = DEFAULT_N_QUARTERS):
    with open(weights_path, encoding="utf-8") as f:
        model_weights = json.load(f)
    wb = build_workbook(model_weights, n_quarters)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    wb.save(out_path)
    return wb
