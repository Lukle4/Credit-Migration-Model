"""
Automated stress tests, organized by standard testing principles:

- Boundary value analysis: zero/negative/huge inputs at the edges of what's valid
- Equivalence partitioning: one representative case per input class (valid tranche,
  invalid tranche, valid weight vector, all-zero weight vector, ...)
- Negative testing: malformed input should fail LOUDLY (an exception), never
  silently produce garbage (NaN, a wrong-shaped result, a probability >1)
- Property-based fuzzing: invariants (rows sum to 1, no negative probabilities,
  no NaN/Inf) checked over hundreds of RANDOM inputs across a wide range, not
  just a few hand-picked examples
- Determinism / regression: same seed -> byte-identical output, every time
- Cross-seed invariance: the pipeline's own internal sanity checks
  (engine_sanity_check, validate_softmax_and_chain, validate_wls_reduces_to_ols)
  are supposed to hold regardless of random seed -- if any of them ever fail
  for some seed, that IS a bug, not just noise

Each check function takes `ns`, a dict-like namespace providing the functions
and objects under test (config, cascade_row, fit_ols, etc.). This lets the
exact same checks run against either the local src/ modules or the notebook's
own extracted code -- see run_stress.py for how each namespace is built.
Every check returns (passed: bool, message: str) and never raises; the
runner is what turns an unexpected exception into a loud FAIL rather than a
crashed test run.
"""

import numpy as np


def _safe(fn):
    try:
        return fn(), None
    except Exception as e:  # noqa: BLE001 -- intentionally broad, this is a test harness
        return None, f"{type(e).__name__}: {e}"


# ---------------------------------------------------------------------------
# 1. Extreme / out-of-range scenario stability (boundary value analysis)
# ---------------------------------------------------------------------------

def check_extreme_scenarios_stay_valid(ns) -> tuple:
    config = ns["config"]
    cascade_row = ns["cascade_row"]
    beta_hat = config.BETA_TRUE  # any valid beta dict works for this check

    extreme_rows = [
        np.array([1, 0, 0, 0]),
        np.array([1, 1e3, -1e3, -1e6]),
        np.array([1, -1e3, 1e3, -1e6]),
        np.array([1, 1e6, 1e6, 1e12]),
        np.array([1, -1e6, -1e6, 1e12]),
        np.array([1, 1e-10, 1e-10, 1e-20]),
    ]
    for x_row in extreme_rows:
        for tranche in ["IG", "NonIG", "Junk"]:
            result, err = _safe(lambda: cascade_row(tranche, x_row, beta_hat))
            if err:
                return False, f"cascade_row raised on extreme input {x_row} / {tranche}: {err}"
            if not np.all(np.isfinite(result)):
                return False, f"non-finite output for {x_row} / {tranche}: {result}"
            if np.any(result < -1e-9) or np.any(result > 1 + 1e-9):
                return False, f"probability out of [0,1] for {x_row} / {tranche}: {result}"
            if abs(sum(result) - 1.0) > 1e-6:
                return False, f"row does not sum to 1 for {x_row} / {tranche}: {result} (sum={sum(result)})"
    return True, f"{len(extreme_rows)} extreme scenarios x 3 tranches all stayed valid"


# ---------------------------------------------------------------------------
# 2. Negative testing: malformed inputs must fail loudly, not silently
# ---------------------------------------------------------------------------

def check_unknown_tranche_fails_loudly(ns) -> tuple:
    config = ns["config"]
    cascade_row = ns["cascade_row"]
    x_row = np.array([1, 0.1, 0.1, 0.01])
    result, err = _safe(lambda: cascade_row("NotATranche", x_row, config.BETA_TRUE))
    if err is None:
        return False, f"cascade_row('NotATranche', ...) did NOT raise -- silently returned {result}"
    return True, f"correctly raised: {err}"


def check_all_zero_weights_do_not_silently_corrupt_fit(ns) -> tuple:
    fit_ols = ns["fit_ols"]
    rng = np.random.default_rng(0)
    X = np.column_stack([np.ones(20), rng.normal(size=20), rng.normal(size=20), rng.normal(size=20)])
    Y = rng.normal(size=20)
    result, err = _safe(lambda: fit_ols(X, Y, weights=np.zeros(20)))
    if err is None and result is not None and np.all(np.isfinite(result["beta_hat"])):
        return False, f"all-zero weights produced a finite beta_hat with no warning/error: {result['beta_hat']}"
    return True, f"all-zero weights correctly failed or produced non-finite output ({err or 'non-finite result'})"


# ---------------------------------------------------------------------------
# 3. Config parameter extremes (monkeypatched, then restored)
# ---------------------------------------------------------------------------

def check_zero_smoothing_epsilon_breaks_as_expected(ns) -> tuple:
    """
    SMOOTHING_EPSILON exists specifically to stop ln(0). Setting it to 0
    should reintroduce that failure -- proving the epsilon is load-bearing,
    not decorative. If this DOESN'T fail, the smoothing logic has silently
    stopped depending on the epsilon somewhere.
    """
    config = ns["config"]
    stage_fn = ns["_stage_log_odds_and_weight"]
    original = config.SMOOTHING_EPSILON
    try:
        config.SMOOTHING_EPSILON = 0
        counts_row = np.array([50, 0, 0, 0])  # a zero count with zero smoothing -> ln(0)
        result, err = _safe(lambda: stage_fn(counts_row, 0, "A", 3))
        if err is None and np.isfinite(result[0]):
            return False, f"zero epsilon did NOT break ln(0) as expected -- got {result}, epsilon may be dead code"
        return True, f"zero epsilon correctly produced a non-finite/erroring result ({err or result})"
    finally:
        config.SMOOTHING_EPSILON = original


def check_tiny_company_pool_still_produces_valid_probabilities(ns) -> tuple:
    config = ns["config"]
    simulate_macro_paths = ns["simulate_macro_paths"]
    build_feature_matrix = ns["build_feature_matrix"]
    build_transition_tensor = ns["build_transition_tensor"]
    simulate_observed_counts = ns["simulate_observed_counts"]

    original = config.COMPANY_POOL_SIZE
    try:
        config.COMPANY_POOL_SIZE = 5  # extreme sparsity
        macro_df = simulate_macro_paths()
        X = build_feature_matrix(macro_df)
        p_true = build_transition_tensor(X, config.BETA_TRUE)
        result, err = _safe(lambda: simulate_observed_counts(p_true))
        if err:
            return False, f"pool_size=5 raised: {err}"
        counts = result
        if not np.all(counts >= 0):
            return False, "negative counts produced with tiny pool size"
        row_totals = counts.sum(axis=2)
        if not np.allclose(row_totals, config.COMPANY_POOL_SIZE):
            return False, f"row totals don't match pool size: {row_totals[0]}"
        return True, "pool_size=5 (extreme sparsity) still produced valid, non-negative counts summing correctly"
    finally:
        config.COMPANY_POOL_SIZE = original


# ---------------------------------------------------------------------------
# 4. Property-based fuzzing: invariants over many random inputs
# ---------------------------------------------------------------------------

def check_random_fuzzing_never_violates_invariants(ns, n_trials: int = 300) -> tuple:
    config = ns["config"]
    cascade_row = ns["cascade_row"]
    rng = np.random.default_rng(12345)

    violations = []
    for i in range(n_trials):
        scale = 10 ** rng.uniform(-3, 3)  # spans tiny to huge
        x1, x2 = rng.normal(scale=scale, size=2)
        x_row = np.array([1.0, x1, x2, x1 * x2])
        tranche = rng.choice(["IG", "NonIG", "Junk"])
        result, err = _safe(lambda: cascade_row(tranche, x_row, config.BETA_TRUE))
        if err:
            violations.append(f"trial {i} ({tranche}, x={x_row}): raised {err}")
            continue
        if not np.all(np.isfinite(result)):
            violations.append(f"trial {i} ({tranche}, x={x_row}): non-finite {result}")
        elif np.any(np.array(result) < -1e-9) or np.any(np.array(result) > 1 + 1e-9):
            violations.append(f"trial {i} ({tranche}, x={x_row}): out of [0,1] {result}")
        elif abs(sum(result) - 1.0) > 1e-6:
            violations.append(f"trial {i} ({tranche}, x={x_row}): sum={sum(result)}")

    if violations:
        return False, f"{len(violations)}/{n_trials} random trials violated an invariant. First: {violations[0]}"
    return True, f"{n_trials} random fuzzed scenarios (scale 1e-3 to 1e3) all satisfied every invariant"


def check_capping_prevents_saturation(ns) -> tuple:
    """
    Regression test for a real user-found issue: wildly out-of-range
    scenarios (e.g. a 10-percentage-point quarterly yield shock) pushed the
    sigmoid into full saturation, producing literal 0%/100% probabilities --
    "100% of IG demotes in one quarter" has no historical precedent, however
    mathematically "correct" it was for a linear-in-logit model extrapolated
    that far. predict_scenario now caps inputs to 3x the training bounds
    before the sigmoid math runs; this confirms extreme scenarios (a) are
    detected as needing capping and (b) no longer saturate to exactly 0 or 1.
    """
    config = ns["config"]
    predict_scenario = ns["predict_scenario"]
    simulate_macro_paths = ns["simulate_macro_paths"]
    build_feature_matrix = ns["build_feature_matrix"]
    compute_x_train_bounds = ns["compute_x_train_bounds"]

    macro_df = simulate_macro_paths()
    X = build_feature_matrix(macro_df)
    x_train_bounds = compute_x_train_bounds(X[: config.N_TRAIN])

    extreme_scenarios = [(-3, 15), (49, 4), (-5, -5), (1000, -1000)]
    for de, dy in extreme_scenarios:
        result, err = _safe(lambda: predict_scenario(de, dy, config.BETA_TRUE, x_train_bounds))
        if err:
            return False, f"predict_scenario raised on ({de},{dy}): {err}"
        if not result["was_capped"]:
            return False, f"scenario ({de},{dy}) should have triggered capping but didn't"
        ig_stay = result["matrix"][0, 0]
        if ig_stay >= 0.999999 or ig_stay <= 0.000001:
            return False, f"scenario ({de},{dy}): IG-stay={ig_stay} still saturated even after capping"
    return True, f"{len(extreme_scenarios)} extreme scenarios all triggered capping and avoided literal 0%/100% saturation"


def check_nonfinite_scenario_input_fails_loudly(ns) -> tuple:
    """
    Regression test for a silent failure found by probing predict_scenario:
    a NaN scenario returned an all-NaN matrix while reporting NO range
    warning (NaN comparisons are always False) and was_capped=True
    (nan != nan) -- contradictory flags around garbage output. +/-inf was
    silently clamped into a plausible-looking answer, and None/text gave a
    cryptic TypeError. All must now raise a clear error; valid input must
    still work.
    """
    config = ns["config"]
    predict_scenario = ns["predict_scenario"]
    simulate_macro_paths = ns["simulate_macro_paths"]
    build_feature_matrix = ns["build_feature_matrix"]
    compute_x_train_bounds = ns["compute_x_train_bounds"]

    X = build_feature_matrix(simulate_macro_paths())
    x_train_bounds = compute_x_train_bounds(X[: config.N_TRAIN])

    bad_inputs = [float("nan"), float("inf"), float("-inf"), None, "abc", "", "0.5", True]
    for bad in bad_inputs:
        for position, args in (("earnings", (bad, 0.1)), ("yield", (0.1, bad))):
            result, err = _safe(lambda: predict_scenario(*args, config.BETA_TRUE, x_train_bounds))
            if err is None:
                return False, f"predict_scenario did NOT raise for {position}={bad!r} -- silently returned a matrix (any NaN: {bool(np.isnan(result['matrix']).any())})"
            if "finite number" not in err:
                return False, f"{position}={bad!r} raised, but with an unhelpful message: {err}"
    ok, err = _safe(lambda: predict_scenario(0.1, -0.2, config.BETA_TRUE, x_train_bounds))
    if err:
        return False, f"a valid scenario raised after adding input validation: {err}"
    return True, f"{len(bad_inputs) * 2} malformed scenario inputs (NaN/inf/None/text/bool) all raised a clear error; valid input still works"


def _config_derived_weights(ns) -> dict:
    """
    A valid model_weights-shaped dict built from config.BETA_TRUE and the
    simulated macro path's training bounds. Any valid weights do for checking
    the generator/engine LOGIC; used where the real, fitted model_weights.json
    isn't available.
    """
    config = ns["config"]
    simulate_macro_paths = ns["simulate_macro_paths"]
    build_feature_matrix = ns["build_feature_matrix"]
    compute_x_train_bounds = ns["compute_x_train_bounds"]

    X = build_feature_matrix(simulate_macro_paths())
    bounds = compute_x_train_bounds(X[: config.N_TRAIN])
    return {
        "tranches": list(config.TRANCHES),
        "default_idx": config.DEFAULT_IDX,
        "max_fall": dict(config.MAX_FALL),
        "cascade_stages": {t: list(s) for t, s in config.CASCADE_STAGES.items()},
        "upgrade_prob": dict(config.UPGRADE_PROB),
        "feature_names": list(config.FEATURE_NAMES),
        "x_train_bounds": bounds,
        "extrapolation_cap_bounds": {name: [lo * 3, hi * 3] for name, (lo, hi) in bounds.items()},
        "beta_hat": {t: {s: [float(v) for v in vec] for s, vec in stages.items()} for t, stages in config.BETA_TRUE.items()},
    }


def _bad_weight_mutations(weights: dict) -> dict:
    """
    Invalid model_weights-shaped dicts, each broken in one specific way that
    a generator template (Sheets or Excel) is not written to handle safely:
    NaN beta, a missing cascade stage, a renamed tranche, an IG upgrade prob
    > 0 (both templates hard-code IG at index 0 with nothing above it),
    inverted or missing cap bounds. Shared between the Sheets and Excel
    generator-rejection checks since both call the same underlying
    sheets_export._validate_weights.
    """
    import copy

    def mutated(fn):
        w = copy.deepcopy(weights)
        fn(w)
        return w

    return {
        "NaN in a beta vector": mutated(lambda w: w["beta_hat"]["IG"]["A"].__setitem__(1, float("nan"))),
        "missing cascade stage": mutated(lambda w: w["beta_hat"]["IG"].pop("C")),
        "renamed tranche": mutated(lambda w: w.__setitem__("tranches", ["IG", "NonIG", "HY", "Default"])),
        "IG upgrade_prob > 0": mutated(lambda w: w["upgrade_prob"].__setitem__("IG", 0.1)),
        "cap bounds lo > hi": mutated(lambda w: w["extrapolation_cap_bounds"].__setitem__("X2_delta_yield", [10.0, -2.0])),
        "missing cap bounds": mutated(lambda w: w.pop("extrapolation_cap_bounds")),
    }


def check_apps_script_generator_rejects_bad_weights(ns) -> tuple:
    """
    The Sheets engine is generated from model_weights.json. The generator must
    refuse weights the JS engine can't handle correctly and must notice if the
    embedded weights were altered after generation -- otherwise a bad re-fit
    ships silently into the deployed Sheet.
    """
    build_apps_script = ns["build_apps_script"]
    verify_apps_script = ns["verify_apps_script"]
    weights = _config_derived_weights(ns)
    script, err = _safe(lambda: build_apps_script(weights))
    if err:
        return False, f"generator rejected a valid weights dict: {err}"

    bad_cases = _bad_weight_mutations(weights)
    for label, w in bad_cases.items():
        _, err = _safe(lambda: build_apps_script(w))
        if err is None:
            return False, f"generator accepted invalid weights: {label}"

    tampered = script.replace(str(weights["beta_hat"]["IG"]["A"][0]), "-1.0", 1)
    _, err = _safe(lambda: verify_apps_script(tampered, weights))
    if err is None:
        return False, "verify_apps_script accepted a script whose embedded weights were altered"
    return True, f"generator rejected {len(bad_cases)} kinds of invalid weights and verify caught a tampered script"


def check_excel_workbook_generator_rejects_bad_weights(ns) -> tuple:
    """Excel counterpart of check_apps_script_generator_rejects_bad_weights -- same underlying validator, same bad-weight cases."""
    build_workbook = ns["build_workbook"]
    weights = _config_derived_weights(ns)
    _, err = _safe(lambda: build_workbook(weights, 8))
    if err:
        return False, f"generator rejected a valid weights dict: {err}"

    bad_cases = _bad_weight_mutations(weights)
    for label, w in bad_cases.items():
        _, err = _safe(lambda: build_workbook(w, 8))
        if err is None:
            return False, f"generator accepted invalid weights: {label}"
    _, err = _safe(lambda: build_workbook(weights, 0))
    if err is None:
        return False, "generator accepted n_quarters=0"
    return True, f"generator rejected {len(bad_cases)} kinds of invalid weights, plus n_quarters=0"


def check_excel_workbook_file_is_in_sync_with_weights(ns) -> tuple:
    """
    The Excel workbook on disk must be buildable from, and verify against,
    the current model_weights.json -- catches a re-fit that forgot to
    regenerate the workbook (verify_workbook checks the embedded weights AND
    the cumulative-chain's own internal row bounds -- see its docstring for
    the historical bug this replaced).
    """
    import os

    verify_workbook = ns["verify_workbook"]
    weights_path, excel_path = ns.get("model_weights_path"), ns.get("excel_path")
    if not (weights_path and excel_path and os.path.exists(weights_path) and os.path.exists(excel_path)):
        return None, f"SKIPPED: model_weights.json and/or the Excel workbook not found ({weights_path}, {excel_path}) -- run the pipeline first"
    import json

    with open(weights_path, encoding="utf-8") as f:
        weights = json.load(f)
    try:
        verify_workbook(excel_path, weights)
    except PermissionError:
        return None, f"SKIPPED: {excel_path} is open in Excel -- can't read it to check (this is not evidence it's stale)"
    except ValueError as e:
        return False, f"STALE or broken: {e} -- regenerate with run_pipeline.py or notebook Step 6"
    return True, "the on-disk workbook's Model_Weights sheet and cumulative-chain bounds match the current model_weights.json"


def check_excel_workbook_matches_real_excel_calculation(ns) -> tuple:
    """
    Regression fixture: these next-quarter and cumulative matrices were
    CACHED VALUES read back from credit_migration_scenario_calculator.xlsx
    after it had genuinely been opened and recalculated by a real copy of
    Excel (captured 2026-09-29, scenario deltaEarnings=0, deltaYield=5, an
    out-of-training-range but uncapped yield shock) -- not something this
    environment computed. Confirms the Python engine still reproduces a
    result Excel's own formula engine actually produced.

    IMPORTANT LIMITATION: this checks the PYTHON engine against a frozen,
    real-Excel-verified number. It does NOT re-execute the CURRENT workbook's
    formulas (nothing in this environment can) -- that the newly generated
    file's formula shapes are structurally sound is covered separately by
    check_excel_workbook_file_is_in_sync_with_weights / verify_workbook.
    """
    predict_scenario = ns["predict_scenario"]

    # the fixture was captured from the REAL fitted model_weights.json (not config.BETA_TRUE),
    # so this check needs the real weights on disk to mean anything
    import json
    import os

    weights_path = ns.get("model_weights_path")
    if not (weights_path and os.path.exists(weights_path)):
        return None, f"SKIPPED: model_weights.json not found ({weights_path}) -- the golden fixture was fit from it, not from BETA_TRUE"
    with open(weights_path, encoding="utf-8") as f:
        w = json.load(f)
    beta_hat = {t: {s: np.array(v) for s, v in stages.items()} for t, stages in w["beta_hat"].items()}

    golden_next_q = np.array([
        [0.8485368304554919, 0.021594781852819173, 0.12916996395021146, 0.0006984237414774487],
        [0.04, 0.3858874585873844, 0.154539249009178, 0.4195732924034376],
        [0.0, 0.05, 0.014031143063444385, 0.9359688569365556],
        [0.0, 0.0, 0.0, 1.0],
    ])
    golden_q10 = np.array([
        [0.19816664004524817, 0.012728768170031407, 0.03292741994782935, 0.7561771718368908],
        [0.017373913916476517, 0.0012227774614950984, 0.002927530033603499, 0.9784757785884248],
        [0.001037120718690634, 8.031393910909449e-05, 0.00017754463834539336, 0.9987050207038549],
        [0.0, 0.0, 0.0, 1.0],
    ])
    result, err = _safe(lambda: predict_scenario(0, 5, beta_hat, w["x_train_bounds"]))
    if err:
        return False, f"predict_scenario(0, 5, ...) raised, but the same scenario computed fine in real Excel: {err}"
    diff_next_q = float(np.abs(result["matrix"] - golden_next_q).max())
    if diff_next_q > 1e-9:
        return False, f"next-quarter matrix differs from the real-Excel-cached values by {diff_next_q:.3e} (weights may have been re-fit since the fixture was captured)"

    running = np.eye(4)
    for _ in range(10):
        running = running @ result["matrix"]
    diff_q10 = float(np.abs(running - golden_q10).max())
    if diff_q10 > 1e-9:
        return False, f"10-quarter cumulative matrix differs from the real-Excel-cached values by {diff_q10:.3e}"
    return True, f"Python matches real-Excel-cached values for scenario (0, 5): next-Q diff={diff_next_q:.1e}, 10Q-cumulative diff={diff_q10:.1e}"


# ---------------------------------------------------------------------------
# Symbolic evaluator for the Calc sheet's cascade-logit formulas (NOT the
# MMULT cumulative chain, checked separately above). The generator only ever
# writes one of a handful of formula shapes, so this recomputes what each
# cell's formula actually encodes, top-to-bottom (every dependency points
# strictly upward), without needing Excel -- and lets a check cross-verify it
# against the already-trusted cascade_row for a given scenario.
#
# Why this exists: a hardcoded (rather than derived-from-tranche-position)
# column offset in an earlier version of excel_export.py silently misplaced
# Junk's upgrade probability into the wrong output column -- the row/column
# WAS populated, just with the wrong quantity, so the structural checks above
# (row bounds, no missing cells) could not have caught it. Found only by the
# user opening the file in real Excel. This evaluator would have; confirmed
# via check_excel_workbook_cascade_matches_python's own self-mutation test.
#
# Kept inline (not imported from a separate module) so the whole suite stays
# copy-pasteable into the notebook's stress-test cell, same as the embedded
# Node harness above.
#
# Raises on any formula shape it doesn't recognize, rather than silently
# skipping it, so a future template change not covered here fails loudly.
# ---------------------------------------------------------------------------
import re as _re

_CALC_FORMULA_PATTERNS = [
    (_re.compile(r"^=SUMPRODUCT\(\$B\$2:\$E\$2, Model_Weights!([A-Z]\d+):([A-Z]\d+)\)$"),
     lambda m, val, mw: sum(val[f"{c}2"] * mw(f"{chr(ord(m.group(1)[0]) + i)}{m.group(1)[1:]}") for i, c in enumerate("BCDE"))),
    (_re.compile(r"^=IF\(([A-Z]\d+)>=0,1/\(1\+EXP\(-\1\)\),EXP\(\1\)/\(1\+EXP\(\1\)\)\)$"),
     lambda m, val, mw: _stable_sigmoid_py(val[m.group(1)])),
    (_re.compile(r"^=1-([A-Z]\d+)$"), lambda m, val, mw: 1.0 - val[m.group(1)]),
    (_re.compile(r"^=\(1-([A-Z]\d+)\)\*\(1-([A-Z]\d+)\)$"), lambda m, val, mw: (1 - val[m.group(1)]) * (1 - val[m.group(2)])),
    (_re.compile(r"^=\(1-([A-Z]\d+)\)\*([A-Z]\d+)$"), lambda m, val, mw: (1 - val[m.group(1)]) * val[m.group(2)]),
    (_re.compile(r"^=([A-Z]\d+)\*\(1-([A-Z]\d+)\)$"), lambda m, val, mw: val[m.group(1)] * (1 - val[m.group(2)])),
    (_re.compile(r"^=([A-Z]\d+)\*([A-Z]\d+)$"), lambda m, val, mw: val[m.group(1)] * val[m.group(2)]),
    (_re.compile(r"^=Model_Weights!([A-Z]\d+)$"), lambda m, val, mw: mw(m.group(1))),
    (_re.compile(r"^=([A-Z]\d+)$"), lambda m, val, mw: val[m.group(1)]),
]


def _stable_sigmoid_py(z: float) -> float:
    import math
    if z >= 0:
        return 1.0 / (1.0 + math.exp(-z))
    ez = math.exp(z)
    return ez / (1.0 + ez)


def _evaluate_calc_sheet(ws_calc, ws_weights, x_row) -> dict:
    """{coordinate: float} for every recognized formula/literal cell in ws_calc, given x_row=[1,x1,x2,x3] seeded at B2:E2."""
    val = {"B2": float(x_row[0]), "C2": float(x_row[1]), "D2": float(x_row[2]), "E2": float(x_row[3])}

    def model_weights_val(ref: str) -> float:
        col = ord(ref[0]) - ord("A") + 1
        row = int(ref[1:])
        cell = ws_weights.cell(row=row, column=col).value
        if not isinstance(cell, (int, float)):
            raise ValueError(f"Model_Weights!{ref} is not numeric: {cell!r}")
        return float(cell)

    for row in ws_calc.iter_rows(min_row=5, min_col=2, max_col=5):  # rows 1-4 are X-vector setup, seeded above
        for cell in row:
            coord = cell.coordinate
            if coord in val:
                continue
            v = cell.value
            if v is None:
                continue
            if isinstance(v, (int, float)):
                val[coord] = float(v)
                continue
            if not isinstance(v, str) or not v.startswith("="):
                continue  # ArrayFormula (the MMULT chain) -- out of scope here
            for pattern, fn in _CALC_FORMULA_PATTERNS:
                m = pattern.match(v)
                if m:
                    val[coord] = fn(m, val, model_weights_val)
                    break
            else:
                raise ValueError(f"{coord}: unrecognized formula shape, evaluator needs updating: {v!r}")
    return val


def _find_tranche_final_rows(ws_calc) -> dict:
    """{tranche: row} for every "<tranche> row [->...]" summary row, located by label text (not a hardcoded row number)."""
    final_row = {}
    for row in ws_calc.iter_rows(min_col=1, max_col=1):
        for cell in row:
            if isinstance(cell.value, str) and cell.value.endswith("]") and " row [" in cell.value:
                final_row[cell.value.split(" row [")[0]] = cell.row
    return final_row


def check_excel_workbook_cascade_matches_python(ns) -> tuple:
    """
    Evaluates the Calc sheet's actual cascade-logit formulas (not just their
    presence/bounds) for a spread of scenarios -- in-range, and several
    extreme/outlier ones that get clamped by capping -- and compares every
    tranche's resulting row to cascade_row, the already-trusted Python
    engine. Also deliberately corrupts one tranche's output on a freshly
    built workbook and confirms this check actually notices (self-test run
    every time, not a one-off manual confirmation) -- see the module docstring
    above for why a purely structural check (rows populated, in bounds)
    cannot catch this class of bug.
    """
    import copy

    build_workbook = ns["build_workbook"]
    cascade_row = ns["cascade_row"]
    weights = _config_derived_weights(ns)
    wb, err = _safe(lambda: build_workbook(weights, 8))
    if err:
        return False, f"build_workbook raised on valid weights: {err}"
    ws_calc, ws_weights = wb["Calc"], wb["Model_Weights"]
    tranches = list(weights["tranches"][:-1])
    final_row = _find_tranche_final_rows(ws_calc)
    if set(final_row) != set(tranches):
        return False, f"could not locate a final row for every tranche: found {sorted(final_row)}, expected {sorted(tranches)}"

    beta_hat = {t: {s: np.array(v) for s, v in stages.items()} for t, stages in weights["beta_hat"].items()}
    cap = weights["extrapolation_cap_bounds"]

    def capped_x_row(de, dy):
        x1 = min(max(de, cap["X1_delta_earnings"][0]), cap["X1_delta_earnings"][1])
        x2 = min(max(dy, cap["X2_delta_yield"][0]), cap["X2_delta_yield"][1])
        x3 = min(max(x1 * x2, cap["X3_interaction"][0]), cap["X3_interaction"][1])
        return [1.0, x1, x2, x3]

    scenarios = [(0, 0), (0.1, -0.2), (-0.5, 0.3),
                 (-3, 10), (49, -1000), (1000, 1000), (-1000, -1000)]  # last four are deliberate outliers, beyond training range
    worst, worst_at = 0.0, None
    for de, dy in scenarios:
        x_row = capped_x_row(de, dy)
        result, err = _safe(lambda: _evaluate_calc_sheet(ws_calc, ws_weights, x_row))
        if err:
            return False, f"formula evaluator failed on scenario ({de},{dy}), x_row={x_row}: {err}"
        val = result
        for tranche in tranches:
            r = final_row[tranche]
            excel_row = np.array([val[f"{c}{r}"] for c in "BCDE"])
            py_row = cascade_row(tranche, np.array(x_row), beta_hat)
            diff = float(np.abs(excel_row - py_row).max())
            if diff > worst:
                worst, worst_at = diff, (de, dy, tranche)
    if worst > 1e-9:
        de, dy, tranche = worst_at
        return False, f"Calc formulas disagree with cascade_row by {worst:.3e} for {tranche} at scenario ({de},{dy})"

    # self-mutation: reproduce the ORIGINAL bug shape (upgrade probability shifted into the wrong
    # column, deepest-fall column left blank) on a copy, and confirm the comparison above would
    # have flagged it -- proves this check has teeth every run, not just when it was written.
    mutated_wb, err = _safe(lambda: build_workbook(weights, 8))
    if err:
        return False, f"build_workbook raised on a second valid build: {err}"
    m_calc = mutated_wb["Calc"]
    junk_row = _find_tranche_final_rows(m_calc).get("Junk")
    if junk_row is None or len(weights["cascade_stages"]["Junk"]) != 1 or weights["upgrade_prob"]["Junk"] <= 0:
        return False, "expected a single-stage Junk tranche with upgrade_prob > 0 to run the self-mutation check against"
    # correctly: B=0 (literal), C=upgrade, D=stay, E=fall-to-default -- shift everything one column
    # left (the original bug) so B=upgrade, C=stay, D=fall, and E is left blank/zero
    upgrade_f, stay_f, fall_f = (m_calc.cell(row=junk_row, column=c).value for c in (3, 4, 5))
    m_calc.cell(row=junk_row, column=2, value=upgrade_f)
    m_calc.cell(row=junk_row, column=3, value=stay_f)
    m_calc.cell(row=junk_row, column=4, value=fall_f)
    m_calc.cell(row=junk_row, column=5, value=0)
    mutated_val, err = _safe(lambda: _evaluate_calc_sheet(m_calc, mutated_wb["Model_Weights"], capped_x_row(-0.1, 0.2)))
    if err:
        return False, f"formula evaluator failed evaluating the self-mutated sheet: {err}"
    mutated_row = np.array([mutated_val[f"{c}{junk_row}"] for c in "BCDE"])
    py_row = cascade_row("Junk", np.array(capped_x_row(-0.1, 0.2)), beta_hat)
    mutation_diff = float(np.abs(mutated_row - py_row).max())
    if mutation_diff < 0.05:
        return False, f"self-mutation test failed to reproduce a detectable difference (diff={mutation_diff:.2e}) -- this check may not actually be exercising the bug class it claims to"

    return True, (f"{len(scenarios)} scenarios (incl. {sum(1 for de,dy in scenarios if abs(de)>3 or abs(dy)>3)} outliers beyond training range) "
                  f"x {len(tranches)} tranches: worst Calc-vs-Python diff = {worst:.2e}; self-mutation check confirmed this test catches the historical Junk-column bug (diff={mutation_diff:.2f})")


def check_default_row_always_absorbing(ns) -> tuple:
    config = ns["config"]
    build_transition_tensor = ns["build_transition_tensor"]
    build_feature_matrix = ns["build_feature_matrix"]
    simulate_macro_paths = ns["simulate_macro_paths"]

    macro_df = simulate_macro_paths()
    X = build_feature_matrix(macro_df)
    p_true = build_transition_tensor(X, config.BETA_TRUE)
    default_rows = p_true[:, config.DEFAULT_IDX, :]
    expected = np.zeros(config.N_TRANCHES)
    expected[config.DEFAULT_IDX] = 1.0
    if not np.allclose(default_rows, expected):
        return False, "Default row is not always exactly [0,0,0,1]"
    return True, "Default row is [0,0,0,1] in all 60 quarters"


# ---------------------------------------------------------------------------
# 5. Determinism / regression
# ---------------------------------------------------------------------------

def check_same_seed_gives_identical_output(ns) -> tuple:
    simulate_macro_paths = ns["simulate_macro_paths"]
    df1 = simulate_macro_paths(seed=999)
    df2 = simulate_macro_paths(seed=999)
    if not df1.equals(df2):
        return False, "same seed produced DIFFERENT macro paths across two runs -- non-determinism"
    return True, "same seed -> byte-identical macro path across two independent runs"


# ---------------------------------------------------------------------------
# 6. Cross-seed invariance of the pipeline's own sanity checks
# ---------------------------------------------------------------------------

def check_sanity_checks_hold_across_seeds(ns, n_seeds: int = 8) -> tuple:
    config = ns["config"]
    simulate_macro_paths = ns["simulate_macro_paths"]
    build_feature_matrix = ns["build_feature_matrix"]
    build_transition_tensor = ns["build_transition_tensor"]
    engine_sanity_check = ns["engine_sanity_check"]
    validate_softmax_and_chain = ns["validate_softmax_and_chain"]

    for seed in range(n_seeds):
        macro_df = simulate_macro_paths(seed=seed * 777 + 3)
        X = build_feature_matrix(macro_df)
        p_true = build_transition_tensor(X, config.BETA_TRUE)
        sanity = engine_sanity_check(X, p_true)
        for tranche, stages in sanity.items():
            for stage, r in stages.items():
                if r["max_abs_diff"] > 1e-6:
                    return False, f"seed={seed}: engine_sanity_check failed for {tranche}-{stage}: {r['max_abs_diff']}"
        checks = validate_softmax_and_chain()
        if not all(checks.values()):
            return False, f"seed={seed}: validate_softmax_and_chain failed: {checks}"
    return True, f"engine_sanity_check + validate_softmax_and_chain held across {n_seeds} different seeds"


# ---------------------------------------------------------------------------
# 7. Generated Google Sheets engine (Code.gs) executed under Node vs. Python
#
# Apps Script itself can't run here, but its custom functions are plain
# JavaScript, so the generated Code.gs is loaded into Node (SpreadsheetApp
# stubbed) and compared with the Python engine. If Node isn't installed these
# checks report SKIPPED -- which means "not tested", never "passed".
# ---------------------------------------------------------------------------

# Loads Code.gs into an isolated vm context, runs a list of calls read from
# stdin, prints one JSON result per call. A call that doesn't return within
# TIMEOUT_MS is killed and reported as an error, which is how a hang (e.g. an
# infinite loop) surfaces as a test failure instead of freezing the run.
NODE_HARNESS_JS = r"""
// Node harness for tests/js_parity_check.py.
//
// Loads sheets/Code.gs into an isolated vm context (Apps Script has no module
// system -- everything is a top-level global, which is exactly what vm gives
// us), stubs the one Apps Script global it references at load time, then runs
// a list of calls read from stdin and prints one JSON result per call.
//
// Usage: node js_parity_harness.js <path-to-Code.gs> < calls.json > results.json
// calls.json: [{"fn": "PREDICT_MATRIX", "args": [de, dy]}, ...]
// Each result is {"ok": <return value>} or {"error": "<message>"}; a call
// that doesn't return within TIMEOUT_MS is killed and reported as an error,
// which is how a hang (e.g. an infinite loop) surfaces as a test failure
// rather than freezing the test run.

const fs = require("fs");
const vm = require("vm");

const TIMEOUT_MS = 2000;
const codePath = process.argv[2];
const calls = JSON.parse(fs.readFileSync(0, "utf8"), (k, v) => (v === "__NaN__" ? NaN : v === "__Infinity__" ? Infinity : v === "__-Infinity__" ? -Infinity : v === "__undefined__" ? undefined : v));

const ctx = vm.createContext({ Math, JSON, parseInt, isFinite, SpreadsheetApp: {} });
vm.runInContext(fs.readFileSync(codePath, "utf8"), ctx, { filename: "Code.gs" });

const results = calls.map((call) => {
  ctx.__args = call.args;
  try {
    const value = vm.runInContext(`${call.fn}.apply(null, __args)`, ctx, { timeout: TIMEOUT_MS });
    return { ok: JSON.parse(JSON.stringify(value === undefined ? null : value)) };
  } catch (e) {
    return { error: String(e && e.message ? e.message : e) };
  }
});

process.stdout.write(JSON.stringify(results));
"""

PARITY_TOL = 1e-12
CHAIN_TOL = 1e-10
PARITY_EARNINGS = [-1000, -49, -5, -3, -2.5, -2, -0.78, -0.5, 0, 0.3, 0.65, 1, 1.94, 2, 5, 49, 1000]
PARITY_YIELD = [-1000, -5, -2, -0.37, 0, 0.31, 1, 5, 10, 15, 100, 1000]
PARITY_GRID = [(de, dy) for de in PARITY_EARNINGS for dy in PARITY_YIELD]
PARITY_HORIZONS = [1, 2, 4, 12, 60, 1000]
PARITY_HORIZON_SCENARIOS = [(0, 0), (-0.5, 0.3), (-3, 15), (49, 4), (1000, -1000)]


def _skip_without_node():
    import shutil

    if shutil.which("node") is None:
        return None, "SKIPPED: `node` not found on PATH, so the generated Code.gs was NOT executed (install Node.js to run this check)"
    return None


def _run_js(script: str, calls: list) -> list:
    """Runs `calls` ([{"fn": ..., "args": [...]}]) against the Code.gs text under Node; returns [{"ok": v} | {"error": msg}]."""
    import json
    import os
    import subprocess
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        code_path = os.path.join(tmp, "Code.gs")
        harness_path = os.path.join(tmp, "harness.js")
        with open(code_path, "w", encoding="utf-8") as f:
            f.write(script)
        with open(harness_path, "w", encoding="utf-8") as f:
            f.write(NODE_HARNESS_JS)
        proc = subprocess.run(["node", harness_path, code_path], input=json.dumps(calls), capture_output=True,
                              text=True, encoding="utf-8", timeout=300)
    if proc.returncode != 0:
        raise RuntimeError(f"node harness failed:\n{proc.stderr}")
    return json.loads(proc.stdout)


def _parity_context(ns) -> dict:
    """The weights, and the Code.gs text generated from them, that the Node checks run against."""
    import json
    import os

    path = ns.get("model_weights_path")
    if path and os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            weights = json.load(f)
        source = "the fitted model_weights.json"
    else:
        weights = _config_derived_weights(ns)
        source = "BETA_TRUE-derived weights (model_weights.json not found)"
    beta_hat = {t: {s: np.array(v) for s, v in stages.items()} for t, stages in weights["beta_hat"].items()}
    return {"beta_hat": beta_hat, "bounds": weights["x_train_bounds"], "script": ns["build_apps_script"](weights), "source": source}


def check_sheets_engine_file_is_in_sync_with_weights(ns) -> tuple:
    """The Code.gs on disk must be exactly what the generator makes from the current model_weights.json."""
    import json
    import os

    weights_path, code_path = ns.get("model_weights_path"), ns.get("code_gs_path")
    if not (weights_path and code_path and os.path.exists(weights_path) and os.path.exists(code_path)):
        return None, f"SKIPPED: model_weights.json and/or Code.gs not found ({weights_path}, {code_path}) -- run the pipeline first"
    with open(weights_path, encoding="utf-8") as f:
        weights = json.load(f)
    with open(code_path, encoding="utf-8", newline="") as f:
        on_disk = f.read()
    if on_disk != ns["build_apps_script"](weights):
        try:
            ns["verify_apps_script"](on_disk, weights)
            return False, "embedded weights match model_weights.json, but the file differs from the generator output (hand-edited?) -- regenerate it"
        except ValueError as e:
            return False, f"STALE: {e} -- regenerate with run_pipeline.py or notebook Step 5"
    return True, f"Code.gs is byte-identical to the generator output for model_weights.json ({len(on_disk.splitlines())} lines)"


def check_sheets_engine_matrix_matches_python(ns) -> tuple:
    skipped = _skip_without_node()
    if skipped:
        return skipped
    ctx = _parity_context(ns)
    predict_scenario = ns["predict_scenario"]
    results = _run_js(ctx["script"], [{"fn": "PREDICT_MATRIX", "args": list(s)} for s in PARITY_GRID])
    worst, worst_at = 0.0, None
    for (de, dy), r in zip(PARITY_GRID, results):
        if "error" in r:
            return False, f"PREDICT_MATRIX({de}, {dy}) raised: {r['error']}"
        js = np.array(r["ok"], dtype=float)
        if (not np.all(np.isfinite(js)) or np.any(js < -1e-12) or np.any(js > 1 + 1e-12)
                or not np.allclose(js.sum(axis=1), 1.0, atol=1e-12)):
            return False, f"PREDICT_MATRIX({de}, {dy}) returned an invalid probability matrix: {r['ok']}"
        diff = float(np.abs(js - predict_scenario(de, dy, ctx["beta_hat"], ctx["bounds"])["matrix"]).max())
        if diff > worst:
            worst, worst_at = diff, (de, dy)
    if worst > PARITY_TOL:
        return False, f"JS vs Python max abs diff {worst:.3e} at {worst_at} exceeds {PARITY_TOL}"
    return True, f"{len(PARITY_GRID)} scenarios (in-range to +/-1000, {ctx['source']}): max abs diff JS vs Python = {worst:.2e}, every matrix valid"


def check_sheets_engine_row_and_cumulative_match_python(ns) -> tuple:
    skipped = _skip_without_node()
    if skipped:
        return skipped
    ctx = _parity_context(ns)
    predict_scenario = ns["predict_scenario"]

    tranches = ["IG", "NonIG", "Junk", "Default"]
    row_scenarios = [(0, 0), (-0.5, 0.3), (-3, 15), (49, 4)]
    row_results = iter(_run_js(ctx["script"], [{"fn": "PREDICT_ROW", "args": [t, de, dy]} for de, dy in row_scenarios for t in tranches]))
    for de, dy in row_scenarios:
        matrix = predict_scenario(de, dy, ctx["beta_hat"], ctx["bounds"])["matrix"]
        for i, t in enumerate(tranches):
            r = next(row_results)
            if "error" in r:
                return False, f"PREDICT_ROW({t}, {de}, {dy}) raised: {r['error']}"
            row = r["ok"][0] if isinstance(r["ok"][0], list) else r["ok"]
            if np.abs(np.array(row) - matrix[i]).max() > PARITY_TOL:
                return False, f"PREDICT_ROW({t}, {de}, {dy}) differs from the Python matrix row"

    calls = [{"fn": "CUMULATIVE_STEADY", "args": [de, dy, n]} for de, dy in PARITY_HORIZON_SCENARIOS for n in PARITY_HORIZONS]
    cum_results = iter(_run_js(ctx["script"], calls))
    worst = 0.0
    for de, dy in PARITY_HORIZON_SCENARIOS:
        single = predict_scenario(de, dy, ctx["beta_hat"], ctx["bounds"])["matrix"]
        running, chain = np.eye(4), {}
        for q in range(1, max(PARITY_HORIZONS) + 1):
            running = running @ single  # sequential product, as in cumulative_sequence
            if q in PARITY_HORIZONS:
                chain[q] = running.copy()
        for n in PARITY_HORIZONS:
            r = next(cum_results)
            if "error" in r:
                return False, f"CUMULATIVE_STEADY({de}, {dy}, {n}) raised: {r['error']}"
            js = np.array(r["ok"], dtype=float)
            if not np.all(np.isfinite(js)) or not np.allclose(js.sum(axis=1), 1.0, atol=1e-9):
                return False, f"CUMULATIVE_STEADY({de}, {dy}, {n}) is not a valid stochastic matrix"
            worst = max(worst, float(np.abs(js - chain[n]).max()))
    if worst > CHAIN_TOL:
        return False, f"cumulative JS vs Python max abs diff {worst:.3e} exceeds {CHAIN_TOL}"
    return True, f"{len(row_scenarios) * len(tranches)} tranche rows and {len(calls)} cumulative matrices (n up to {max(PARITY_HORIZONS)}) match Python; worst cumulative diff {worst:.2e}"


def check_sheets_engine_warning_flags_match_python(ns) -> tuple:
    skipped = _skip_without_node()
    if skipped:
        return skipped
    ctx = _parity_context(ns)
    predict_scenario = ns["predict_scenario"]
    results = _run_js(ctx["script"], [{"fn": "SCENARIO_WARNING", "args": list(s)} for s in PARITY_GRID])
    for (de, dy), r in zip(PARITY_GRID, results):
        if "error" in r:
            return False, f"SCENARIO_WARNING({de}, {dy}) raised: {r['error']}"
        py = predict_scenario(de, dy, ctx["beta_hat"], ctx["bounds"])
        msg = r["ok"]
        if bool(py["range_warnings"]) != msg.startswith("EXTRAPOLATION WARNING"):
            return False, f"({de}, {dy}): Python out-of-range={bool(py['range_warnings'])} but JS said {msg!r}"
        if py["was_capped"] != ("INPUT CAPPED" in msg):
            return False, f"({de}, {dy}): Python was_capped={py['was_capped']} but JS said {msg!r}"
    return True, f"range-warning and INPUT CAPPED flags agree with Python on all {len(PARITY_GRID)} scenarios"


def check_sheets_engine_rejects_malformed_input(ns) -> tuple:
    """
    Regression test for three bugs found by running the JavaScript under Node:
    a NaN input returned an all-NaN matrix while SCENARIO_WARNING said "OK";
    a blank/text input threw the cryptic "toFixed is not a function"; and a
    horizon of Infinity or 1e9 hung until the timeout (2.7 was also silently
    floored). All must now raise a clear, specific error.
    """
    skipped = _skip_without_node()
    if skipped:
        return skipped
    ctx = _parity_context(ns)
    nan, inf = "__NaN__", "__Infinity__"  # the harness turns these markers into real NaN/Infinity
    finite = "must be a finite number"
    horizon = "whole number between 1 and"
    cases = [
        ("blank scenario cell", "PREDICT_MATRIX", ["", 0.1], finite),
        ("text scenario", "PREDICT_MATRIX", [0.1, "abc"], finite),
        ("NaN scenario", "PREDICT_MATRIX", [nan, 0.1], finite),
        ("Infinity scenario", "PREDICT_MATRIX", [0.1, inf], finite),
        ("missing argument", "PREDICT_MATRIX", [0.1, "__undefined__"], finite),
        ("NaN into PREDICT_ROW", "PREDICT_ROW", ["IG", nan, 0.1], finite),
        ("NaN into the Default row", "PREDICT_ROW", ["Default", nan, 0.1], finite),
        ("NaN into SCENARIO_WARNING (must not say OK)", "SCENARIO_WARNING", [nan, 0.1], finite),
        ("blank into SCENARIO_WARNING", "SCENARIO_WARNING", ["", 0.1], finite),
        ("horizon = Infinity (used to hang)", "CUMULATIVE_STEADY", [0, 0, inf], horizon),
        ("horizon = 1e9 (used to hang)", "CUMULATIVE_STEADY", [0, 0, 1e9], horizon),
        ("horizon as text '4'", "CUMULATIVE_STEADY", [0, 0, "4"], horizon),
        ("horizon = 2.7 (used to floor silently)", "CUMULATIVE_STEADY", [0, 0, 2.7], horizon),
        ("horizon = 0", "CUMULATIVE_STEADY", [0, 0, 0], horizon),
        ("horizon = -1", "CUMULATIVE_STEADY", [0, 0, -1], horizon),
        ("horizon = NaN", "CUMULATIVE_STEADY", [0, 0, nan], horizon),
        ("horizon blank", "CUMULATIVE_STEADY", [0, 0, ""], horizon),
        ("horizon 1001 (over the limit)", "CUMULATIVE_STEADY", [0, 0, 1001], horizon),
        ("unknown tranche", "PREDICT_ROW", ["Bogus", 0, 0], "Unknown tranche"),
    ]
    results = _run_js(ctx["script"], [{"fn": fn, "args": args} for _, fn, args, _ in cases])
    for (label, _, _, expected), r in zip(cases, results):
        if "error" not in r:
            return False, f"{label}: did NOT raise -- silently returned {str(r['ok'])[:80]}"
        if expected not in r["error"]:
            return False, f"{label}: raised, but message {r['error']!r} lacks {expected!r}"
    return True, f"all {len(cases)} malformed-input cases raised a clear, specific error (no silent NaN, no hang)"


ALL_CHECKS = [
    check_extreme_scenarios_stay_valid,
    check_unknown_tranche_fails_loudly,
    check_all_zero_weights_do_not_silently_corrupt_fit,
    check_zero_smoothing_epsilon_breaks_as_expected,
    check_tiny_company_pool_still_produces_valid_probabilities,
    check_random_fuzzing_never_violates_invariants,
    check_capping_prevents_saturation,
    check_nonfinite_scenario_input_fails_loudly,
    check_apps_script_generator_rejects_bad_weights,
    check_excel_workbook_generator_rejects_bad_weights,
    check_default_row_always_absorbing,
    check_same_seed_gives_identical_output,
    check_sanity_checks_hold_across_seeds,
    check_sheets_engine_file_is_in_sync_with_weights,
    check_sheets_engine_matrix_matches_python,
    check_sheets_engine_row_and_cumulative_match_python,
    check_sheets_engine_warning_flags_match_python,
    check_sheets_engine_rejects_malformed_input,
    check_excel_workbook_file_is_in_sync_with_weights,
    check_excel_workbook_matches_real_excel_calculation,
    check_excel_workbook_cascade_matches_python,
]


def run_all(ns, label: str, checks=None) -> bool:
    """Runs `checks` (default: ALL_CHECKS). A check returning (None, msg) is SKIPPED: not a failure, but reported loudly as untested."""
    checks = ALL_CHECKS if checks is None else checks
    print(f"\n{'=' * 70}\nSTRESS SUITE: {label}\n{'=' * 70}")
    all_passed = True
    n_skipped = 0
    for check in checks:
        try:
            passed, message = check(ns)
        except Exception as e:  # a check itself blowing up is also a FAIL, not a crash
            passed, message = False, f"check raised unexpectedly: {type(e).__name__}: {e}"
        if passed is None:
            status = "SKIP"
            n_skipped += 1
        else:
            status = "PASS" if passed else "FAIL"
            if not passed:
                all_passed = False
        print(f"  [{status}] {check.__name__}: {message}")
    summary = "ALL PASSED" if all_passed else "AT LEAST ONE FAILURE"
    if all_passed and n_skipped:
        summary += f" ({n_skipped} check(s) SKIPPED, i.e. NOT tested -- see above)"
    print(f"\n{summary} -- {label}")
    return all_passed
