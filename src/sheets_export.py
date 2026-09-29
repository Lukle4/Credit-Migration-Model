"""
Generates the Google Sheets engine (sheets/Code.gs) from model_weights.json.

Why this exists: the Apps Script engine is a plain-JS re-implementation of the
Python cascade-logit engine, and it needs the fitted beta_hat, the training
bounds and the extrapolation caps. Pasting those in by hand meant a re-fit
could silently leave the deployed Sheet running stale weights. Here the JS
engine is a fixed template and the weights are injected from the exported
model_weights.json (whitelisted keys only), then read back out of the
generated text and compared to the source dict, so what gets pasted into
Apps Script is guaranteed to be the model that was just fitted.

Apps Script cannot run inside Jupyter -- this only GENERATES the file. It
still has to be pasted into Extensions > Apps Script (see
sheets/SETUP_GUIDE.md). tests/js_parity_check.py runs the generated file
under Node and compares it to the Python engine.
"""

import json
import math
import os
import re

# The JS template hard-codes this 4-tranche structure (matMul4_, the
# IG/NonIG/Junk loop in buildTransitionMatrix_), so any other tranche layout
# must fail here rather than produce a Sheet that silently mis-indexes.
EXPECTED_TRANCHES = ["IG", "NonIG", "Junk", "Default"]
NON_DEFAULT_TRANCHES = ["IG", "NonIG", "Junk"]

# The only keys the JS engine reads. Anything else in model_weights.json
# (notes, future additions) is deliberately NOT embedded.
EMBEDDED_KEYS = [
    "tranches", "default_idx", "max_fall", "cascade_stages", "upgrade_prob",
    "feature_names", "x_train_bounds", "extrapolation_cap_bounds", "beta_hat",
]

_WEIGHTS_PLACEHOLDER = "__MODEL_WEIGHTS_JSON__"
_WEIGHTS_BLOCK_RE = re.compile(r"// BEGIN MODEL_WEIGHTS\nvar MODEL_WEIGHTS = (.*?);\n// END MODEL_WEIGHTS", re.S)

APPS_SCRIPT_TEMPLATE = r'''/**
 * Credit Migration Rate Sensitivity Model -- Google Sheets engine.
 *
 * GENERATED FILE -- do not edit by hand. Produced by src/sheets_export.py
 * (notebook Step 5 / run_pipeline.py) from model_weights.json, so the fitted
 * weights below are always the ones from the run that generated it. To change
 * the engine logic, edit the template in src/sheets_export.py; to refresh the
 * weights, re-run the pipeline/notebook and re-paste this whole file.
 *
 * This re-implements the Python cascade-logit Markov engine (src/markov_dgp.py
 * cascade_row + src/markov_engine.py forecast_transition_tensor) natively in
 * Apps Script, driven by the beta_hat exported from model_weights.json.
 *
 * IMPORTANT UNITS NOTE (matches src/matrix_builder.py): deltaEarnings and
 * deltaYield inputs are in PERCENTAGE POINTS, i.e. type 0.5 for +0.5% (50bp),
 * not 0.005. The model was fit on this scale -- feeding decimals in in a
 * different scale will silently produce near-zero / meaningless probabilities.
 *
 * EXTRAPOLATION CAPPING (matches src/markov_engine.predict_scenario): inputs
 * far outside the training range are clamped to extrapolation_cap_bounds
 * (per-feature [lo, hi] pairs, NOT one shared multiplier -- earnings and
 * yield saturate at very different multiples of their own training range,
 * see config.py) before the matrix math runs, so a wildly extreme scenario
 * can't saturate the sigmoid into a literal, historically unprecedented
 * 0%/100%. SCENARIO_WARNING() reports when this happens.
 *
 * SETUP: paste this whole file into Extensions > Apps Script, save, then run
 * setupSheet() once from the Apps Script editor (Run button) and approve the
 * one-time authorization prompt. See SETUP_GUIDE.md for the full walkthrough.
 */

// ---------------------------------------------------------------------------
// Model weights -- injected from model_weights.json by src/sheets_export.py.
// Everything below the END marker is fixed engine code.
// ---------------------------------------------------------------------------
// BEGIN MODEL_WEIGHTS
var MODEL_WEIGHTS = __MODEL_WEIGHTS_JSON__;
// END MODEL_WEIGHTS

// ---------------------------------------------------------------------------
// Core engine (mirrors src/markov_dgp.cascade_row exactly)
// ---------------------------------------------------------------------------

/**
 * Numerically stable logistic sigmoid, matching src/markov_dgp._sigmoid.
 * The naive 1/(1+exp(-z)) form can overflow for very negative z -- a real
 * risk once a scenario far outside the training range is queried (allowed
 * through with a warning, not blocked -- see checkScenarioRange_). Keeping
 * the exponent argument always <= 0 means it can only underflow toward 0
 * (safe), never overflow.
 */
function sigmoid_(z) {
  if (z >= 0) return 1.0 / (1.0 + Math.exp(-z));
  var expZ = Math.exp(z);
  return expZ / (1.0 + expZ);
}

function buildFeatureRow_(deltaEarnings, deltaYield) {
  var x1 = deltaEarnings;
  var x2 = deltaYield;
  return [1, x1, x2, x1 * x2];
}

function dot_(x, beta) {
  var s = 0;
  for (var i = 0; i < x.length; i++) s += x[i] * beta[i];
  return s;
}

// Longest horizon CUMULATIVE_STEADY will chain. Only exists so a typo like
// 1e9 fails with a clear error instead of hanging the sheet until Apps
// Script's custom-function timeout.
var MAX_HORIZON_QUARTERS = 1000;

/** Human-readable description of a bad argument, for error messages. */
function describeValue_(value) {
  if (value === "") return "an empty cell";
  if (typeof value === "string") return "text '" + value + "'";
  return String(value);
}

/**
 * Throws a clear error unless value is a finite number. Without this, a
 * blank cell / text / NaN flows through Math.max/Math.min and the sigmoid and
 * either produces an all-NaN matrix (with SCENARIO_WARNING reporting "OK") or
 * a cryptic "toFixed is not a function" -- both found by running this
 * engine under Node.
 */
function assertFiniteNumber_(value, label) {
  if (typeof value !== "number" || !isFinite(value)) {
    throw new Error(label + " must be a finite number (got " + describeValue_(value) +
      "). Enter it in percentage points, e.g. 0.5 for +0.5%.");
  }
}

function assertValidScenario_(deltaEarnings, deltaYield) {
  assertFiniteNumber_(deltaEarnings, "Delta Earnings");
  assertFiniteNumber_(deltaYield, "Delta Yield");
}

/** Throws a clear, Sheets-visible error for a misspelled/unknown tranche name. */
function assertValidTranche_(tranche) {
  if (MODEL_WEIGHTS.tranches.indexOf(tranche) === -1) {
    throw new Error("Unknown tranche '" + tranche + "'. Must be exactly one of: " + MODEL_WEIGHTS.tranches.join(", "));
  }
}

/**
 * Warns (does not block) when a scenario falls outside the observed
 * training range -- the model is linear-in-the-logit and was fit on macro
 * deltas roughly in the +/-1 percentage-point range; querying it far
 * outside that is an extrapolation the fit was never validated on. Mirrors
 * src/markov_engine.assess_scenario_range. Always evaluated against the RAW
 * (uncapped) input, so it still tells you honestly how extreme your typed
 * scenario was, even though the matrix math itself uses the capped version.
 * @return Array of warning strings, empty if the scenario is in range.
 */
function checkScenarioRange_(deltaEarnings, deltaYield) {
  assertValidScenario_(deltaEarnings, deltaYield);
  var xRow = buildFeatureRow_(deltaEarnings, deltaYield);
  var names = MODEL_WEIGHTS.feature_names.slice(1); // skip intercept
  var warnings = [];
  for (var i = 0; i < names.length; i++) {
    var bounds = MODEL_WEIGHTS.x_train_bounds[names[i]];
    var val = xRow[i + 1];
    if (val < bounds[0] || val > bounds[1]) {
      warnings.push(names[i] + "=" + val.toFixed(3) + " is outside training range [" +
        bounds[0].toFixed(3) + ", " + bounds[1].toFixed(3) + "] -- extrapolation");
    }
  }
  return warnings;
}

/**
 * Clamps X1/X2 to their resolved extrapolation_cap_bounds (per-feature
 * [lo, hi] pairs -- NOT one shared multiplier, see MODEL_WEIGHTS and
 * config.py for why each was chosen independently: earnings and the
 * interaction term are multiples of their own training range, yield is an
 * explicit literal range chosen from where the fitted model is actually
 * still informative). Recomputes X3 = X1*X2 from the ALREADY-capped X1/X2
 * and additionally clamps X3 itself to its own resolved bounds. Matches
 * src/markov_engine.cap_scenario_to_bounds exactly.
 *
 * Fixes a real issue found by stress-testing this deliverable: a wildly
 * out-of-range scenario pushed the sigmoid into full saturation, returning
 * a literal 0%/100% -- e.g. "100% of IG demotes in one quarter", which has
 * no historical precedent however "correct" it is for a linear model
 * extrapolated that far. Capping keeps severe-but-plausible stress
 * scenarios expressive while preventing that saturation.
 * @return {xRow: number[], cappedDims: string[]} -- cappedDims is empty if
 *   the scenario was already within the cap.
 */
function capScenarioToBounds_(deltaEarnings, deltaYield) {
  assertValidScenario_(deltaEarnings, deltaYield);
  var b1 = MODEL_WEIGHTS.extrapolation_cap_bounds.X1_delta_earnings;
  var b2 = MODEL_WEIGHTS.extrapolation_cap_bounds.X2_delta_yield;
  var b3 = MODEL_WEIGHTS.extrapolation_cap_bounds.X3_interaction;

  var x1 = Math.min(Math.max(deltaEarnings, b1[0]), b1[1]);
  var x2 = Math.min(Math.max(deltaYield, b2[0]), b2[1]);
  var x3Raw = x1 * x2;
  var x3 = Math.min(Math.max(x3Raw, b3[0]), b3[1]);

  var cappedDims = [];
  if (x1 !== deltaEarnings) cappedDims.push("Delta Earnings: " + deltaEarnings.toFixed(3) + " -> " + x1.toFixed(3) + " (cap [" + b1[0].toFixed(3) + ", " + b1[1].toFixed(3) + "])");
  if (x2 !== deltaYield) cappedDims.push("Delta Yield: " + deltaYield.toFixed(3) + " -> " + x2.toFixed(3) + " (cap [" + b2[0].toFixed(3) + ", " + b2[1].toFixed(3) + "])");
  if (x3 !== x3Raw) cappedDims.push("Interaction: " + x3Raw.toFixed(3) + " -> " + x3.toFixed(3) + " (cap [" + b3[0].toFixed(3) + ", " + b3[1].toFixed(3) + "])");

  return {xRow: [1, x1, x2, x3], cappedDims: cappedDims};
}

/** Decodes one tranche's cascading logit stages into a full 1x4 probability row. */
function cascadeRow_(tranche, xRow) {
  var idx = MODEL_WEIGHTS.tranches.indexOf(tranche);
  var u = MODEL_WEIGHTS.upgrade_prob[tranche];
  var maxFall = MODEL_WEIGHTS.max_fall[tranche];
  var betas = MODEL_WEIGHTS.beta_hat[tranche];

  var pFallGe1 = sigmoid_(dot_(xRow, betas.A));
  var pStay = 1.0 - pFallGe1;

  var pFallEq = {};
  if (maxFall === 1) {
    pFallEq[1] = pFallGe1;
  } else {
    var pFallGe2Given = sigmoid_(dot_(xRow, betas.B));
    pFallEq[1] = pFallGe1 * (1.0 - pFallGe2Given);
    var pFallGe2 = pFallGe1 * pFallGe2Given;
    if (maxFall === 2) {
      pFallEq[2] = pFallGe2;
    } else {
      var pEq3Given = sigmoid_(dot_(xRow, betas.C));
      pFallEq[2] = pFallGe2 * (1.0 - pEq3Given);
      pFallEq[3] = pFallGe2 * pEq3Given;
    }
  }

  var row = [0, 0, 0, 0];
  if (u > 0) row[idx - 1] = u;
  row[idx] += (1.0 - u) * pStay;
  for (var k in pFallEq) {
    row[idx + parseInt(k, 10)] += (1.0 - u) * pFallEq[k];
  }
  return row;
}

/** Full 4x4 transition matrix for a given quarter's macro scenario (capped -- see capScenarioToBounds_). */
function buildTransitionMatrix_(deltaEarnings, deltaYield) {
  var xRow = capScenarioToBounds_(deltaEarnings, deltaYield).xRow;
  var matrix = [];
  for (var i = 0; i < 4; i++) matrix.push([0, 0, 0, 0]);
  ["IG", "NonIG", "Junk"].forEach(function (tranche) {
    var idx = MODEL_WEIGHTS.tranches.indexOf(tranche);
    matrix[idx] = cascadeRow_(tranche, xRow);
  });
  matrix[MODEL_WEIGHTS.default_idx][MODEL_WEIGHTS.default_idx] = 1.0;
  return matrix;
}

function matMul4_(A, B) {
  var out = [];
  for (var i = 0; i < 4; i++) {
    var row = [];
    for (var j = 0; j < 4; j++) {
      var s = 0;
      for (var k = 0; k < 4; k++) s += A[i][k] * B[k][j];
      row.push(s);
    }
    out.push(row);
  }
  return out;
}

// ---------------------------------------------------------------------------
// Custom spreadsheet functions (usable directly as =PREDICT_ROW(...) etc.)
// ---------------------------------------------------------------------------

/**
 * Predicted transition row for one tranche under a given macro scenario.
 * @param {string} tranche One of "IG", "NonIG", "Junk", "Default".
 * @param {number} deltaEarnings Change in earnings growth, in percentage points (0.5 = +0.5%).
 * @param {number} deltaYield Change in 10Y yield, in percentage points (0.25 = +25bp).
 * @return A 1x4 array of probabilities [->IG, ->NonIG, ->Junk, ->Default].
 * @customfunction
 */
function PREDICT_ROW(tranche, deltaEarnings, deltaYield) {
  assertValidTranche_(tranche);
  assertValidScenario_(deltaEarnings, deltaYield);
  if (tranche === "Default") return [0, 0, 0, 1];
  var xRow = capScenarioToBounds_(deltaEarnings, deltaYield).xRow;
  return [cascadeRow_(tranche, xRow)];
}

/**
 * Full 4x4 predicted transition matrix for a given macro scenario (one quarter ahead).
 * @param {number} deltaEarnings Change in earnings growth, in percentage points.
 * @param {number} deltaYield Change in 10Y yield, in percentage points.
 * @return A 4x4 array, rows/cols ordered [IG, NonIG, Junk, Default].
 * @customfunction
 */
function PREDICT_MATRIX(deltaEarnings, deltaYield) {
  return buildTransitionMatrix_(deltaEarnings, deltaYield);
}

/**
 * N-quarter cumulative transition matrix, assuming the SAME macro scenario
 * persists every quarter for n quarters (a sustained-stress what-if). Uses
 * sequential chain multiplication, matching src/markov_engine.cumulative_sequence.
 * @param {number} deltaEarnings Change in earnings growth, in percentage points.
 * @param {number} deltaYield Change in 10Y yield, in percentage points.
 * @param {number} nQuarters Number of quarters to chain (whole number, 1 to 1000).
 * @return A 4x4 array, rows/cols ordered [IG, NonIG, Junk, Default].
 * @customfunction
 */
function CUMULATIVE_STEADY(deltaEarnings, deltaYield, nQuarters) {
  if (typeof nQuarters !== "number" || !isFinite(nQuarters) || Math.floor(nQuarters) !== nQuarters ||
      nQuarters < 1 || nQuarters > MAX_HORIZON_QUARTERS) {
    throw new Error("nQuarters must be a whole number between 1 and " + MAX_HORIZON_QUARTERS +
      " (got " + describeValue_(nQuarters) + ")");
  }
  var n = nQuarters;
  var single = buildTransitionMatrix_(deltaEarnings, deltaYield);
  var result = single;
  for (var q = 1; q < n; q++) {
    result = matMul4_(result, single);
  }
  return result;
}

/**
 * Human-readable check of whether a scenario is inside the range the model
 * was actually fit on, AND whether the input(s) got capped before the
 * matrix math ran (see capScenarioToBounds_). Not required for the matrix
 * math to run -- purely informational, put next to the scenario inputs so
 * an extreme what-if doesn't look as confident as an in-range one, and so
 * you know the displayed matrix reflects the CAPPED scenario, not your
 * literal typed value, whenever capping kicked in.
 * @param {number} deltaEarnings Change in earnings growth, in percentage points.
 * @param {number} deltaYield Change in 10Y yield, in percentage points.
 * @return A text warning, or "OK - within training range".
 * @customfunction
 */
function SCENARIO_WARNING(deltaEarnings, deltaYield) {
  var warnings = checkScenarioRange_(deltaEarnings, deltaYield);
  var capped = capScenarioToBounds_(deltaEarnings, deltaYield).cappedDims;
  if (warnings.length === 0) return "OK - within training range";
  var msg = "EXTRAPOLATION WARNING: " + warnings.join("; ");
  if (capped.length > 0) {
    msg += " | INPUT CAPPED before computing (result reflects the capped value, not your typed input): " + capped.join("; ");
  }
  return msg;
}

// ---------------------------------------------------------------------------
// One-click UI setup. Run this once from the Apps Script editor (Run button)
// after pasting this file in. Builds a "Scenario" sheet with input cells,
// both output grids, a range-check warning cell, and a heatmap
// conditional-formatting color scale.
// ---------------------------------------------------------------------------

function setupSheet() {
  var ss = SpreadsheetApp.getActiveSpreadsheet();
  var sheet = ss.getSheetByName("Scenario");
  if (sheet) ss.deleteSheet(sheet);
  sheet = ss.insertSheet("Scenario");

  sheet.getRange("A1").setValue("Credit Migration Scenario Calculator").setFontSize(16).setFontWeight("bold");

  // --- Inputs ---
  sheet.getRange("A3").setValue("Inputs (percentage points, e.g. 0.5 = +0.5%)").setFontWeight("bold");
  sheet.getRange("A4").setValue("Delta Earnings Growth");
  sheet.getRange("A5").setValue("Delta 10Y Yield");
  sheet.getRange("A6").setValue("Cumulative Horizon (quarters)");
  sheet.getRange("C4").setValue(0).setBackground("#fff3cd").setBorder(true, true, true, true, false, false);
  sheet.getRange("C5").setValue(0).setBackground("#fff3cd").setBorder(true, true, true, true, false, false);
  sheet.getRange("C6").setValue(4).setBackground("#fff3cd").setBorder(true, true, true, true, false, false);

  sheet.getRange("A7").setValue("Scenario check");
  sheet.getRange("C7").setFormula("=SCENARIO_WARNING($C$4,$C$5)").setFontStyle("italic");

  // --- Next-quarter matrix ---
  var headers = ["", "-> IG", "-> NonIG", "-> Junk", "-> Default"];
  var rowLabels = ["IG ->", "NonIG ->", "Junk ->", "Default ->"];

  sheet.getRange("A9").setValue("Next-Quarter Transition Matrix").setFontWeight("bold");
  sheet.getRange("A10:E10").setValues([headers]).setFontWeight("bold");
  sheet.getRange("A11:A14").setValues(rowLabels.map(function (r) { return [r]; })).setFontWeight("bold");
  sheet.getRange("B11").setFormula("=PREDICT_MATRIX($C$4,$C$5)");
  sheet.getRange("B11:E14").setNumberFormat("0.00%");

  // --- Cumulative matrix ---
  sheet.getRange("A17").setValue("N-Quarter Cumulative Transition Matrix (sustained scenario)").setFontWeight("bold");
  sheet.getRange("A18:E18").setValues([headers]).setFontWeight("bold");
  sheet.getRange("A19:A22").setValues(rowLabels.map(function (r) { return [r]; })).setFontWeight("bold");
  sheet.getRange("B19").setFormula("=CUMULATIVE_STEADY($C$4,$C$5,$C$6)");
  sheet.getRange("B19:E22").setNumberFormat("0.00%");

  // --- Heatmap conditional formatting on both grids ---
  [["B11:E14"], ["B19:E22"]].forEach(function (rangeA1) {
    var range = sheet.getRange(rangeA1[0]);
    var rule = SpreadsheetApp.newConditionalFormatRule()
      .setGradientMaxpointWithValue("#d73027", SpreadsheetApp.InterpolationType.NUMBER, "1")
      .setGradientMidpointWithValue("#ffffbf", SpreadsheetApp.InterpolationType.NUMBER, "0.5")
      .setGradientMinpointWithValue("#1a9850", SpreadsheetApp.InterpolationType.NUMBER, "0")
      .setRanges([range])
      .build();
    var rules = sheet.getConditionalFormatRules();
    rules.push(rule);
    sheet.setConditionalFormatRules(rules);
  });

  sheet.setColumnWidths(1, 5, 110);
  ss.setActiveSheet(sheet);
  SpreadsheetApp.flush();
}
'''


def _is_finite_number(v) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)


def _validate_weights(w: dict) -> None:
    """Raises ValueError for anything the JS template can't handle correctly."""
    missing = [k for k in EMBEDDED_KEYS if k not in w]
    if missing:
        raise ValueError(f"model weights are missing required keys: {missing}")
    if list(w["tranches"]) != EXPECTED_TRANCHES:
        raise ValueError(f"tranches must be exactly {EXPECTED_TRANCHES} (the JS engine hard-codes this layout), got {w['tranches']}")
    if w["default_idx"] != EXPECTED_TRANCHES.index("Default"):
        raise ValueError(f"default_idx must be {EXPECTED_TRANCHES.index('Default')}, got {w['default_idx']}")

    feature_names = list(w["feature_names"])
    if len(feature_names) != 4:
        raise ValueError(f"feature_names must have 4 entries (intercept + X1, X2, X3), got {feature_names}")
    macro_features = feature_names[1:]

    for block in ("x_train_bounds", "extrapolation_cap_bounds"):
        if sorted(w[block]) != sorted(macro_features):
            raise ValueError(f"{block} keys must be exactly {macro_features}, got {sorted(w[block])}")
        for name, pair in w[block].items():
            if len(pair) != 2 or not all(_is_finite_number(v) for v in pair) or pair[0] > pair[1]:
                raise ValueError(f"{block}[{name!r}] must be a finite [lo, hi] pair with lo <= hi, got {pair}")

    for idx, tranche in enumerate(NON_DEFAULT_TRANCHES):
        stages = list(w["cascade_stages"][tranche])
        if len(stages) != w["max_fall"][tranche]:
            raise ValueError(f"{tranche}: {len(stages)} cascade stages but max_fall={w['max_fall'][tranche]}")
        # Both the JS (row[idx+k]) and Excel (column 2+idx+k) engines index the cascade's deepest
        # fall directly by tranche position -- a max_fall deep enough to overshoot Default would
        # silently write past the last real target column/row instead of raising. Not just a Junk/NonIG
        # coincidence: this holds for any tranche count, since idx + max_fall must land at Default's
        # index at the deepest, never beyond it.
        default_idx = len(EXPECTED_TRANCHES) - 1
        if idx + w["max_fall"][tranche] > default_idx:
            raise ValueError(f"{tranche} (position {idx}): max_fall={w['max_fall'][tranche]} would fall past Default "
                              f"(position {default_idx}) -- a tranche can fall at most {default_idx - idx} notches")
        if sorted(w["beta_hat"][tranche]) != sorted(stages):
            raise ValueError(f"{tranche}: beta_hat stages {sorted(w['beta_hat'][tranche])} != cascade_stages {sorted(stages)}")
        for stage, vec in w["beta_hat"][tranche].items():
            if len(vec) != len(feature_names) or not all(_is_finite_number(v) for v in vec):
                raise ValueError(f"beta_hat[{tranche!r}][{stage!r}] must be {len(feature_names)} finite numbers, got {vec}")
        u = w["upgrade_prob"][tranche]
        if not (_is_finite_number(u) and 0 <= u < 1):
            raise ValueError(f"upgrade_prob[{tranche!r}] must be in [0, 1), got {u}")
    if w["upgrade_prob"]["IG"] != 0:
        raise ValueError("upgrade_prob['IG'] must be 0 -- nothing sits above IG, and the JS engine would write to row index -1")


def _embedded_subset(model_weights: dict) -> dict:
    return {k: model_weights[k] for k in EMBEDDED_KEYS}


def extract_embedded_weights(script: str) -> dict:
    """Reads the MODEL_WEIGHTS object back out of a generated Code.gs."""
    m = _WEIGHTS_BLOCK_RE.search(script)
    if not m:
        raise ValueError("no BEGIN/END MODEL_WEIGHTS block found in the script")
    return json.loads(m.group(1))


def verify_apps_script(script: str, model_weights: dict) -> bool:
    """Raises ValueError unless the weights embedded in `script` exactly equal `model_weights`."""
    expected = json.loads(json.dumps(_embedded_subset(model_weights), allow_nan=False))
    embedded = extract_embedded_weights(script)
    if embedded != expected:
        differing = [k for k in EMBEDDED_KEYS if embedded.get(k) != expected.get(k)]
        raise ValueError(f"embedded MODEL_WEIGHTS does not match the source weights (differs in: {differing})")
    return True


def _compact_json(obj, level: int = 0) -> str:
    """JSON with one dict key per line but number lists kept on one line (a beta vector reads as one row)."""
    if isinstance(obj, dict):
        pad = "  " * (level + 1)
        items = [f"{pad}{json.dumps(k)}: {_compact_json(v, level + 1)}" for k, v in obj.items()]
        return "{\n" + ",\n".join(items) + "\n" + "  " * level + "}"
    return json.dumps(obj, allow_nan=False)


def build_apps_script(model_weights: dict) -> str:
    """Returns the full Code.gs text with `model_weights` (the model_weights.json dict) injected."""
    _validate_weights(model_weights)
    weights_js = _compact_json(_embedded_subset(model_weights))
    script = APPS_SCRIPT_TEMPLATE.replace(_WEIGHTS_PLACEHOLDER, weights_js)
    verify_apps_script(script, model_weights)
    return script


def export_apps_script(weights_path: str, out_path: str) -> str:
    """Reads model_weights.json, writes the generated Code.gs to out_path, and returns its text."""
    with open(weights_path, encoding="utf-8") as f:
        model_weights = json.load(f)
    script = build_apps_script(model_weights)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8", newline="\n") as f:
        f.write(script)
    return script
