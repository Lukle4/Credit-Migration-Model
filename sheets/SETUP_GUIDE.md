# Setup guide: deploying the model to Google Sheets

Takes about 5 minutes, one-time.

## 1. Create the spreadsheet

1. Go to [sheets.google.com](https://sheets.google.com) and create a new blank spreadsheet.
2. Rename it (top-left title) to something like "Credit Migration Model".

## 2. Paste in the script

1. In the menu bar: **Extensions → Apps Script**. This opens the Apps Script editor in a new tab.
2. You'll see a default file `Code.gs` with a blank/boilerplate `function myFunction() {}`. Select all its contents (Ctrl+A) and delete.
3. Open [`Code.gs`](Code.gs) from this project folder (in Colab: copy the script printed by the notebook's Step 5 cell, or use the `Code.gs` it downloads), copy its entire contents, and paste into the empty Apps Script editor. This file is generated from `model_weights.json`, so don't edit it by hand (see section 5).
4. Click the save icon (or Ctrl+S). If prompted for a project name, call it anything (e.g. "Credit Migration Engine").

## 3. Run the one-click setup

1. At the top of the Apps Script editor, there's a function dropdown (next to the "Debug" button) — select **setupSheet**.
2. Click **Run**.
3. Google will show an **Authorization required** dialog. This is expected — it's a personal script running under your own account, not a published/verified app, so Google shows a generic warning every time. Click **Review permissions** → choose your Google account → you'll likely see "Google hasn't verified this app" → click **Advanced** → **Go to [your project name] (unsafe)** → **Allow**.
   - This step has to be done by you personally, in your own Google account — it can't be done on your behalf.
4. Switch back to your spreadsheet tab. A new **"Scenario"** sheet should now exist, pre-built with input cells, a scenario range-check cell, two 4×4 matrices, and heatmap coloring.

If `setupSheet` errors out, it's almost always the authorization step above not having completed — run it again after allowing access.

## 4. Using it

On the **Scenario** sheet:

- **C4** = Δ Earnings Growth, **C5** = Δ 10Y Yield — both in **percentage points** (type `0.5` for +0.5%, i.e. +50bp — not `0.005`). This matches exactly how the Python model was fit; entering raw decimals will silently give near-meaningless (all-probability-mass-on-"stay") results.
- **C6** = number of quarters for the cumulative view (assumes the same scenario persists every quarter — a sustained-stress what-if, not a specific quarter-by-quarter path).
- **C7 ("Scenario check")** tells you whether C4/C5 are inside the range the model was actually fit on. It'll say `OK - within training range`, or flag which input is an extrapolation (e.g. `Δ yield = 5.000 is outside training range [-0.37, 0.31]`). The matrices below still compute a result either way — this is a warning, not a block — but a flagged scenario's output should be treated as far less reliable than an in-range one. The model is linear-in-the-logit and was only ever validated on macro deltas roughly in the ±1 percentage-point range.
  - If your input is *very* extreme, C7 will also show an **"INPUT CAPPED"** note. Each input is clamped to its own bound before the matrix math runs — Δ Earnings to roughly `[-2.34, 1.94]` (3x its training range), Δ Yield to an explicit `[-2.0, 10.0]`, and the interaction term to roughly `[-3.60, 0.67]` — otherwise the sigmoid saturates and returns a literal, historically-impossible 0%/100% (e.g. "100% of IG demotes in one quarter"). These aren't one shared multiplier: earnings and yield saturate at very different points because their fitted sensitivities differ, so each was tuned separately to where the model is still actually informative (see `Model_Weights` tab in the Excel version, or `extrapolation_cap_bounds` in `model_weights.json`, for the exact current values). The displayed matrix reflects the *capped* scenario, not your literal typed value, whenever this note appears.
- The **"Next-Quarter Transition Matrix"** grid (rows 10–14) shows the one-quarter-ahead forecast under that scenario, colour-scaled green (low probability) to red (high probability).
- The **"N-Quarter Cumulative Transition Matrix"** grid (rows 18–22) shows the same scenario compounded over C6 quarters.
- Try it: set C5 (Δ yield) to `1.5` (a sharp +150bp rate shock) and watch the Junk row's "→ Default" cell jump — and notice C7 will flag it as out of range, since that's a much bigger shock than anything in the training data.

You can also call the four custom functions directly in any cell, e.g. `=PREDICT_ROW("Junk", -0.3, 1.0)`. They fail loudly rather than returning garbage: a misspelled tranche name (must be exactly `"IG"`, `"NonIG"`, `"Junk"` or `"Default"`), a blank/text/NaN/infinite scenario input, or a quarters argument that isn't a whole number from 1 to 1000 each throws an error whose message says what was wrong (hover the red error cell to read it).

## 5. Refreshing the model later

`Code.gs` is generated, so refreshing means regenerating it, never pasting numbers into it by hand:

1. Re-run `python run_pipeline.py` (or the notebook through Step 5). This rewrites `model_weights.json` and regenerates `sheets/Code.gs` with the new weights, after checking that the weights embedded in the script are identical to the ones just fitted.
2. In the Apps Script editor, select everything in `Code.gs`, delete it, and paste in the regenerated file. Save.
3. The sheet's formulas pick up the new weights the next time they recalculate. There is no need to re-run `setupSheet()` unless you want to rebuild the layout from scratch (which wipes and recreates the "Scenario" sheet).

## Testing status

The custom functions' math (`PREDICT_ROW`, `PREDICT_MATRIX`, `CUMULATIVE_STEADY`, `SCENARIO_WARNING`) was run under Node and matches the Python engine to about 2e-16 across 204 scenarios (`python tests/js_parity_check.py`). `setupSheet()`, which builds the layout, number formats and heatmap, has **not** been run in Apps Script, because it needs Google's `SpreadsheetApp`. If it errors when you run it, the error message is the useful thing to bring back.
