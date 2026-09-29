"""
Step 4: Markov engine and prediction.

- Softmax/sigmoid forecast: Y_hat_test = X_test . beta_hat, converted to a
  probability row per tranche via the same cascade decode used for Ptrue
  (markov_dgp.cascade_row), since beta_hat has the identical
  {tranche: {stage: 4-vector}} shape as beta_true.
- Validated against a known, hand-computable 2-state chain before being
  trusted on the real 4-state cascade.
- Multi-period cumulative transition probability: because each quarter's
  matrix is macro-varying (non-homogeneous), the n-period cumulative matrix
  is the SEQUENTIAL product P(t1) @ P(t2) @ ... @ P(tn), NOT P(t)^n -- using
  a matrix power would silently assume a constant/homogeneous chain, which
  this model explicitly is not.
- Evaluation: MSE and Frobenius norm between the true and forecasted
  cumulative matrices on the held-out test quarters (Ptrue is used here for
  the first and only time in the whole pipeline -- pure evaluation, never
  fitting).
- Extrapolation guard: beta_hat is a linear-in-the-logit model fit on macro
  deltas roughly in the +/-1 percentage-point range. Nothing stops it from
  being queried on a wildly larger scenario (a real risk once this is wired
  into the Sheets scenario calculator, where a user can type anything) --
  it will still return a confident-looking probability, but it's an
  extrapolation the model was never validated on. assess_scenario_range()
  flags this without blocking the forecast, since a deliberate stress-test
  scenario may be exactly what's being explored.
"""

import json

import numpy as np

import config
from markov_dgp import cascade_row


def chain_multiply(matrices: list) -> np.ndarray:
    """Sequential matrix product P(t1) @ P(t2) @ ... @ P(tn)."""
    result = matrices[0]
    for m in matrices[1:]:
        result = result @ m
    return result


def validate_softmax_and_chain() -> dict:
    """
    Sanity checks the primitives against hand-verifiable cases before they
    are trusted on the real cascade / non-homogeneous chain.
    """
    checks = {}

    # 1. sigmoid(logit(p)) recovers p exactly, for a simple known probability.
    p = 0.3
    logit = np.log(p / (1 - p))
    recovered = 1.0 / (1.0 + np.exp(-logit))
    checks["sigmoid_inverts_logit"] = bool(np.isclose(recovered, p, atol=1e-12))

    # 2. A known, simple 2-state homogeneous chain: chain-multiplying n copies
    #    of P must equal P @ P @ ... (n times), cross-checked against
    #    np.linalg.matrix_power as an independent reference implementation.
    P = np.array([[0.9, 0.1], [0.2, 0.8]])
    n = 5
    via_chain_multiply = chain_multiply([P] * n)
    via_matrix_power = np.linalg.matrix_power(P, n)
    checks["chain_multiply_matches_matrix_power"] = bool(np.allclose(via_chain_multiply, via_matrix_power))

    # 3. Rows of the 2-state chain result must still sum to 1 (probability simplex preserved).
    checks["chain_result_rows_sum_to_one"] = bool(np.allclose(via_chain_multiply.sum(axis=1), 1.0))

    return checks


def forecast_transition_tensor(X: np.ndarray, beta_hat: dict) -> np.ndarray:
    """Same shape/semantics as markov_dgp.build_transition_tensor, but driven by beta_hat."""
    n = X.shape[0]
    tensor = np.zeros((n, config.N_TRANCHES, config.N_TRANCHES))
    for t in range(n):
        x_row = X[t]
        for tranche in ["IG", "NonIG", "Junk"]:
            idx = config.TRANCHES.index(tranche)
            tensor[t, idx, :] = cascade_row(tranche, x_row, beta_hat)
        tensor[t, config.DEFAULT_IDX, config.DEFAULT_IDX] = 1.0
    return tensor


def cumulative_sequence(matrix_tensor: np.ndarray) -> np.ndarray:
    """Returns shape (n, 4, 4): P_1, P_1@P_2, P_1@P_2@P_3, ..., i.e. P_hat_1..P_hat_n."""
    n = matrix_tensor.shape[0]
    out = np.zeros_like(matrix_tensor)
    running = np.eye(config.N_TRANCHES)
    for t in range(n):
        running = running @ matrix_tensor[t]
        out[t] = running
    return out


def evaluate_cumulative(p_true_cum: np.ndarray, p_forecast_cum: np.ndarray) -> dict:
    n = p_true_cum.shape[0]
    per_period = []
    for t in range(n):
        diff = p_true_cum[t] - p_forecast_cum[t]
        mse = float(np.mean(diff ** 2))
        frob = float(np.linalg.norm(diff, ord="fro"))
        per_period.append({"quarter_offset": t + 1, "mse": mse, "frobenius_norm": frob})

    overall_diff = p_true_cum - p_forecast_cum
    overall_mse = float(np.mean(overall_diff ** 2))
    overall_frob = float(np.linalg.norm(overall_diff.reshape(n, -1), ord="fro"))

    return {"per_period": per_period, "overall_mse": overall_mse, "overall_frobenius_norm": overall_frob}


def beta_dict_to_json(beta_dict: dict) -> dict:
    return {tr: {st: vec.tolist() for st, vec in stages.items()} for tr, stages in beta_dict.items()}


def compute_x_train_bounds(X_train: np.ndarray) -> dict:
    """
    Observed min/max of X_train's macro columns (skips X0, the intercept,
    which is always 1). Exported alongside beta_hat so any downstream
    consumer -- the Python forecast functions here, or the Sheets scenario
    calculator -- can tell whether a queried scenario is inside the range
    the model was actually fit on.
    """
    bounds = {}
    for i, name in enumerate(config.FEATURE_NAMES[1:], start=1):
        bounds[name] = [float(X_train[:, i].min()), float(X_train[:, i].max())]
    return bounds


def assess_scenario_range(x_row: np.ndarray, x_train_bounds: dict) -> list:
    """
    Returns a list of human-readable warnings (empty if the scenario is
    within the observed training range). Does not block the forecast --
    an extreme scenario may be a deliberate stress test -- just makes the
    extrapolation visible instead of silent.
    """
    warnings = []
    for i, name in enumerate(config.FEATURE_NAMES[1:], start=1):
        lo, hi = x_train_bounds[name]
        val = float(x_row[i])
        if val < lo or val > hi:
            warnings.append(f"{name}={val:.3f} is outside the observed training range [{lo:.3f}, {hi:.3f}] -- this forecast is an extrapolation")
    return warnings


def compute_extrapolation_cap_bounds(x_train_bounds: dict) -> dict:
    """
    Resolves config's per-feature cap settings into concrete [lo, hi] bounds
    for each of X1/X2/X3. X1 and X3 are multiples of their own observed
    training range; X2 is an explicit literal range (see config.py for why
    each was chosen that way). Exported to model_weights.json so Sheets/
    Excel only ever need to clip against three plain [lo, hi] pairs, with no
    need to replicate the "multiple vs. explicit" distinction downstream.
    """
    x1_lo, x1_hi = x_train_bounds["X1_delta_earnings"]
    x3_lo, x3_hi = x_train_bounds["X3_interaction"]
    return {
        "X1_delta_earnings": [x1_lo * config.EXTRAPOLATION_CAP_X1_MULTIPLE, x1_hi * config.EXTRAPOLATION_CAP_X1_MULTIPLE],
        "X2_delta_yield": list(config.EXTRAPOLATION_CAP_X2_BOUNDS),
        "X3_interaction": [x3_lo * config.EXTRAPOLATION_CAP_X3_MULTIPLE, x3_hi * config.EXTRAPOLATION_CAP_X3_MULTIPLE],
    }


def cap_scenario_to_bounds(x_row: np.ndarray, x_train_bounds: dict):
    """
    Clamps X1 (delta earnings) and X2 (delta yield) to their resolved cap
    bounds (see compute_extrapolation_cap_bounds), then recomputes
    X3 = X1*X2 from the ALREADY-capped X1/X2 and additionally clamps X3
    itself to its own resolved bounds. Capping the interaction term
    independently matters: capping X1 and X2 alone doesn't sufficiently
    bound their product, which can still reach far outside X3's own cap
    range even when X1 and X2 are each individually within theirs (verified
    numerically -- see README).

    Returns (x_row_capped, capped_dims: list[str]) so the caller can report
    exactly which inputs were adjusted, rather than silently substituting a
    different scenario than the one requested.
    """
    cap_bounds = compute_extrapolation_cap_bounds(x_train_bounds)
    x1_cap_lo, x1_cap_hi = cap_bounds["X1_delta_earnings"]
    x2_cap_lo, x2_cap_hi = cap_bounds["X2_delta_yield"]
    x3_cap_lo, x3_cap_hi = cap_bounds["X3_interaction"]

    x1_raw, x2_raw = float(x_row[1]), float(x_row[2])
    x1_capped = float(np.clip(x1_raw, x1_cap_lo, x1_cap_hi))
    x2_capped = float(np.clip(x2_raw, x2_cap_lo, x2_cap_hi))
    x3_raw = x1_capped * x2_capped
    x3_capped = float(np.clip(x3_raw, x3_cap_lo, x3_cap_hi))

    capped_dims = []
    if x1_capped != x1_raw:
        capped_dims.append(f"X1_delta_earnings: {x1_raw:.3f} -> {x1_capped:.3f} (cap [{x1_cap_lo:.3f}, {x1_cap_hi:.3f}])")
    if x2_capped != x2_raw:
        capped_dims.append(f"X2_delta_yield: {x2_raw:.3f} -> {x2_capped:.3f} (cap [{x2_cap_lo:.3f}, {x2_cap_hi:.3f}])")
    if x3_capped != x3_raw:
        capped_dims.append(f"X3_interaction: {x3_raw:.3f} -> {x3_capped:.3f} (cap [{x3_cap_lo:.3f}, {x3_cap_hi:.3f}])")

    return np.array([1.0, x1_capped, x2_capped, x3_capped]), capped_dims


def _assert_finite_scenario_input(value, label: str) -> None:
    """
    Scenario inputs must be real, finite numbers. Without this, NaN flows
    through np.clip and the sigmoid into an all-NaN matrix while the range
    check (NaN comparisons are always False) reports no warning, and +/-inf
    is silently clamped into a plausible-looking answer for an input that
    was never a real scenario.
    """
    is_number = isinstance(value, (int, float, np.integer, np.floating)) and not isinstance(value, bool)
    if not is_number or not np.isfinite(value):
        raise ValueError(f"{label} must be a finite number in percentage points (got {value!r})")


def predict_scenario(delta_earnings: float, delta_yield: float, beta_hat: dict, x_train_bounds: dict) -> dict:
    """
    The exploratory "what if" entry point (what Sheets/Excel's scenario
    calculators call) -- distinct from forecast_transition_tensor, which is
    used for batch evaluation on real (always in-range) X_test data and
    stays uncapped/unmodified. Caps the input (see cap_scenario_to_bounds),
    builds the full 4x4 transition matrix from the capped values, and
    reports both the raw-vs-in-range check and what, if anything, was capped.
    Raises ValueError for NaN/inf/non-numeric inputs instead of returning garbage.
    """
    _assert_finite_scenario_input(delta_earnings, "delta_earnings")
    _assert_finite_scenario_input(delta_yield, "delta_yield")
    x_row_raw = np.array([1.0, delta_earnings, delta_yield, delta_earnings * delta_yield])
    range_warnings = assess_scenario_range(x_row_raw, x_train_bounds)
    x_row_capped, capped_dims = cap_scenario_to_bounds(x_row_raw, x_train_bounds)

    matrix = np.zeros((config.N_TRANCHES, config.N_TRANCHES))
    for tranche in ["IG", "NonIG", "Junk"]:
        idx = config.TRANCHES.index(tranche)
        matrix[idx, :] = cascade_row(tranche, x_row_capped, beta_hat)
    matrix[config.DEFAULT_IDX, config.DEFAULT_IDX] = 1.0

    return {
        "matrix": matrix,
        "range_warnings": range_warnings,
        "capped_dims": capped_dims,
        "was_capped": len(capped_dims) > 0,
    }


def export_model_weights(beta_hat: dict, path: str, x_train_bounds: dict) -> None:
    payload = {
        "tranches": config.TRANCHES,
        "default_idx": config.DEFAULT_IDX,
        "max_fall": config.MAX_FALL,
        "cascade_stages": config.CASCADE_STAGES,
        "upgrade_prob": config.UPGRADE_PROB,
        "feature_names": config.FEATURE_NAMES,
        "feature_units_note": "X1/X2 are delta_earnings/delta_yield in PERCENTAGE POINTS (1.0 = 100bp), X3 = X1*X2",
        "x_train_bounds": x_train_bounds,
        "x_train_bounds_note": "Observed min/max of X1/X2/X3 in the training data. Scenarios outside this range are extrapolations the model was never validated on.",
        "extrapolation_cap_bounds": compute_extrapolation_cap_bounds(x_train_bounds),
        "extrapolation_cap_note": "X1/X2/X3 are clamped to these [lo, hi] bounds before scenario prediction, to prevent sigmoid saturation (literal 0%/100%) on extreme inputs. Per-feature, not one shared multiplier -- see config.py for why each was chosen independently.",
        "beta_hat": beta_dict_to_json(beta_hat),
    }
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)


if __name__ == "__main__":
    from simulate_macro import simulate_macro_paths
    from matrix_builder import build_feature_matrix
    from markov_dgp import build_transition_tensor, simulate_observed_counts
    from feature_engineering import build_all_features
    from ols_model import fit_all_stages

    print("=== Softmax / chain validation ===")
    print(validate_softmax_and_chain())

    macro_df = simulate_macro_paths()
    X = build_feature_matrix(macro_df)
    p_true = build_transition_tensor(X, config.BETA_TRUE)
    counts = simulate_observed_counts(p_true)
    feats = build_all_features(macro_df, counts)

    fits = fit_all_stages(feats["X_train"], feats["Y_train"], feats["W_train"])
    beta_hat = {tr: {st: fits[tr][st]["beta_hat"] for st in stages} for tr, stages in feats["Y_train"].items()}

    X_test = feats["X_test"]
    test_idx = feats["test_idx"]

    p_forecast_test = forecast_transition_tensor(X_test, beta_hat)
    p_true_test = p_true[test_idx]

    forecast_cum = cumulative_sequence(p_forecast_test)
    true_cum = cumulative_sequence(p_true_test)

    evaluation = evaluate_cumulative(true_cum, forecast_cum)
    print("\n=== Cumulative forecast evaluation (test quarters) ===")
    print("Overall MSE:", evaluation["overall_mse"])
    print("Overall Frobenius norm:", evaluation["overall_frobenius_norm"])
    for p in evaluation["per_period"][:3] + evaluation["per_period"][-3:]:
        print(p)

    x_train_bounds = compute_x_train_bounds(feats["X_train"])
    print("\n=== Extrapolation range check on test quarters ===")
    any_out_of_range = False
    for t, x_row in enumerate(X_test):
        warnings = assess_scenario_range(x_row, x_train_bounds)
        if warnings:
            any_out_of_range = True
            print(f"test quarter {t+1}:", warnings)
    if not any_out_of_range:
        print("all test quarters within the observed training range (expected, same generative process)")

    export_model_weights(beta_hat, "../model_weights.json", x_train_bounds)
    print("\nExported model_weights.json")
