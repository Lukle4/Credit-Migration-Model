"""
Main entry point: runs steps 1-4 end to end and prints/exports a full
diagnostic report. Run with:  python run_pipeline.py
"""

import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

import numpy as np

import config
from simulate_macro import simulate_macro_paths, noise_budget_report
from matrix_builder import build_feature_matrix
from markov_dgp import build_transition_tensor, simulate_observed_counts
from feature_engineering import build_all_features
from ols_model import fit_all_stages, beta_hat_vs_true, engine_sanity_check, validate_wls_reduces_to_ols, monte_carlo_wls_vs_ols
from markov_engine import (
    validate_softmax_and_chain, forecast_transition_tensor, cumulative_sequence,
    evaluate_cumulative, export_model_weights, beta_dict_to_json,
    compute_x_train_bounds, assess_scenario_range,
)
from sheets_export import export_apps_script
from excel_export import export_workbook


def main():
    print("=" * 70)
    print("STEP 1a: Macro simulation")
    print("=" * 70)
    macro_df = simulate_macro_paths()
    noise_report = noise_budget_report(macro_df)
    for series_name, r in noise_report.items():
        print(f"  {series_name}: idiosyncratic_std={r['idiosyncratic_shock_std']:.4f} "
              f"realized_std={r['realized_total_std']:.4f} "
              f"noise/total_ratio={r['noise_to_total_std_ratio']:.3f} "
              f"outliers(|z|>{config.OUTLIER_Z_THRESHOLD})={r['n_outlier_quarters']}")

    print("\n" + "=" * 70)
    print("STEP 1b: Markov DGP (Ptrue ground truth + Phat_t observed)")
    print("=" * 70)
    X = build_feature_matrix(macro_df)
    p_true = build_transition_tensor(X, config.BETA_TRUE)
    counts = simulate_observed_counts(p_true)
    for tranche in ["IG", "NonIG", "Junk"]:
        idx = config.TRANCHES.index(tranche)
        p_fall_any = 1 - p_true[:, idx, idx] - (p_true[:, idx, idx - 1] if idx > 0 else 0)
        print(f"  {tranche}: P(any fall) mean={p_fall_any.mean():.4f} "
              f"min={p_fall_any.min():.4f} max={p_fall_any.max():.4f} "
              f"| P(straight to default) max={p_true[:, idx, config.DEFAULT_IDX].max():.4f}")

    print("\n" + "=" * 70)
    print("STEP 2: Feature engineering (chronological 80/20 split, log-odds, X matrix)")
    print("=" * 70)
    feats = build_all_features(macro_df, counts)
    print(f"  X_train: {feats['X_train'].shape}  X_test: {feats['X_test'].shape}")
    print(f"  train quarters: 1-{config.N_TRAIN}  test quarters: {config.N_TRAIN + 1}-{config.N_QUARTERS}")
    print(f"  condition number of raw X_train'X_train: {np.linalg.cond(feats['X_train'].T @ feats['X_train']):.3e}")

    print("\n" + "=" * 70)
    print("STEP 3: OLS engine")
    print("=" * 70)
    print("  -- Engine sanity check (noiseless Ptrue, should match beta_true exactly) --")
    sanity = engine_sanity_check(X, p_true)
    for tranche, stages in sanity.items():
        for stage, r in stages.items():
            print(f"    {tranche}-{stage}: max|beta_hat-beta_true| = {r['max_abs_diff']:.9f}")

    print("  -- WLS-reduces-to-OLS check (uniform weights must exactly match plain OLS) --")
    wls_check = validate_wls_reduces_to_ols(feats["X_train"], feats["Y_train"]["IG"]["A"])
    print(f"    {wls_check}")
    assert wls_check["passed"], "WLS with uniform weights diverged from plain OLS"

    print("  -- Monte Carlo: is WLS actually better than OLS on average, or just this one seed? --")
    print("     (40 independent redraws of the company-pool sampling noise, same macro path)")
    mc = monte_carlo_wls_vs_ols(p_true, feats["X_train"], feats["train_idx"], n_seeds=40)
    for tranche, stages in mc.items():
        for stage, r in stages.items():
            verdict = "WLS wins" if r["wls_wins"] else "OLS wins"
            if r["difference_within_one_std"]:
                verdict += " (within 1 std -- treat as a toss-up, not a real difference)"
            print(f"    {tranche}-{stage}: OLS={r['ols_mean_cos_sim']:.3f}+/-{r['ols_std']:.3f}  "
                  f"WLS={r['wls_mean_cos_sim']:.3f}+/-{r['wls_std']:.3f}  {verdict}")

    print("  -- Real WLS fit on noisy Phat_t (train only), weighted by count-implied reliability --")
    fits = fit_all_stages(feats["X_train"], feats["Y_train"], feats["W_train"])
    comparison = beta_hat_vs_true(fits)
    for tranche, stages in fits.items():
        for stage, res in stages.items():
            c = comparison[tranche][stage]
            print(f"    {tranche}-{stage}: R2={res['r_squared']:.3f} "
                  f"cos_sim(beta_hat,beta_true)={c['cosine_similarity']:.3f} "
                  f"dof={res['dof']} cond(preconditioned)={res['condition_number_preconditioned']:.2f}")
            print(f"        beta_true={np.round(c['beta_true'],3)}  beta_hat={np.round(c['beta_hat'],3)}")
            print(f"        SE(beta_hat)={np.round(res['se_beta'],3)}  t_stats={np.round(res['t_stats'],2)}")

    beta_hat = {tr: {st: fits[tr][st]["beta_hat"] for st in stages} for tr, stages in feats["Y_train"].items()}

    print("\n" + "=" * 70)
    print("STEP 4: Markov engine, cumulative forecast, evaluation, export")
    print("=" * 70)
    checks = validate_softmax_and_chain()
    print("  Softmax/chain validation:", checks)
    assert all(checks.values()), "softmax/chain validation failed"

    test_idx = feats["test_idx"]
    p_forecast_test = forecast_transition_tensor(feats["X_test"], beta_hat)
    p_true_test = p_true[test_idx]

    forecast_cum = cumulative_sequence(p_forecast_test)
    true_cum = cumulative_sequence(p_true_test)
    evaluation = evaluate_cumulative(true_cum, forecast_cum)
    print(f"  Overall cumulative MSE: {evaluation['overall_mse']:.6e}")
    print(f"  Overall cumulative Frobenius norm: {evaluation['overall_frobenius_norm']:.5f}")
    print(f"  1-quarter-ahead Frobenius norm: {evaluation['per_period'][0]['frobenius_norm']:.5f}")
    print(f"  {config.N_TEST}-quarter-ahead (full test horizon) Frobenius norm: {evaluation['per_period'][-1]['frobenius_norm']:.5f}")

    x_train_bounds = compute_x_train_bounds(feats["X_train"])
    print("\n  -- Extrapolation range check on test quarters --")
    any_out_of_range = False
    for t, x_row in enumerate(feats["X_test"]):
        range_warnings = assess_scenario_range(x_row, x_train_bounds)
        if range_warnings:
            any_out_of_range = True
            print(f"    test quarter {t + 1}:", range_warnings)
    if not any_out_of_range:
        print("    all test quarters within the observed training range (expected, same generative process)")

    out_dir = os.path.dirname(__file__)
    export_model_weights(beta_hat, os.path.join(out_dir, "model_weights.json"), x_train_bounds)
    print(f"\n  Exported model_weights.json")

    print("\n" + "=" * 70)
    print("STEP 5: Generate the Google Sheets engine (sheets/Code.gs)")
    print("=" * 70)
    apps_script = export_apps_script(
        os.path.join(out_dir, "model_weights.json"), os.path.join(out_dir, "sheets", "Code.gs"))
    print(f"  Wrote sheets/Code.gs ({len(apps_script.splitlines())} lines); embedded weights verified identical to model_weights.json")

    print("\n" + "=" * 70)
    print("STEP 6: Generate the Excel scenario calculator")
    print("=" * 70)
    excel_out = os.path.join(out_dir, "credit_migration_scenario_calculator.xlsx")
    try:
        export_workbook(os.path.join(out_dir, "model_weights.json"), excel_out, n_quarters=config.N_QUARTERS)
        print(f"  Wrote credit_migration_scenario_calculator.xlsx (self-verified against model_weights.json)")
    except PermissionError:
        print(f"  SKIPPED: {excel_out} is open in Excel (close it and re-run to refresh the workbook)")

    eval_report = {
        "noise_budget": {k: {kk: (vv if not isinstance(vv, list) else vv) for kk, vv in v.items()} for k, v in noise_report.items()},
        "engine_sanity_check_max_abs_diff": {
            tr: {st: r["max_abs_diff"] for st, r in stages.items()} for tr, stages in sanity.items()
        },
        "ols_diagnostics": {
            tr: {
                st: {
                    "r_squared": res["r_squared"],
                    "se_beta": res["se_beta"].tolist(),
                    "t_stats": res["t_stats"].tolist(),
                    "dof": res["dof"],
                    "cosine_similarity_beta_hat_vs_true": comparison[tr][st]["cosine_similarity"],
                } for st, res in stages.items()
            } for tr, stages in fits.items()
        },
        "wls_reduces_to_ols_check": wls_check,
        "wls_vs_ols_monte_carlo": mc,
        "softmax_chain_validation": checks,
        "cumulative_forecast_evaluation": evaluation,
        "x_train_bounds": x_train_bounds,
    }
    with open(os.path.join(out_dir, "evaluation_report.json"), "w") as f:
        json.dump(eval_report, f, indent=2)
    print("  Exported evaluation_report.json")


if __name__ == "__main__":
    main()
