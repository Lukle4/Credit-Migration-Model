"""
Step 3: OLS / WLS engine.

Implements beta_hat = (X'WX)^-1 X'WY (W = identity for plain OLS), with two
numerical-stability refinements:

1. Column-norm preconditioning: the raw design matrix has some scale
   mismatch (intercept ~O(1) vs. macro-delta columns), so the linear solve
   is done on a rescaled X, then divided back -- exact, not approximate,
   and beta_hat ends up in the same raw units as beta_true either way.

2. Weighted least squares (weights optional, default None = plain OLS):
   the log-odds Y = ln(n_fall/n_stay) computed from finite company counts
   has variance ~= 1/n_fall + 1/n_stay (delta method), which is different
   every quarter -- a quarter with 40 observed events is far more
   informative than one with 2. Plain OLS treats every quarter as equally
   reliable, which is wrong and especially costly for the rare Stage B/C
   events. WLS is implemented via the standard sqrt(w) transform (Xw =
   sqrt(w)*X, Yw = sqrt(w)*Y, then ordinary least squares on Xw/Yw), so it's
   still exactly the closed-form solve above, just on reweighted data.

   WLS with uniform weights must reproduce plain OLS exactly -- see
   validate_wls_reduces_to_ols() below, which checks this explicitly rather
   than assuming the algebra is bug-free.

Also reports R^2, residual standard error, per-coefficient standard errors
and t-stats using degrees of freedom n - k (k=4, the design matrix is full
rank now that X3 is an interaction term rather than an exact linear
combination of X1, X2). All sums of squares (SSE, SST, sigma^2) are computed
on the WEIGHTED scale for consistency when weights are supplied.
"""

import numpy as np

import config


def fit_ols(X: np.ndarray, Y: np.ndarray, weights: np.ndarray = None) -> dict:
    n, k = X.shape
    w = np.ones(n) if weights is None else np.asarray(weights, dtype=float)
    sqrt_w = np.sqrt(w)

    # Standard WLS-via-OLS trick: reweight both sides, then it's an ordinary
    # least-squares problem again -- still the same closed-form solve.
    Xw = X * sqrt_w[:, None]
    Yw = Y * sqrt_w

    s = np.linalg.norm(Xw, axis=0)
    s[s == 0] = 1.0  # guard, not expected to trigger
    Xw_scaled = Xw / s

    XtX_scaled = Xw_scaled.T @ Xw_scaled
    XtY_scaled = Xw_scaled.T @ Yw
    beta_scaled = np.linalg.solve(XtX_scaled, XtY_scaled)
    beta_hat = beta_scaled / s

    # Cross-check against SVD-based lstsq, on the SAME weighted/reweighted
    # data -- comparing a weighted solve against an unweighted lstsq would
    # always show a spurious "disagreement" that isn't actually a bug.
    beta_lstsq, *_ = np.linalg.lstsq(Xw, Yw, rcond=None)
    max_disagreement = float(np.max(np.abs(beta_hat - beta_lstsq)))

    Y_hat = X @ beta_hat  # fitted values in the original (unweighted) space
    residuals = Y - Y_hat
    weighted_residuals = sqrt_w * residuals
    sse = float(weighted_residuals @ weighted_residuals)

    # R^2 here is on the WEIGHTED scale (weighted SSE / weighted SST), the
    # standard convention for WLS. NOT directly comparable to a plain-OLS R^2
    # computed elsewhere (or to this same pipeline's R^2 before WLS was
    # introduced) even though both are called "R^2" -- say so explicitly
    # wherever these numbers get quoted.
    y_bar_w = float(np.sum(w * Y) / np.sum(w))
    sst = float(np.sum(w * (Y - y_bar_w) ** 2))
    r_squared = 1.0 - sse / sst if sst > 0 else np.nan

    dof = n - k
    sigma2 = sse / dof if dof > 0 else np.nan
    XtX_scaled_inv = np.linalg.inv(XtX_scaled)
    var_beta = sigma2 * (XtX_scaled_inv / np.outer(s, s))
    se_beta = np.sqrt(np.diag(var_beta))
    t_stats = beta_hat / se_beta

    return {
        "beta_hat": beta_hat,
        "Y_hat": Y_hat,
        "residuals": residuals,
        "r_squared": r_squared,
        "se_beta": se_beta,
        "t_stats": t_stats,
        "dof": dof,
        "weighted": weights is not None,
        "condition_number_raw": float(np.linalg.cond(X.T @ X)),
        "condition_number_preconditioned": float(np.linalg.cond(XtX_scaled)),
        "lstsq_max_disagreement": max_disagreement,
    }


def validate_wls_reduces_to_ols(X: np.ndarray, Y: np.ndarray) -> dict:
    """
    WLS with uniform weights (all 1s) must exactly reproduce plain OLS.
    Checked explicitly -- like validate_softmax_and_chain in the Markov
    engine -- rather than trusted by algebra alone.
    """
    plain = fit_ols(X, Y, weights=None)
    weighted_uniform = fit_ols(X, Y, weights=np.ones(len(Y)))
    max_diff = float(np.max(np.abs(plain["beta_hat"] - weighted_uniform["beta_hat"])))
    return {"max_beta_diff": max_diff, "passed": bool(max_diff < 1e-8)}


def monte_carlo_wls_vs_ols(p_true_tensor: np.ndarray, X_train: np.ndarray, train_idx: np.ndarray, n_seeds: int = 40) -> dict:
    """
    Is WLS actually a better estimator than plain OLS, or did a single
    sampling draw just happen to favor one of them for a given stage? Since
    this is a synthetic study, that question is directly answerable: hold
    the macro path (and therefore Ptrue) fixed, redraw the finite-sample
    company-pool noise n_seeds times, and compare each estimator's average
    cosine similarity to the known beta_true. A single-seed comparison (as
    used everywhere else in this pipeline, seed=config.RANDOM_SEED) can be
    misleading for the sparsest stages (few observed events -> high
    seed-to-seed variance in which estimator "wins") -- this check is what
    actually settles it.
    """
    import markov_dgp as mdgp
    import feature_engineering as fe

    scores = {tr: {st: {"ols": [], "wls": []} for st in stages} for tr, stages in config.CASCADE_STAGES.items()}
    for i in range(n_seeds):
        counts = mdgp.simulate_observed_counts(p_true_tensor, seed=i * 1000 + 7)
        Y_train, W_train = fe.build_Y_and_W_by_stage(counts, train_idx)
        for tranche, stages in config.CASCADE_STAGES.items():
            for stage in stages:
                y = Y_train[tranche][stage]
                w = W_train[tranche][stage]
                bt = config.BETA_TRUE[tranche][stage]
                for key, weights in (("ols", None), ("wls", w)):
                    bh = fit_ols(X_train, y, weights=weights)["beta_hat"]
                    cos_sim = float(bh @ bt / (np.linalg.norm(bh) * np.linalg.norm(bt)))
                    scores[tranche][stage][key].append(cos_sim)

    summary = {}
    for tranche, stages in scores.items():
        summary[tranche] = {}
        for stage, vals in stages.items():
            ols_arr, wls_arr = np.array(vals["ols"]), np.array(vals["wls"])
            summary[tranche][stage] = {
                "ols_mean_cos_sim": float(ols_arr.mean()), "ols_std": float(ols_arr.std()),
                "wls_mean_cos_sim": float(wls_arr.mean()), "wls_std": float(wls_arr.std()),
                "wls_wins": bool(wls_arr.mean() > ols_arr.mean()),
                "difference_within_one_std": bool(abs(wls_arr.mean() - ols_arr.mean()) < max(ols_arr.std(), wls_arr.std())),
            }
    return summary


def fit_all_stages(X_train: np.ndarray, Y_train: dict, W_train: dict = None) -> dict:
    """Runs fit_ols once per (tranche, stage) -- 6 regressions total. WLS if W_train given."""
    results = {}
    for tranche, stages in Y_train.items():
        results[tranche] = {}
        for stage, y in stages.items():
            w = W_train[tranche][stage] if W_train is not None else None
            results[tranche][stage] = fit_ols(X_train, y, weights=w)
    return results


def beta_hat_vs_true(fit_results: dict, beta_true: dict = config.BETA_TRUE) -> dict:
    comparison = {}
    for tranche, stages in fit_results.items():
        comparison[tranche] = {}
        for stage, res in stages.items():
            bh = res["beta_hat"]
            bt = beta_true[tranche][stage]
            abs_err = np.abs(bh - bt)
            # bt components are never 0 in the current calibration, but guard
            # anyway rather than silently emitting inf if that ever changes.
            rel_err = np.where(bt != 0, abs_err / np.where(bt != 0, np.abs(bt), 1.0), np.nan)
            cos_sim = float(bh @ bt / (np.linalg.norm(bh) * np.linalg.norm(bt)))
            comparison[tranche][stage] = {
                "beta_true": bt, "beta_hat": bh,
                "abs_error": abs_err, "rel_error": rel_err,
                "cosine_similarity": cos_sim,
            }
    return comparison


def engine_sanity_check(feature_matrix: np.ndarray, p_true_tensor: np.ndarray) -> dict:
    """
    Verifies the OLS engine + cascade math are correct in isolation from
    sampling noise: builds Y directly from the exact (noiseless) Ptrue
    probabilities instead of noisy observed counts, fits plain OLS (no
    weights -- there's no sampling noise to weight against here), and
    confirms beta_hat recovers beta_true almost exactly (up to float
    precision). This isolates "does the engine work" from "how much does
    finite-sample noise degrade the estimate", which real Phat-based fits
    will show.
    """
    results = {}
    for tranche in ["IG", "NonIG", "Junk"]:
        idx = config.TRANCHES.index(tranche)
        max_fall = config.MAX_FALL[tranche]
        results[tranche] = {}
        for stage in config.CASCADE_STAGES[tranche]:
            y_vals = []
            for t in range(feature_matrix.shape[0]):
                row = p_true_tensor[t, idx, :]
                if stage == "A":
                    p_stay = row[idx]
                    p_fall = row[idx + 1: idx + 1 + max_fall].sum()
                    y_vals.append(np.log(p_fall / p_stay))
                elif stage == "B":
                    p_eq1 = row[idx + 1]
                    p_ge2 = row[idx + 2: idx + 1 + max_fall].sum()
                    y_vals.append(np.log(p_ge2 / p_eq1))
                elif stage == "C":
                    y_vals.append(np.log(row[idx + 3] / row[idx + 2]))
            fit = fit_ols(feature_matrix, np.array(y_vals))
            results[tranche][stage] = {
                "beta_true": config.BETA_TRUE[tranche][stage],
                "beta_hat_noiseless": fit["beta_hat"],
                "max_abs_diff": float(np.max(np.abs(fit["beta_hat"] - config.BETA_TRUE[tranche][stage]))),
            }
    return results


if __name__ == "__main__":
    from simulate_macro import simulate_macro_paths
    from matrix_builder import build_feature_matrix
    from markov_dgp import build_transition_tensor, simulate_observed_counts
    from feature_engineering import build_all_features

    macro_df = simulate_macro_paths()
    X = build_feature_matrix(macro_df)
    p_true = build_transition_tensor(X, config.BETA_TRUE)
    counts = simulate_observed_counts(p_true)
    feats = build_all_features(macro_df, counts)

    print("=== WLS-reduces-to-OLS check ===")
    print(validate_wls_reduces_to_ols(feats["X_train"], feats["Y_train"]["IG"]["A"]))

    print("\n=== Engine sanity check (noiseless Ptrue) ===")
    sanity = engine_sanity_check(X, p_true)
    for tranche, stages in sanity.items():
        for stage, r in stages.items():
            print(f"{tranche}-{stage}: max|beta_hat-beta_true| = {r['max_abs_diff']:.6f}")

    print("\n=== Real WLS fit on noisy Phat (train only) ===")
    fits = fit_all_stages(feats["X_train"], feats["Y_train"], feats["W_train"])
    comparison = beta_hat_vs_true(fits)
    for tranche, stages in comparison.items():
        for stage, c in stages.items():
            print(f"{tranche}-{stage}: R2={fits[tranche][stage]['r_squared']:.3f} "
                  f"cos_sim={c['cosine_similarity']:.3f} "
                  f"cond(precond)={fits[tranche][stage]['condition_number_preconditioned']:.1f}")
