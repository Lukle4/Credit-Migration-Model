"""
Step 2: feature engineering.

2a. Chronological 80/20 train/test split on Phat_t ONLY. This must be a
    time-ordered split (first 48 quarters train, last 12 test), never a
    random shuffle: shuffling a time series would let the AR(1)-autocorrelated
    macro process leak future information into training. Ptrue is never
    touched here -- it is reserved for step 4's final out-of-sample
    evaluation only, never used to fit anything.

2b. Log-odds transform: Y = ln(Pij / Pii) per cascade stage (see
    markov_dgp.cascade_row for the stage definitions), computed from raw
    observed counts with additive smoothing so a zero count never produces
    ln(0). Smoothing is applied here, at the point of use, not baked into
    the stored raw Phat_t counts.

    Alongside Y, each quarter also gets a WEIGHT for WLS: the log-odds of a
    ratio of counts has variance ~= 1/n_fall + 1/n_stay (delta method), which
    varies quarter to quarter -- a quarter with 2 observed events is far
    noisier than one with 40. Plain OLS treats every quarter as equally
    informative, which understates uncertainty and wastes signal, especially
    for the rare Stage B/C events. The weight is the inverse of that
    variance, computed from the SAME smoothed counts used for Y (so weight
    and Y are never inconsistent with each other).

2c. Matrix builder: see matrix_builder.py (shared with the DGP).
"""

import numpy as np

import config
from matrix_builder import build_feature_matrix


def train_test_indices():
    train_idx = np.arange(0, config.N_TRAIN)
    test_idx = np.arange(config.N_TRAIN, config.N_QUARTERS)
    return train_idx, test_idx


def _stage_log_odds_and_weight(counts_row: np.ndarray, tranche_idx: int, stage: str, max_fall: int):
    eps = config.SMOOTHING_EPSILON
    if stage == "A":
        n_stay = counts_row[tranche_idx] + eps
        n_fall = counts_row[tranche_idx + 1: tranche_idx + 1 + max_fall].sum() + eps
    elif stage == "B":
        n_stay = counts_row[tranche_idx + 1] + eps
        n_fall = counts_row[tranche_idx + 2: tranche_idx + 1 + max_fall].sum() + eps
    elif stage == "C":
        n_stay = counts_row[tranche_idx + 2] + eps
        n_fall = counts_row[tranche_idx + 3] + eps
    else:
        raise ValueError(f"unknown stage {stage}")

    y = float(np.log(n_fall / n_stay))
    variance = 1.0 / n_fall + 1.0 / n_stay
    weight = float(1.0 / variance)
    return y, weight


def build_Y_and_W_by_stage(counts: np.ndarray, indices: np.ndarray):
    """
    Returns (Y, W), each {tranche: {stage: np.ndarray of length len(indices)}},
    computed strictly from the given quarter indices (train-only or test-only
    calls never see counts outside their own slice's quarters, since each
    quarter's counts are already independent draws -- no cross-quarter
    leakage is possible here, but the CALLER must still never pass Ptrue in
    and never mix train/test indices together).
    """
    Y, W = {}, {}
    for tranche in ["IG", "NonIG", "Junk"]:
        idx = config.TRANCHES.index(tranche)
        max_fall = config.MAX_FALL[tranche]
        Y[tranche], W[tranche] = {}, {}
        for stage in config.CASCADE_STAGES[tranche]:
            pairs = [
                _stage_log_odds_and_weight(counts[t, idx, :], idx, stage, max_fall)
                for t in indices
            ]
            Y[tranche][stage] = np.array([p[0] for p in pairs])
            w = np.array([p[1] for p in pairs])
            # Winsorize: a single near-zero-count quarter can otherwise get
            # a wildly unstable weight and dominate the fit -- see config.py.
            cap = config.WLS_WEIGHT_CAP_MULTIPLE * np.median(w)
            W[tranche][stage] = np.minimum(w, cap)
    return Y, W


def build_all_features(macro_df, counts: np.ndarray):
    """One-stop entry point: returns X_train, X_test, Y_train, Y_test, W_train, W_test."""
    X = build_feature_matrix(macro_df)
    train_idx, test_idx = train_test_indices()

    X_train, X_test = X[train_idx], X[test_idx]
    Y_train, W_train = build_Y_and_W_by_stage(counts, train_idx)
    Y_test, W_test = build_Y_and_W_by_stage(counts, test_idx)

    return {
        "X_train": X_train, "X_test": X_test,
        "Y_train": Y_train, "Y_test": Y_test,
        "W_train": W_train, "W_test": W_test,
        "train_idx": train_idx, "test_idx": test_idx,
    }


if __name__ == "__main__":
    from simulate_macro import simulate_macro_paths
    from markov_dgp import build_transition_tensor, simulate_observed_counts

    macro_df = simulate_macro_paths()
    X = build_feature_matrix(macro_df)
    p_true = build_transition_tensor(X, config.BETA_TRUE)
    counts = simulate_observed_counts(p_true)

    feats = build_all_features(macro_df, counts)
    print("X_train shape:", feats["X_train"].shape)
    print("X_test shape:", feats["X_test"].shape)
    print("Y_train['IG']['A'][:5]:", feats["Y_train"]["IG"]["A"][:5])
    print("W_train['IG']['A'][:5]:", feats["W_train"]["IG"]["A"][:5])
    print("condition number of X_train'X_train:", np.linalg.cond(feats["X_train"].T @ feats["X_train"]))
