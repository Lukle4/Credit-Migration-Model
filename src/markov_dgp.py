"""
Step 1b: Markov data-generating process.

Ground truth (Ptrue): for each of the 3 non-default tranches, a cascading
continuation-ratio logit ("fall>=1 vs stay", then "fall>=2 vs fall==1", then
"fall==3 vs fall==2") converts the macro feature row into a full probability
distribution over destinations. This is the only severity-model design that
lets every stage be fit later with the plain closed-form OLS formula
beta_hat = (X'X)^-1 X'Y, because each stage is exactly Y = ln(Pij/Pii) with
Pij/Pii redefined on a conditional subset -- see README for the full
rationale. Upgrades (one tranche up only) are fixed, non-macro-sensitive
constants (config.UPGRADE_PROB), a flagged simplifying assumption. Default
(tranche index 3) is an absorbing state: row is always [0, 0, 0, 1].

Observed data (Phat_t): Ptrue perturbed by genuine finite-sample noise. Each
quarter, config.COMPANY_POOL_SIZE companies "at risk" in each source tranche
are drawn via a multinomial with Ptrue's row as the true probability vector;
Phat_t is the resulting empirical frequency table. This is the "full dataset"
referred to in the project plan.
"""

import numpy as np
import pandas as pd

import config
from matrix_builder import build_feature_matrix


def _sigmoid(z: float) -> float:
    """
    Numerically stable logistic sigmoid. The naive 1/(1+exp(-z)) form can
    overflow exp() for very negative z -- a real risk once out-of-range
    macro scenarios are allowed through with a warning rather than blocked
    (see markov_engine.assess_scenario_range). This formulation keeps the
    argument passed to exp() always <= 0, so it can only underflow toward 0
    (safe) and never overflow.
    """
    if z >= 0:
        return 1.0 / (1.0 + np.exp(-z))
    exp_z = np.exp(z)
    return exp_z / (1.0 + exp_z)


def cascade_row(tranche: str, x_row: np.ndarray, beta_stage_dict: dict) -> np.ndarray:
    """
    Decode one tranche's cascading logit stages into a full 1x4 probability
    row over [IG, NonIG, Junk, Default]. beta_stage_dict is keyed
    tranche -> stage -> 4-vector (same shape as config.BETA_TRUE, so this
    function works identically for beta_true and beta_hat).
    """
    idx = config.TRANCHES.index(tranche)
    u = config.UPGRADE_PROB[tranche]
    max_fall = config.MAX_FALL[tranche]
    betas = beta_stage_dict[tranche]

    logit_a = float(x_row @ betas["A"])
    p_fall_ge1 = _sigmoid(logit_a)
    p_stay = 1.0 - p_fall_ge1

    p_fall_eq = {}
    if max_fall == 1:
        p_fall_eq[1] = p_fall_ge1
    else:
        logit_b = float(x_row @ betas["B"])
        p_fall_ge2_given = _sigmoid(logit_b)
        p_fall_eq[1] = p_fall_ge1 * (1.0 - p_fall_ge2_given)
        p_fall_ge2 = p_fall_ge1 * p_fall_ge2_given
        if max_fall == 2:
            p_fall_eq[2] = p_fall_ge2
        else:  # max_fall == 3
            logit_c = float(x_row @ betas["C"])
            p_eq3_given = _sigmoid(logit_c)
            p_fall_eq[2] = p_fall_ge2 * (1.0 - p_eq3_given)
            p_fall_eq[3] = p_fall_ge2 * p_eq3_given

    row = np.zeros(config.N_TRANCHES)
    if u > 0:
        row[idx - 1] = u
    row[idx] += (1.0 - u) * p_stay
    for k, p in p_fall_eq.items():
        row[idx + k] += (1.0 - u) * p
    return row


def build_transition_tensor(feature_matrix: np.ndarray, beta_stage_dict: dict) -> np.ndarray:
    """Returns shape (n_quarters, 4, 4): one full transition matrix per quarter."""
    n = feature_matrix.shape[0]
    tensor = np.zeros((n, config.N_TRANCHES, config.N_TRANCHES))
    for t in range(n):
        x_row = feature_matrix[t]
        for tranche in ["IG", "NonIG", "Junk"]:
            idx = config.TRANCHES.index(tranche)
            tensor[t, idx, :] = cascade_row(tranche, x_row, beta_stage_dict)
        tensor[t, config.DEFAULT_IDX, config.DEFAULT_IDX] = 1.0  # absorbing
    return tensor


def simulate_observed_counts(p_true_tensor: np.ndarray, seed: int = config.RANDOM_SEED) -> np.ndarray:
    """
    Draws a finite company pool per source tranche per quarter from the true
    row probabilities. Returns raw integer counts, shape (n_quarters, 4, 4),
    NOT yet converted to frequencies or smoothed -- that happens in feature
    engineering so the stored "observed data" stays genuinely raw.
    """
    rng = np.random.default_rng(seed + 1)  # distinct stream from macro shocks
    n = p_true_tensor.shape[0]
    counts = np.zeros((n, config.N_TRANCHES, config.N_TRANCHES), dtype=int)
    for t in range(n):
        for tranche in ["IG", "NonIG", "Junk"]:
            idx = config.TRANCHES.index(tranche)
            probs = p_true_tensor[t, idx, :]
            probs = probs / probs.sum()  # guard against float drift
            counts[t, idx, :] = rng.multinomial(config.COMPANY_POOL_SIZE, probs)
        counts[t, config.DEFAULT_IDX, config.DEFAULT_IDX] = config.COMPANY_POOL_SIZE
    return counts


def counts_to_frequencies(counts: np.ndarray) -> np.ndarray:
    pool = counts.sum(axis=2, keepdims=True)
    return counts / pool


if __name__ == "__main__":
    from simulate_macro import simulate_macro_paths

    macro_df = simulate_macro_paths()
    X = build_feature_matrix(macro_df)
    p_true = build_transition_tensor(X, config.BETA_TRUE)
    print("Ptrue[0] (quarter 1, IG row -> [IG,NonIG,Junk,Default]):")
    print(p_true[0, 0, :])
    print("Row sums (should all be 1.0):", p_true[0].sum(axis=1))

    counts = simulate_observed_counts(p_true)
    phat = counts_to_frequencies(counts)
    print("Phat[0] (quarter 1, observed, finite-sample noise):")
    print(phat[0, 0, :])
