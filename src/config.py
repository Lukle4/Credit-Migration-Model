"""
Central configuration: every hardcoded / ground-truth constant used across the
pipeline lives here so the thesis write-up can cite one source of truth and so
nothing is silently duplicated between modules.
"""

import numpy as np

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
RANDOM_SEED = 42

# ---------------------------------------------------------------------------
# Time structure
# ---------------------------------------------------------------------------
N_QUARTERS = 60          # analysis horizon (15 years)
N_BURNIN = 20             # extra AR(1) warm-up quarters, generated then discarded
TRAIN_FRACTION = 0.80     # chronological split: first 80% of the 60 quarters
N_TRAIN = int(round(N_QUARTERS * TRAIN_FRACTION))   # 48
N_TEST = N_QUARTERS - N_TRAIN                        # 12

# ---------------------------------------------------------------------------
# Macro simulation (mean-reverting AR(1) / Vasicek-style)
# Idiosyncratic shock sizes are deliberately kept small relative to the
# mean-reversion signal, per instruction to minimise noise contamination.
# simulate_macro.py reports a diagnostic "noise budget" so this is auditable
# rather than just asserted.
# ---------------------------------------------------------------------------
YIELD_START = 0.035        # 3.5% starting 10Y UST yield
# The long-run mean the yield reverts to is a deterministic CYCLE, not a flat
# constant. A constant target would make idiosyncratic shocks the only
# source of variation at all (there would be no other driver of Y), which
# is the opposite of "minimise noise" -- it makes noise 100% of the signal.
# A cyclical target (proxying real rate-hiking/cutting cycles) gives genuine
# macro-driven variation for beta to explain, while shocks stay small
# relative to that cycle -- see simulate_macro.noise_budget_report for the
# resulting signal-to-noise ratio, printed and flagged on every run.
YIELD_CYCLE_BASE = 0.035          # 3.5% cycle midpoint
YIELD_CYCLE_AMPLITUDE = 0.012     # +/-1.2%, so yield cycles roughly 2.3%-4.7%
YIELD_CYCLE_PERIOD_QUARTERS = 32  # 8-year full hike/cut cycle
YIELD_MEAN_REVERSION_SPEED = 0.35    # kappa, per quarter (tracks the cycle reasonably closely)
YIELD_SHOCK_STD = 0.0009             # 9bps idiosyncratic quarterly shock std -- small vs. the cycle

EARNINGS_START = 0.015     # 1.5% starting quarterly earnings growth
EARNINGS_LONG_RUN_MEAN = 0.015
EARNINGS_MEAN_REVERSION_SPEED = 0.30  # kappa, per quarter
EARNINGS_YIELD_SENSITIVITY = -2.5     # d(earnings growth) / d(yield change): rising rates hurt earnings
EARNINGS_SHOCK_STD = 0.0012            # 12bps idiosyncratic quarterly shock std -- small vs. the yield-driven signal

OUTLIER_Z_THRESHOLD = 3.0  # |z| beyond this on quarterly changes gets flagged in diagnostics

# ---------------------------------------------------------------------------
# Credit tranches
# Tranche 3 (Default) is an absorbing state: its row is always [0, 0, 0, 1].
# ---------------------------------------------------------------------------
TRANCHES = ["IG", "NonIG", "Junk", "Default"]
N_TRANCHES = len(TRANCHES)
DEFAULT_IDX = 3

# How many notches down each non-default tranche can fall in a single quarter
# (bounded by how many tranches exist below it).
MAX_FALL = {"IG": 3, "NonIG": 2, "Junk": 1}

# Cascading continuation-ratio logit stages needed per tranche.
# Stage A: fall>=1 vs stay | Stage B: fall>=2 vs fall==1 (given fall>=1)
# Stage C: fall==3 vs fall==2 (given fall>=2)
CASCADE_STAGES = {"IG": ["A", "B", "C"], "NonIG": ["A", "B"], "Junk": ["A"]}

# Fixed (non-macro-sensitive) upgrade probabilities, one tranche up only.
# Flagged simplifying assumption: this thesis is scoped to downgrade
# sensitivity to rates/earnings; upgrade dynamics are held fixed so the
# estimation problem stays tractable with only N_TRAIN=48 quarters of data.
# IG has no upgrade destination (already top tranche).
UPGRADE_PROB = {"IG": 0.0, "NonIG": 0.04, "Junk": 0.05}

# ---------------------------------------------------------------------------
# Ground-truth Beta_true: one 4-vector [intercept, d_earnings, d_yield,
# interaction] per (tranche, stage). Signs are chosen to be economically
# sensible (earnings growth lowers downgrade odds, rising yields raise them)
# but the exact magnitudes are hardcoded synthetic ground truth by design.
# ---------------------------------------------------------------------------
BETA_TRUE = {
    "IG":    {"A": np.array([-5.4, -1.3, 1.0, -0.4]),
              "B": np.array([-1.8, -0.5, 0.6, -0.15]),
              "C": np.array([-2.3, -0.4, 0.5, -0.1])},
    "NonIG": {"A": np.array([-4.2, -1.1, 1.1, -0.35]),
              "B": np.array([-1.3, -0.45, 0.65, -0.15])},
    "Junk":  {"A": np.array([-2.6, -0.9, 1.25, -0.3])},
}

FEATURE_NAMES = ["X0_intercept", "X1_delta_earnings", "X2_delta_yield", "X3_interaction"]

# ---------------------------------------------------------------------------
# Observed-data simulation (Phat_t): finite company pool per source tranche
# per quarter, sampled from Ptrue via multinomial draws -> empirical
# transition frequencies with genuine sampling noise.
#
# Deliberately uniform across tranches, not a claim about real-world universe
# sizes: real corporate bond markets don't have equal numbers of IG, Non-IG,
# and Junk-rated issuers (most real universes skew investment-grade). Using
# the same pool size for all three tranches is a stated simplifying
# assumption, kept intentionally rather than calibrated to any actual
# rating-agency universe statistics.
# ---------------------------------------------------------------------------
COMPANY_POOL_SIZE = 3000  # companies "at risk" per source tranche per quarter

# Additive (Laplace-style) smoothing applied when computing log-odds from
# empirical counts, so that a zero count never produces ln(0) or ln(inf).
# Applied only at the feature-engineering stage, never baked into the raw
# stored Phat_t counts themselves.
SMOOTHING_EPSILON = 0.5

# Scenario-input capping: for exploratory "what if" predictions (Sheets,
# Excel, or a hand-typed Python scenario), macro inputs are clamped before
# being fed to the sigmoid math. Without this, a wildly out-of-range input
# pushes the logit so far into sigmoid saturation that the model returns
# literal 0%/100% -- a mathematically "valid" but economically absurd result
# (100% of a tranche demoted in one quarter has no historical precedent).
# Capping keeps extreme-but-plausible stress scenarios expressive while
# preventing nonsensical saturation; the displayed result becomes the
# model's response to the CAPPED input, which the caller must surface
# clearly, not the raw typed value. Applies only to prediction/forecasting
# (markov_engine.predict_scenario), never to the ground-truth DGP or the
# train/test evaluation on real data, both of which stay within the
# training range by construction.
#
# PER-FEATURE, not one shared multiplier: X1 (earnings) and X2 (yield) have
# very different fitted coefficient magnitudes, so the same multiple of
# their own training ranges saturates them at very different points --
# verified numerically (see README): earnings saturates by ~3x its training
# range, yield doesn't saturate until much further out. A single shared
# multiplier would either leave yield needlessly restricted or push
# earnings' cap somewhere it can no longer meaningfully differentiate
# anyway. Confirmed with the user rather than picked unilaterally.
EXTRAPOLATION_CAP_X1_MULTIPLE = 3.0  # earnings: 3x its own training range (symmetric)

# Yield: an EXPLICIT literal range, not a multiplier of the training range.
# A small multiplier already saturates on the upside (yield's coefficient is
# smaller than earnings', but the training range is even narrower), while a
# larger multiplier overshoots into saturation past ~12pp. These bounds are
# chosen from where the model is actually still informative, not derived
# mechanically from the training data.
EXTRAPOLATION_CAP_X2_BOUNDS = (-2.0, 10.0)  # percentage points

# Interaction term: needs its OWN cap, independent of X1/X2's -- capping the
# two factors alone doesn't sufficiently bound their product (verified
# numerically: the product of the widened X1/X2 ranges can still reach the
# high-20s/negative-30s, reintroducing full saturation at the corners).
# 15x its own training range keeps genuinely severe combined scenarios
# expressive without completely re-opening the saturation problem.
EXTRAPOLATION_CAP_X3_MULTIPLE = 15.0

# WLS weight cap: each stage's per-quarter weight (1/variance, from the
# delta-method variance of a log-ratio-of-counts) is winsorized at this
# multiple of that stage's own median weight. Without this, a single
# near-zero-count quarter -- most likely for the rarest cascade stages
# (Stage B/C) -- can get a wildly unstable weight estimate and dominate the
# fit, making WLS *worse* than plain OLS for exactly the stages it's meant
# to help. Verified empirically (see README) rather than assumed.
WLS_WEIGHT_CAP_MULTIPLE = 5.0
