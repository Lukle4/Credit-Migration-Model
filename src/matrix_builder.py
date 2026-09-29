"""
Shared X-matrix builder (step 2c), used both by the ground-truth Markov DGP
(step 1b needs the exact same macro design matrix to generate Ptrue) and by
feature engineering (step 2, to build X_train/X_test for OLS). Keeping one
implementation guarantees beta_hat is estimated on the identical feature
definition beta_true was generated from.

X columns: X0 = intercept, X1 = delta_earnings, X2 = delta_yield,
X3 = X1 * X2 (interaction term -- NOT X1 - X2). The interaction form was
chosen specifically because X1 - X2 would be an exact linear combination of
existing columns, making (X'X) singular; the interaction term is a nonlinear
combination and keeps the design matrix full rank.

X1/X2 are expressed in PERCENTAGE POINTS (1.0 = 100bp), not raw decimals
(0.01 = 100bp). config.BETA_TRUE's magnitudes (~2-2.5) are calibrated for
this scale -- a 100bp yield move should shift a fall-vs-stay log-odds by an
economically meaningful ~2, not by 0.02. Feeding raw decimals in would
silently shrink the true signal to near-zero relative to sampling noise.
"""

import numpy as np
import pandas as pd


def build_feature_matrix(macro_df: pd.DataFrame) -> np.ndarray:
    x0 = np.ones(len(macro_df))
    x1 = macro_df["delta_earnings"].to_numpy() * 100.0
    x2 = macro_df["delta_yield"].to_numpy() * 100.0
    x3 = x1 * x2
    return np.column_stack([x0, x1, x2, x3])
