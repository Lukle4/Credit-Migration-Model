"""
Step 1a: macro feature simulation.

Generates a quarterly 10Y UST yield series and a quarterly corporate earnings
growth series as mean-reverting AR(1) processes, with earnings carrying an
explicit negative sensitivity to yield changes. A burn-in period is simulated
and discarded to remove start-up transient bias before the analysis window
begins.
"""

import numpy as np
import pandas as pd

import config


def simulate_macro_paths(seed: int = config.RANDOM_SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    total_periods = config.N_BURNIN + config.N_QUARTERS
    yield_level = np.empty(total_periods + 1)
    earnings_growth = np.empty(total_periods + 1)
    yield_level[0] = config.YIELD_START
    earnings_growth[0] = config.EARNINGS_START

    yield_shocks = rng.normal(0.0, config.YIELD_SHOCK_STD, size=total_periods)
    earnings_shocks = rng.normal(0.0, config.EARNINGS_SHOCK_STD, size=total_periods)

    # Deterministic cyclical long-run target the yield mean-reverts toward
    # (see config.py for why this can't be a flat constant).
    t_idx = np.arange(total_periods)
    cycle_target = config.YIELD_CYCLE_BASE + config.YIELD_CYCLE_AMPLITUDE * np.sin(
        2 * np.pi * t_idx / config.YIELD_CYCLE_PERIOD_QUARTERS
    )

    for t in range(total_periods):
        dy = config.YIELD_MEAN_REVERSION_SPEED * (cycle_target[t] - yield_level[t]) \
             + yield_shocks[t]
        yield_level[t + 1] = yield_level[t] + dy

        de = config.EARNINGS_MEAN_REVERSION_SPEED * (config.EARNINGS_LONG_RUN_MEAN - earnings_growth[t]) \
             + config.EARNINGS_YIELD_SENSITIVITY * dy \
             + earnings_shocks[t]
        earnings_growth[t + 1] = earnings_growth[t] + de

    # Drop burn-in; keep N_QUARTERS+1 levels so we can take N_QUARTERS diffs.
    yield_level = yield_level[config.N_BURNIN:]
    earnings_growth = earnings_growth[config.N_BURNIN:]

    delta_yield = np.diff(yield_level)
    delta_earnings = np.diff(earnings_growth)

    df = pd.DataFrame({
        "quarter": np.arange(1, config.N_QUARTERS + 1),
        "yield_level": yield_level[1:],
        "earnings_growth_level": earnings_growth[1:],
        "delta_yield": delta_yield,
        "delta_earnings": delta_earnings,
    })
    return df


def noise_budget_report(df: pd.DataFrame) -> dict:
    """
    Diagnostic report flagging how much of the quarter-to-quarter variation
    is idiosyncratic noise vs. explained by the mean-reversion signal, plus
    a count of outlier quarters. Printed by run_pipeline.py so noise
    contamination is auditable rather than just asserted.
    """
    report = {}
    for col, shock_std, series_name in [
        ("delta_yield", config.YIELD_SHOCK_STD, "yield"),
        ("delta_earnings", config.EARNINGS_SHOCK_STD, "earnings"),
    ]:
        series = df[col].to_numpy()
        total_std = series.std(ddof=1)
        # idiosyncratic-shock-std / total-std: lower = signal-dominated, as intended
        noise_ratio = shock_std / total_std if total_std > 0 else np.nan
        z = (series - series.mean()) / total_std if total_std > 0 else np.zeros_like(series)
        outlier_mask = np.abs(z) > config.OUTLIER_Z_THRESHOLD
        report[series_name] = {
            "idiosyncratic_shock_std": shock_std,
            "realized_total_std": total_std,
            "noise_to_total_std_ratio": noise_ratio,
            "n_outlier_quarters": int(outlier_mask.sum()),
            "outlier_quarters": df.loc[outlier_mask, "quarter"].tolist(),
            "min": series.min(),
            "max": series.max(),
        }
    return report


if __name__ == "__main__":
    macro_df = simulate_macro_paths()
    print(macro_df.head())
    print(noise_budget_report(macro_df))
