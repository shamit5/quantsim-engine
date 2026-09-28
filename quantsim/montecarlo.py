"""Monte Carlo simulation and statistical diagnostics (NumPy + SciPy)."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

TRADING_DAYS = 252


def simulate_gbm(s0: float, mu: float, sigma: float, horizon: int, n_paths: int, seed: int = 0) -> np.ndarray:
    """Geometric Brownian motion paths, shape (horizon + 1, n_paths). mu and sigma are annualised."""
    rng = np.random.default_rng(seed)
    dt = 1 / TRADING_DAYS
    z = rng.standard_normal((horizon, n_paths))
    log_steps = (mu - 0.5 * sigma ** 2) * dt + sigma * np.sqrt(dt) * z
    paths = s0 * np.exp(np.cumsum(log_steps, axis=0))
    return np.vstack([np.full(n_paths, s0), paths])


def bootstrap_paths(returns: pd.Series | np.ndarray, horizon: int, n_paths: int, seed: int = 0,
                    initial: float = 1.0) -> np.ndarray:
    """Resample historical returns with replacement to build equity paths, shape (horizon + 1, n_paths)."""
    r = np.asarray(returns, dtype=float)
    rng = np.random.default_rng(seed)
    sampled = r[rng.integers(0, len(r), size=(horizon, n_paths))]
    paths = initial * np.cumprod(1 + sampled, axis=0)
    return np.vstack([np.full(n_paths, initial), paths])


def risk_report(paths: np.ndarray, alpha: float = 0.05) -> dict:
    """Terminal-return risk statistics from simulated paths."""
    final_ret = paths[-1] / paths[0] - 1
    var = -np.percentile(final_ret, alpha * 100)
    tail = final_ret[final_ret <= -var]
    cvar = -tail.mean() if len(tail) else var
    return {
        "MedianReturn": float(np.median(final_ret)),
        "MeanReturn": float(final_ret.mean()),
        f"VaR{int((1 - alpha) * 100)}": float(var),
        f"CVaR{int((1 - alpha) * 100)}": float(cvar),
        "ProbLoss": float((final_ret < 0).mean()),
    }


def return_diagnostics(returns: pd.Series) -> dict:
    """Is the return stream normal, and is its mean distinguishable from zero?"""
    r = returns.dropna()
    jb_stat, jb_p = stats.jarque_bera(r)
    t_stat, t_p = stats.ttest_1samp(r, 0.0)
    mu, sd = stats.norm.fit(r)
    return {"skew": float(stats.skew(r)), "excess_kurtosis": float(stats.kurtosis(r)),
            "jarque_bera_p": float(jb_p), "mean_t_stat": float(t_stat), "mean_t_pvalue": float(t_p),
            "fit_mu": float(mu), "fit_sigma": float(sd)}
