"""Trading signals. Each function maps a price Series to a position Series in [-1, 1]."""
from __future__ import annotations

import numpy as np
import pandas as pd


def sma_crossover(prices: pd.Series, fast: int = 20, slow: int = 100, long_only: bool = True) -> pd.Series:
    """Long when the fast moving average is above the slow one."""
    if fast >= slow:
        raise ValueError("fast window must be smaller than slow window")
    f, s = prices.rolling(fast).mean(), prices.rolling(slow).mean()
    sig = (f > s).astype(float)
    if not long_only:
        sig = sig * 2 - 1
    return sig.where(s.notna(), 0.0)


def momentum(prices: pd.Series, lookback: int = 120, long_only: bool = True) -> pd.Series:
    """Follow the sign of the trailing return."""
    sig = np.sign(prices.pct_change(lookback))
    if long_only:
        sig = sig.clip(lower=0)
    return sig.fillna(0.0)


def mean_reversion(prices: pd.Series, window: int = 20, entry_z: float = 2.0, exit_z: float = 0.5) -> pd.Series:
    """Buy when price is far below its rolling mean, sell when far above; exit near the mean."""
    z = ((prices - prices.rolling(window).mean()) / prices.rolling(window).std()).to_numpy()
    pos = np.zeros(len(z))
    state = 0
    for i, zi in enumerate(z):
        if np.isnan(zi):
            state = 0
        elif state == 0:
            state = 1 if zi < -entry_z else (-1 if zi > entry_z else 0)
        elif state == 1 and zi >= -exit_z:
            state = 0
        elif state == -1 and zi <= exit_z:
            state = 0
        pos[i] = state
    return pd.Series(pos, index=prices.index)


STRATEGIES = {
    "sma_crossover": sma_crossover,
    "momentum": momentum,
    "mean_reversion": mean_reversion,
}
