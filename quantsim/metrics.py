"""Performance and risk metrics computed from a periodic return series."""
from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = 252


def equity_curve(returns: pd.Series) -> pd.Series:
    return (1 + returns).cumprod()


def cagr(returns: pd.Series) -> float:
    if len(returns) == 0:
        return 0.0
    return float(equity_curve(returns).iloc[-1] ** (TRADING_DAYS / len(returns)) - 1)


def annual_vol(returns: pd.Series) -> float:
    return float(returns.std(ddof=1) * np.sqrt(TRADING_DAYS))


def sharpe(returns: pd.Series, rf: float = 0.0) -> float:
    ex = returns - rf / TRADING_DAYS
    sd = ex.std(ddof=1)
    return float(ex.mean() / sd * np.sqrt(TRADING_DAYS)) if sd > 0 else 0.0


def sortino(returns: pd.Series, rf: float = 0.0) -> float:
    ex = returns - rf / TRADING_DAYS
    downside = ex[ex < 0].std(ddof=1)
    return float(ex.mean() / downside * np.sqrt(TRADING_DAYS)) if downside and downside > 0 else 0.0


def drawdown_series(returns: pd.Series) -> pd.Series:
    eq = equity_curve(returns)
    return eq / eq.cummax() - 1


def max_drawdown(returns: pd.Series) -> float:
    return float(drawdown_series(returns).min())


def summary(returns: pd.Series) -> dict:
    mdd = max_drawdown(returns)
    c = cagr(returns)
    return {
        "CAGR": c,
        "Volatility": annual_vol(returns),
        "Sharpe": sharpe(returns),
        "Sortino": sortino(returns),
        "MaxDrawdown": mdd,
        "Calmar": c / abs(mdd) if mdd < 0 else 0.0,
        "TotalReturn": float(equity_curve(returns).iloc[-1] - 1),
    }
