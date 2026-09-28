"""Vectorised backtester with transaction costs and slippage (no look-ahead bias)."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import pandas as pd


@dataclass
class BacktestResult:
    name: str
    returns: pd.Series
    positions: pd.Series | pd.DataFrame
    n_trades: int
    initial_capital: float = 100_000.0
    per_asset: dict = field(default_factory=dict)

    @property
    def equity(self) -> pd.Series:
        return self.initial_capital * (1 + self.returns).cumprod()


def backtest(prices: pd.Series, signal: pd.Series, cost_bps: float = 5.0, slippage_bps: float = 2.0,
             initial_capital: float = 100_000.0, name: str = "strategy") -> BacktestResult:
    """Run one asset. The signal from day t is executed on day t+1 (shift) so the
    strategy never trades on information it would not have had yet."""
    ret = prices.pct_change().fillna(0.0)
    pos = signal.reindex(prices.index).fillna(0.0).shift(1).fillna(0.0)
    turnover = pos.diff().abs().fillna(0.0)
    costs = turnover * (cost_bps + slippage_bps) / 1e4
    strat_ret = pos * ret - costs
    return BacktestResult(name, strat_ret, pos, int((turnover > 0).sum()), initial_capital)


def backtest_universe(prices: pd.DataFrame, strategy: Callable[..., pd.Series], name: str | None = None,
                      cost_bps: float = 5.0, slippage_bps: float = 2.0, initial_capital: float = 100_000.0,
                      **params) -> BacktestResult:
    """Run a strategy on every column and hold the assets at equal weight."""
    results = {c: backtest(prices[c], strategy(prices[c], **params), cost_bps, slippage_bps,
                           initial_capital, name=c) for c in prices.columns}
    rets = pd.concat({c: r.returns for c, r in results.items()}, axis=1).mean(axis=1)
    pos = pd.concat({c: r.positions for c, r in results.items()}, axis=1)
    trades = sum(r.n_trades for r in results.values())
    return BacktestResult(name or strategy.__name__, rets, pos, trades, initial_capital, results)


def buy_and_hold(prices: pd.DataFrame, initial_capital: float = 100_000.0) -> BacktestResult:
    rets = prices.pct_change().fillna(0.0).mean(axis=1)
    return BacktestResult("buy_and_hold", rets, pd.DataFrame(1.0, index=prices.index, columns=prices.columns),
                          0, initial_capital)
