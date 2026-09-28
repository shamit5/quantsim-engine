import numpy as np
import pandas as pd
import pytest

from quantsim import (backtest, bootstrap_paths, clean_prices, generate_synthetic_prices, inject_dirt,
                      max_drawdown, mean_reversion, risk_report, simulate_gbm, sma_crossover)


def test_etl_repairs_dirty_data():
    clean = generate_synthetic_prices(n_assets=4, n_days=800, seed=3)
    dirty = inject_dirt(clean, seed=5, frac=0.02)
    fixed, rep = clean_prices(dirty)
    assert not fixed.isna().any().any()
    assert fixed.index.is_monotonic_increasing and not fixed.index.duplicated().any()
    assert (fixed > 0).all().all()
    assert rep.duplicates_removed > 0 and rep.non_positive_fixed > 0 and rep.gaps_filled > 0
    # repaired series stays close to the truth
    err = ((fixed / clean.reindex(fixed.index)) - 1).abs().to_numpy()
    assert np.nanmedian(err) < 1e-9 and np.nanmax(err) < 0.5


def test_backtest_has_no_lookahead():
    idx = pd.bdate_range("2020-01-01", periods=6)
    prices = pd.Series([100, 110, 121, 133.1, 146.41, 161.05], index=idx)
    res = backtest(prices, pd.Series(1.0, index=idx), cost_bps=0, slippage_bps=0)
    # a signal generated at the close of day 0 is only held from day 1, so day 0 earns nothing
    assert res.returns.iloc[0] == 0.0
    assert res.equity.iloc[-1] / res.initial_capital == pytest.approx(161.05 / 100, rel=1e-6)
    # a signal that "knows" the next day's move must not be able to trade it the same day
    rets = prices.pct_change().shift(-1).fillna(0)
    foresight = (rets > 0).astype(float)
    res2 = backtest(prices, foresight, cost_bps=0, slippage_bps=0)
    assert res2.returns.iloc[1] == pytest.approx(foresight.iloc[0] * prices.pct_change().iloc[1])


def test_costs_reduce_returns():
    p = generate_synthetic_prices(1, 500, seed=1)["ASSET_1"]
    sig = sma_crossover(p, 10, 50)
    free = backtest(p, sig, 0, 0).returns.sum()
    paid = backtest(p, sig, 25, 10).returns.sum()
    assert paid < free


def test_max_drawdown_known_value():
    r = pd.Series([0.10, -0.20, 0.05])          # peak 1.10 -> trough 0.88
    assert max_drawdown(r) == pytest.approx(-0.20)


def test_strategy_signal_ranges_and_validation():
    p = generate_synthetic_prices(1, 400, seed=2)["ASSET_1"]
    assert set(mean_reversion(p).unique()) <= {-1.0, 0.0, 1.0}
    with pytest.raises(ValueError):
        sma_crossover(p, fast=50, slow=20)


def test_monte_carlo_shapes_and_stats():
    g = simulate_gbm(100, 0.08, 0.2, horizon=252, n_paths=2000, seed=1)
    assert g.shape == (253, 2000) and (g > 0).all()
    b = bootstrap_paths(np.random.default_rng(0).normal(0.0005, 0.01, 1000), 100, 500)
    rep = risk_report(b)
    assert rep["CVaR95"] >= rep["VaR95"] and 0 <= rep["ProbLoss"] <= 1
