"""QuantSim: a modular backtesting and Monte Carlo simulation engine."""
from .data import generate_synthetic_prices, inject_dirt, fetch_yfinance, load_csv
from .etl import clean_prices, CleaningReport
from .strategies import STRATEGIES, sma_crossover, momentum, mean_reversion
from .backtest import backtest, backtest_universe, buy_and_hold, BacktestResult
from .metrics import summary, max_drawdown, sharpe, cagr
from .montecarlo import simulate_gbm, bootstrap_paths, risk_report, return_diagnostics

__version__ = "1.0.0"
