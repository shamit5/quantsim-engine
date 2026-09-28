# QuantSim: Algorithmic Trading & Quantitative Simulation Engine

A modular Python engine for backtesting trading strategies and stress-testing them with Monte Carlo simulation.

## Features
- **Data ingestion:** synthetic regime-switching data (offline), CSV, or Yahoo Finance with retry, backoff and cache fallback.
- **Self-healing ETL** (`quantsim/etl.py`): repairs duplicate timestamps, missing values, non-positive prices and one-day price spikes (robust median/MAD z-score), then reports what it fixed.
- **Strategies:** SMA crossover, momentum, mean reversion (add your own in `strategies.py`).
- **Backtester:** vectorised, no look-ahead bias (signals execute next day), transaction costs and slippage.
- **Metrics:** CAGR, volatility, Sharpe, Sortino, max drawdown, Calmar.
- **Monte Carlo (NumPy/SciPy):** GBM and bootstrap simulation, VaR/CVaR, Jarque-Bera normality test, t-test on mean return.
- **Visualisation:** matplotlib dashboard and an interactive Streamlit app.
- **Tests:** pytest suite covering ETL repair, look-ahead bias, costs, drawdown and Monte Carlo.

## Run
```
pip install -r requirements.txt
python run_demo.py                      # offline demo on synthetic data
python run_demo.py --source yfinance --tickers AAPL MSFT GOOGL AMZN JPM
streamlit run app.py                    # interactive dashboard
pytest -q
```

## Notes and limitations
- Results on synthetic data only show the tooling works; they are not evidence of a real trading edge.
- Daily-bar backtest, equal-weight portfolio, no position sizing, borrow costs or intraday fills.
- Parameter sweeps are in-sample; use walk-forward validation before trusting any result.
- Not financial advice.
