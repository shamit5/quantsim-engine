"""End-to-end demo: ingest -> clean -> backtest -> sweep -> Monte Carlo -> dashboard.

    python run_demo.py                                   # synthetic data (offline)
    python run_demo.py --source yfinance --tickers AAPL MSFT GOOGL AMZN JPM --start 2015-01-01
    python run_demo.py --source csv --csv prices.csv
"""
import argparse
import json
import time
from pathlib import Path

import pandas as pd

from quantsim import (buy_and_hold, backtest_universe, bootstrap_paths, clean_prices, fetch_yfinance,
                      generate_synthetic_prices, inject_dirt, load_csv, return_diagnostics, risk_report,
                      summary, STRATEGIES)
from quantsim.viz import dashboard


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=["synthetic", "yfinance", "csv"], default="synthetic")
    ap.add_argument("--tickers", nargs="+", default=["AAPL", "MSFT", "GOOGL", "AMZN", "JPM"])
    ap.add_argument("--start", default="2015-01-01")
    ap.add_argument("--csv")
    ap.add_argument("--out", default="outputs")
    ap.add_argument("--cost-bps", type=float, default=5.0)
    args = ap.parse_args()
    out = Path(args.out); out.mkdir(exist_ok=True)

    t0 = time.perf_counter()
    if args.source == "synthetic":
        raw = inject_dirt(generate_synthetic_prices(n_assets=5, n_days=2520), seed=1, frac=0.01)
    elif args.source == "yfinance":
        raw = fetch_yfinance(args.tickers, args.start)
    else:
        raw = load_csv(args.csv)

    prices, rep = clean_prices(raw)
    t_etl = time.perf_counter() - t0
    print("== ETL report ==")
    print(json.dumps(rep.as_dict(), indent=2))
    print(f"ETL time: {t_etl:.3f}s  ({rep.records_in / t_etl:,.0f} records/s)\n")

    t1 = time.perf_counter()
    results = {"buy_and_hold": buy_and_hold(prices)}
    params = {"sma_crossover": dict(fast=20, slow=100), "momentum": dict(lookback=120),
              "mean_reversion": dict(window=20, entry_z=2.0, exit_z=0.5)}
    for name, fn in STRATEGIES.items():
        results[name] = backtest_universe(prices, fn, name=name, cost_bps=args.cost_bps, **params[name])
    t_bt = time.perf_counter() - t1

    table = pd.DataFrame({k: summary(v.returns) for k, v in results.items()}).T
    table["Trades"] = [v.n_trades for v in results.values()]
    print("== Performance ==")
    print(table.round(3).to_string(), "\n")
    print(f"Backtest time (3 strategies x {prices.shape[1]} assets): {t_bt:.3f}s\n")
    table.to_csv(out / "performance.csv")

    fasts, slows = [5, 10, 20, 30, 40], [50, 80, 100, 150, 200]
    grid = pd.DataFrame(index=pd.Index(fasts, name="fast"), columns=pd.Index(slows, name="slow"), dtype=float)
    for f in fasts:
        for s in slows:
            r = backtest_universe(prices, STRATEGIES["sma_crossover"], cost_bps=args.cost_bps, fast=f, slow=s)
            grid.loc[f, s] = summary(r.returns)["Sharpe"]
    grid.to_csv(out / "sma_sweep_sharpe.csv")

    best = table.drop("buy_and_hold")["Sharpe"].idxmax()
    paths = bootstrap_paths(results[best].returns, horizon=252, n_paths=5000, seed=7)
    print(f"== Monte Carlo (5000 bootstrap paths, 1y) for best-Sharpe strategy: {best} ==")
    print(json.dumps({k: round(v, 4) for k, v in risk_report(paths).items()}, indent=2))
    print("\n== Return diagnostics (SciPy) ==")
    print(json.dumps({k: round(v, 4) for k, v in return_diagnostics(results[best].returns).items()}, indent=2))

    dashboard(results, paths, grid, str(out / "dashboard.png"))
    print(f"\nSaved dashboard.png, performance.csv, sma_sweep_sharpe.csv to {out}/")


if __name__ == "__main__":
    main()
