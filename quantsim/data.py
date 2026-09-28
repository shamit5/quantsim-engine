"""Data sources: synthetic regime-switching GBM, CSV files, and Yahoo Finance (with retry + cache)."""
from __future__ import annotations

import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)


def generate_synthetic_prices(n_assets: int = 5, n_days: int = 2520, seed: int = 42,
                              start: str = "2015-01-01") -> pd.DataFrame:
    """Correlated geometric Brownian motion with a shared bull/bear drift regime.

    The regime switching gives trend-following strategies something to find, while
    keeping the data fully reproducible for testing.
    """
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(start, periods=n_days)

    # Shared two-state Markov regime (0 = bull, 1 = bear)
    regime = np.zeros(n_days, dtype=int)
    for t in range(1, n_days):
        flip = rng.random() < 0.006
        regime[t] = 1 - regime[t - 1] if flip else regime[t - 1]
    regime_drift = np.where(regime == 0, 0.20, -0.15) / 252

    base_mu = rng.uniform(-0.02, 0.05, n_assets) / 252
    sigma = rng.uniform(0.15, 0.35, n_assets) / np.sqrt(252)

    a = rng.normal(size=(n_assets, n_assets))
    corr = a @ a.T
    d = np.sqrt(np.diag(corr))
    corr = corr / np.outer(d, d)
    chol = np.linalg.cholesky(0.3 * corr + 0.7 * np.eye(n_assets))
    shocks = rng.normal(size=(n_days, n_assets)) @ chol.T

    log_ret = (base_mu + regime_drift[:, None]) - 0.5 * sigma ** 2 + sigma * shocks
    prices = 100 * np.exp(np.cumsum(log_ret, axis=0))
    return pd.DataFrame(prices, index=dates, columns=[f"ASSET_{i + 1}" for i in range(n_assets)])


def inject_dirt(df: pd.DataFrame, seed: int = 0, frac: float = 0.01) -> pd.DataFrame:
    """Corrupt a clean price table (missing values, spikes, bad prices, duplicate rows)
    so the ETL pipeline has real problems to repair."""
    rng = np.random.default_rng(seed)
    out = df.copy()
    n, m = out.shape
    k = max(1, int(n * m * frac))
    vals = out.to_numpy(copy=True)

    def pick(count):
        return rng.integers(1, n - 1, count), rng.integers(0, m, count)

    r, c = pick(k)                      # missing values
    vals[r, c] = np.nan
    r, c = pick(max(1, k // 5))         # one-day price spikes (x8 or /8)
    vals[r, c] = vals[r, c] * rng.choice([8.0, 0.125], size=len(r))
    r, c = pick(max(1, k // 10))        # invalid non-positive prices
    vals[r, c] = 0.0
    out = pd.DataFrame(vals, index=out.index, columns=out.columns)
    dup = out.iloc[rng.integers(0, n, max(1, k // 5))]   # duplicated timestamps
    return pd.concat([out, dup]).sample(frac=1.0, random_state=seed)  # shuffled order


def load_csv(path: str | Path) -> pd.DataFrame:
    """Load a wide price CSV (first column = date, remaining columns = tickers)."""
    return pd.read_csv(path, index_col=0)


def fetch_yfinance(tickers: list[str], start: str, end: str | None = None, retries: int = 3,
                   backoff: float = 2.0, cache_dir: str | Path = "data_cache") -> pd.DataFrame:
    """Download adjusted close prices with retry + linear backoff.
    Falls back to the last cached copy if every attempt fails."""
    cache = Path(cache_dir)
    cache.mkdir(parents=True, exist_ok=True)
    key = cache / ("_".join(sorted(tickers)) + ".csv")

    for attempt in range(1, retries + 1):
        try:
            import yfinance as yf
            df = yf.download(tickers, start=start, end=end, auto_adjust=True, progress=False)["Close"]
            if isinstance(df, pd.Series):
                df = df.to_frame(tickers[0])
            if df.empty:
                raise ValueError("empty download")
            df.to_csv(key)
            return df
        except Exception as exc:  # network errors, rate limits, bad symbols
            wait = backoff * attempt
            log.warning("fetch attempt %d/%d failed (%s)", attempt, retries, exc)
            if attempt < retries:
                time.sleep(wait)

    if key.exists():
        log.warning("using cached data from %s", key)
        return pd.read_csv(key, index_col=0)
    raise RuntimeError(f"could not fetch {tickers} and no cache is available")
