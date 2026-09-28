"""Self-healing ETL: validate, repair, and report on raw price data."""
from __future__ import annotations

from dataclasses import dataclass, asdict, field

import numpy as np
import pandas as pd


@dataclass
class CleaningReport:
    records_in: int = 0
    rows_in: int = 0
    duplicates_removed: int = 0
    non_positive_fixed: int = 0
    spikes_fixed: int = 0
    gaps_filled: int = 0
    columns_dropped: list = field(default_factory=list)
    rows_dropped: int = 0
    records_out: int = 0

    def as_dict(self) -> dict:
        return asdict(self)


def clean_prices(df: pd.DataFrame, max_gap: int = 5, spike_z: float = 8.0,
                 max_missing_frac: float = 0.2) -> tuple[pd.DataFrame, CleaningReport]:
    """Return (clean_prices, report).

    Steps: coerce types -> parse/sort dates -> drop duplicate timestamps -> mask invalid
    (<= 0) prices -> detect one-day spikes with a robust (median/MAD) z-score on log
    returns -> drop columns that are mostly missing -> interpolate short gaps ->
    drop whatever cannot be repaired safely.
    """
    rep = CleaningReport(records_in=int(df.size), rows_in=len(df))
    x = df.apply(pd.to_numeric, errors="coerce")
    x.index = pd.to_datetime(x.index, errors="coerce")
    x = x[~x.index.isna()].sort_index()

    rep.duplicates_removed = int(x.index.duplicated().sum())
    x = x[~x.index.duplicated(keep="last")]

    rep.non_positive_fixed = int((x <= 0).sum().sum())
    x = x.mask(x <= 0)

    # Spike = large return immediately reversed by an equally large opposite return.
    logret = np.log(x).diff()
    med = logret.rolling(60, min_periods=10, center=True).median()
    mad = (logret - med).abs().rolling(60, min_periods=10, center=True).median() * 1.4826
    z = (logret - med) / mad.replace(0, np.nan)
    spike = (z.abs() > spike_z) & (z.shift(-1).abs() > spike_z) & (np.sign(z) != np.sign(z.shift(-1)))
    rep.spikes_fixed = int(spike.sum().sum())
    x = x.mask(spike)

    missing_frac = x.isna().mean()
    drop_cols = missing_frac[missing_frac > max_missing_frac].index.tolist()
    rep.columns_dropped = drop_cols
    x = x.drop(columns=drop_cols)

    before = int(x.isna().sum().sum())
    x = x.interpolate(method="time", limit=max_gap, limit_area="inside")
    rep.gaps_filled = before - int(x.isna().sum().sum())

    n_before = len(x)
    x = x.dropna(how="any")
    rep.rows_dropped = n_before - len(x)
    rep.records_out = int(x.size)
    return x, rep
