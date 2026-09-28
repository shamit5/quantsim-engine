"""Plotting helpers (matplotlib)."""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .metrics import drawdown_series


def plot_equity(results: dict, ax=None):
    ax = ax or plt.gca()
    for name, res in results.items():
        ax.plot(res.equity.index, res.equity.values, label=name, lw=1.4)
    ax.set_title("Equity curves"); ax.set_ylabel("Portfolio value"); ax.legend(); ax.grid(alpha=.3)
    return ax


def plot_drawdowns(results: dict, ax=None):
    ax = ax or plt.gca()
    for name, res in results.items():
        dd = drawdown_series(res.returns)
        ax.plot(dd.index, dd.values * 100, label=name, lw=1.1)
    ax.set_title("Drawdown (%)"); ax.legend(); ax.grid(alpha=.3)
    return ax


def plot_fan(paths: np.ndarray, ax=None, title="Monte Carlo fan chart"):
    ax = ax or plt.gca()
    x = np.arange(paths.shape[0])
    q = np.percentile(paths, [5, 25, 50, 75, 95], axis=1)
    ax.fill_between(x, q[0], q[4], alpha=.2, label="5-95%")
    ax.fill_between(x, q[1], q[3], alpha=.35, label="25-75%")
    ax.plot(x, q[2], lw=1.5, label="median")
    ax.set_title(title); ax.set_xlabel("Days ahead"); ax.legend(); ax.grid(alpha=.3)
    return ax


def plot_heatmap(grid: pd.DataFrame, ax=None, title="Sharpe by parameters"):
    ax = ax or plt.gca()
    im = ax.imshow(grid.values, cmap="RdYlGn", aspect="auto")
    ax.set_xticks(range(grid.shape[1])); ax.set_xticklabels(grid.columns)
    ax.set_yticks(range(grid.shape[0])); ax.set_yticklabels(grid.index)
    ax.set_xlabel(str(grid.columns.name or "")); ax.set_ylabel(str(grid.index.name or "")); ax.set_title(title)
    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            ax.text(j, i, f"{grid.values[i, j]:.2f}", ha="center", va="center", fontsize=8)
    ax.figure.colorbar(im, ax=ax)
    return ax


def dashboard(results: dict, paths: np.ndarray, grid: pd.DataFrame, path: str):
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    plot_equity(results, axes[0, 0]); plot_drawdowns(results, axes[0, 1])
    plot_fan(paths, axes[1, 0]); plot_heatmap(grid, axes[1, 1])
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)
