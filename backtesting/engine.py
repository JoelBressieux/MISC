"""Backtesting engine that glues data, strategy and reporting together."""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from typing import Dict, Iterable, Tuple

import pandas as pd

from .data import Universe, download_price_history
from .metrics import compute_yearly_returns


@dataclass
class BacktestResult:
    """Results container returned by :func:`run_backtest`.

    Attributes
    ----------
    strategy_returns:
        Daily returns of the strategy being evaluated.
    benchmark_returns:
        Daily returns of the benchmark (S&P 500).
    weights:
        Daily weights used by the strategy (forward-filled between rebalances).
    metadata:
        Extra information such as the rebalance dates. Mostly here so we can
        stash additional goodies later without refactoring.
    """

    strategy_returns: pd.Series
    benchmark_returns: pd.Series
    weights: pd.DataFrame
    metadata: Dict[str, object]


def align_weights_to_daily_frequency(weights: pd.DataFrame, daily_index: pd.Index) -> pd.DataFrame:
    """Forward-fill monthly weights to daily frequency."""

    # Reindex ensures we have rows for every trading day. Forward filling keeps
    # the previous allocation active until the next rebalance. Backward fill is
    # only used to populate any initial NaNs (e.g. before the first rebalance).
    expanded = weights.reindex(daily_index, method="ffill")
    expanded = expanded.fillna(method="bfill").fillna(0.0)
    return expanded


def compute_portfolio_returns(returns: pd.DataFrame, weights: pd.DataFrame) -> pd.Series:
    """Compute daily portfolio returns given daily asset returns and weights."""

    # Element-wise multiply and then sum across columns to get the total return
    # for each day. ``mul`` handles the broadcasting nicely and is faster than a
    # for-loop. Pandas is basically vectorised hugs for quants.
    return (returns.mul(weights)).sum(axis=1)


def run_backtest(
    universe: Universe,
    prices: pd.DataFrame,
    strategy_weights: pd.DataFrame,
    start: dt.datetime,
    end: dt.datetime,
) -> BacktestResult:
    """Run a backtest for the supplied strategy weights."""

    # Slice the period of interest. ``loc`` plays nice with date indices.
    prices = prices.loc[start:end]

    # Compute daily returns. ``pct_change`` gives us percentage moves, and we
    # drop the initial NaN because nobody wants a mysterious blank return.
    daily_returns = prices.pct_change().dropna(how="all")

    # Align strategy weights to daily frequency so that every day has a weight
    # vector describing how our imaginary trader is allocated.
    aligned_weights = align_weights_to_daily_frequency(strategy_weights, daily_returns.index)

    # Strategy daily returns.
    strategy_returns = compute_portfolio_returns(daily_returns, aligned_weights)

    # Benchmark returns: fetch the index, compute returns, align to trading days.
    benchmark_prices = download_price_history([universe.benchmark], start=start, end=end)
    benchmark_prices = benchmark_prices.reindex(daily_returns.index).fillna(method="ffill")
    benchmark_returns = benchmark_prices.pct_change().iloc[:, 0].fillna(0.0)

    return BacktestResult(
        strategy_returns=strategy_returns,
        benchmark_returns=benchmark_returns.loc[strategy_returns.index],
        weights=aligned_weights,
        metadata={
            "start": start,
            "end": end,
            "rebalance_dates": strategy_weights.index,
        },
    )


__all__ = ["BacktestResult", "run_backtest"]
