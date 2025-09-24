"""Turn raw backtest data into human-friendly tables."""

from __future__ import annotations

import pandas as pd

from .metrics import compute_yearly_returns, summarize_performance


def build_summary_table(results: pd.Series) -> pd.DataFrame:
    """Convert summary statistics into a DataFrame for pretty printing."""

    return pd.DataFrame(results).T


def compile_performance_tables(strategy_returns: pd.Series, benchmark_returns: pd.Series) -> dict:
    """Create summary and yearly performance tables for reporting."""

    summary = pd.concat(
        [
            summarize_performance(strategy_returns).rename("Momentum"),
            summarize_performance(benchmark_returns).rename("S&P 500"),
        ],
        axis=1,
    ).T

    yearly = pd.concat(
        [
            compute_yearly_returns(strategy_returns).rename("Momentum"),
            compute_yearly_returns(benchmark_returns).rename("S&P 500"),
        ],
        axis=1,
    )
    yearly.index = yearly.index.year

    return {"summary": summary, "yearly": yearly}


__all__ = ["compile_performance_tables"]
