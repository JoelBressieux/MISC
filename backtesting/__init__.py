"""Backtesting playground for experimenting with investment strategies."""

from .data import Universe, download_price_history, fetch_sp500_universe
from .engine import BacktestResult, run_backtest
from .metrics import (
    compute_cagr,
    compute_max_drawdown,
    compute_sharpe_ratio,
    compute_volatility,
    compute_yearly_returns,
    summarize_performance,
)
from .plotting import plot_cumulative_performance
from .reporting import compile_performance_tables
from .strategies import OneYearMomentum, Strategy

__all__ = [
    "BacktestResult",
    "OneYearMomentum",
    "Strategy",
    "Universe",
    "compile_performance_tables",
    "compute_cagr",
    "compute_max_drawdown",
    "compute_sharpe_ratio",
    "compute_volatility",
    "compute_yearly_returns",
    "download_price_history",
    "fetch_sp500_universe",
    "plot_cumulative_performance",
    "run_backtest",
    "summarize_performance",
]
