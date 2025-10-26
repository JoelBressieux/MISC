"""Backtesting playground for experimenting with investment strategies."""

from .data import (
    Universe,
    fetch_sp500_universe_cached,
    fetch_sp500_universe,   # alias for convenience
    download_price_history,
)
from .main import compute_start_end_dates
from .strategies import OneYearMomentum
from .engine import run_backtest
from .reporting import compile_performance_tables
from .plotting import plot_cumulative_performance

__all__ = [
    "Universe",
    "fetch_sp500_universe_cached",
    "fetch_sp500_universe",
    "download_price_history",
    "compute_start_end_dates",
    "OneYearMomentum",
    "run_backtest",
    "compile_performance_tables",
    "plot_cumulative_performance",
]