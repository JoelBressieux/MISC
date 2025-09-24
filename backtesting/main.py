"""Entry point for running the backtest demo from the command line."""

from __future__ import annotations

import argparse
import datetime as dt
from pathlib import Path

import numpy as np
import pandas as pd

from .data import download_price_history, fetch_sp500_universe
from .engine import run_backtest
from .plotting import plot_cumulative_performance
from .reporting import compile_performance_tables
from .strategies import OneYearMomentum


DEFAULT_YEARS = 10


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments just in case we feel adventurous later."""

    parser = argparse.ArgumentParser(description="Run the demo backtest.")
    parser.add_argument(
        "--years",
        type=int,
        default=DEFAULT_YEARS,
        help="Number of years of history to include. More history, more drama.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("output"),
        help="Where to stash plots and tables. Bring your own treasure chest.",
    )
    return parser.parse_args()


def compute_start_end_dates(years: int) -> tuple[dt.datetime, dt.datetime]:
    """Helper returning the start and end dates for the backtest."""

    end = dt.datetime.today()
    start = end - dt.timedelta(days=365 * years + 400)  # generous cushion for lookback
    return start, end


def main() -> None:
    """Run a 12-month momentum strategy and compare it to the S&P 500."""

    args = parse_args()
    start, end = compute_start_end_dates(args.years)

    print("Fetching the S&P 500 universe. Stand by for corporate alphabet soup...")
    universe = fetch_sp500_universe()

    print(f"Universe size: {len(universe.tickers)} tickers. That's a lot of tickers.")

    print("Downloading price history from Yahoo Finance. Coffee break recommended.")
    prices = download_price_history(universe.tickers, start=start, end=end)

    print("Generating strategy weights using the 12-month momentum wizardry.")
    strategy = OneYearMomentum()
    strategy_weights = strategy.generate_weights(prices)

    print("Running the backtest. Buckle up, volatility incoming.")
    result = run_backtest(universe, prices, strategy_weights, start=start, end=end)

    print("Compiling performance tables. Numbers incoming!")
    tables = compile_performance_tables(result.strategy_returns, result.benchmark_returns)

    summary_table = tables["summary"].applymap(lambda x: f"{x:.2%}" if np.isfinite(x) else "N/A")
    yearly_table = tables["yearly"].applymap(lambda x: f"{x:.2%}" if np.isfinite(x) else "N/A")

    print("\n=== Performance Summary ===")
    print(summary_table.to_markdown())

    print("\n=== Yearly Returns ===")
    print(yearly_table.to_markdown())

    cumulative = pd.DataFrame(
        {
            "Momentum": (1 + result.strategy_returns).cumprod(),
            "S&P 500": (1 + result.benchmark_returns).cumprod(),
        }
    )

    plot_path = args.output / "cumulative_performance.png"
    print(f"Saving cumulative performance chart to {plot_path} (frame it if it looks nice).")
    plot_cumulative_performance(cumulative, plot_path)

    print("All done! Feel free to experiment with other strategies later. Bring snacks.")


if __name__ == "__main__":
    main()
