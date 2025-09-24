"""Utilities for fetching S&P 500 data.

This module centralises all data access code so that the rest of the
backtesting stack can pretend the data magically appears. Having the data
plumbing in one place also makes it easier to swap out the source later on
(e.g. if we get fancy and pay for a data feed instead of mooching the free
internet)."""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from functools import lru_cache
from typing import Iterable, List

import pandas as pd
import yfinance as yf


# ----------------------------------------------------------------------------
# Data containers
# ----------------------------------------------------------------------------

@dataclass(frozen=True)
class Universe:
    """Simple container describing our investment universe.

    Attributes
    ----------
    tickers:
        A list of ticker symbols representing the investable assets. In our
        case this will typically hold the constituents of the S&P 500 index.
    benchmark:
        The ticker used as the benchmark index, by default the S&P 500 itself.
    """

    tickers: List[str]
    benchmark: str = "^GSPC"


# ----------------------------------------------------------------------------
# Universe helpers
# ----------------------------------------------------------------------------


@lru_cache(maxsize=1)
def fetch_sp500_universe() -> Universe:
    """Fetch the latest S&P 500 constituents from Wikipedia.

    Using :func:`pandas.read_html` keeps the dependency surface tiny and avoids
    storing a static CSV that would go stale faster than a forgotten sandwich.
    The result is cached via :func:`functools.lru_cache` because the S&P 500
    rarely changes more than a handful of tickers in a single day, and it's
    polite to not hammer Wikipedia like a trading bot on caffeine.
    """

    # The table we want lives on the first HTML table of the Wikipedia page.
    tables = pd.read_html("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies")
    sp500_table = tables[0]
    tickers = sp500_table["Symbol"].tolist()

    # Some tickers contain dots (e.g. BRK.B). Yahoo Finance expects them to be
    # replaced by hyphens. We do that conversion here so the rest of the stack
    # can remain blissfully unaware of such punctuation drama.
    tickers = [ticker.replace(".", "-") for ticker in tickers]
    return Universe(tickers=tickers)


# ----------------------------------------------------------------------------
# Price helpers
# ----------------------------------------------------------------------------


def download_price_history(
    tickers: Iterable[str],
    start: dt.datetime,
    end: dt.datetime,
) -> pd.DataFrame:
    """Download daily adjusted close prices for the provided tickers.

    Parameters
    ----------
    tickers:
        Iterable of ticker symbols to fetch.
    start, end:
        Date range for the historical data. ``end`` is inclusive because
        ``yfinance`` interprets it that way when we specify ``auto_adjust``.

    Returns
    -------
    pandas.DataFrame
        DataFrame indexed by date containing the adjusted close prices.

    Notes
    -----
    ``yfinance`` returns a DataFrame with multiple columns when fetching
    several tickers. We request the ``Adj Close`` column and then squeeze it to
    a nice flat structure.
    """

    data = yf.download(
        tickers=list(tickers),
        start=start,
        end=end,
        auto_adjust=True,
        progress=False,
        threads=True,
    )

    # ``yfinance`` returns a multi-index column when multiple tickers are
    # fetched. For a single ticker it just returns a Series. Either way we want
    # a DataFrame with tickers along the columns so we do some gentle coercion.
    if isinstance(data, pd.DataFrame) and "Adj Close" in data.columns:
        prices = data["Adj Close"].copy()
    else:
        prices = data.copy()

    if isinstance(prices, pd.Series):
        prices = prices.to_frame(name=list(tickers)[0])

    # Ensure chronological order and drop any stray NaNs at the top caused by
    # incomplete history.
    prices = prices.sort_index()
    prices = prices.loc[~prices.index.duplicated(keep="first")]
    return prices


__all__ = [
    "Universe",
    "fetch_sp500_universe",
    "download_price_history",
]
