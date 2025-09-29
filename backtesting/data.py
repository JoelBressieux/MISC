"""Utilities for fetching S&P 500 data.

This module centralises all data access code so that the rest of the
backtesting stack can pretend the data magically appears. Having the data
plumbing in one place also makes it easier to swap out the source later on
(e.g. if we get fancy and pay for a data feed instead of mooching the free
internet)."""
# backtesting/data.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Dict

import pandas as pd
import requests
import yfinance as yf


# --------------------------- Public data containers ---------------------------

#from dataclasses import dataclass

@dataclass(frozen=True)
class Universe:
    tickers: list[str]
    metadata: pd.DataFrame
    benchmark: str = "SPY"   # <- default benchmark (use SPY; Stooq works too)


# --------------------------- Helpers (internal) -------------------------------

def _detect_symbol_column(df: pd.DataFrame) -> str:
    cand = {str(c).strip().lower(): c for c in df.columns}
    for key in ("symbol", "ticker symbol", "ticker"):
        if key in cand:
            return cand[key]
    raise RuntimeError("Could not find a ticker column (Symbol/Ticker) on the Wikipedia page.")

def _clean_to_yahoo_symbol(s: pd.Series) -> pd.Series:
    # Convert dots to Yahoo hyphen style: BRK.B -> BRK-B
    return (
        s.astype(str)
         .str.strip()
         .str.replace(".", "-", regex=False)
    )

def _fetch_sp500_from_wikipedia() -> pd.DataFrame:
    """
    Download the S&P 500 constituents table from Wikipedia with a proper User-Agent.
    Returns a DataFrame with a normalized 'Symbol' column (Yahoo style).
    """
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.0 Safari/605.1.15"
        )
    }
    resp = requests.get(url, headers=headers, timeout=20)
    resp.raise_for_status()

    tables = pd.read_html(resp.text)
    if not tables:
        raise RuntimeError("No tables found on the Wikipedia page.")

    df = tables[0].copy()
    sym_col = _detect_symbol_column(df)
    df.rename(columns={sym_col: "Symbol"}, inplace=True)
    df["Symbol"] = _clean_to_yahoo_symbol(df["Symbol"])
    df = (
        df.dropna(subset=["Symbol"])
          .drop_duplicates(subset=["Symbol"])
          .sort_values("Symbol")
          .reset_index(drop=True)
    )
    return df

def _to_list(x: Iterable[str] | str) -> List[str]:
    if isinstance(x, str):
        return [x]
    return list(x)

def _chunk(seq: List[str], n: int):
    for i in range(0, len(seq), n):
        yield seq[i:i + n]

def _read_parquet_or_csv(fp: Path) -> pd.DataFrame:
    if fp.suffix == ".parquet" and fp.exists():
        return pd.read_parquet(fp)
    if fp.with_suffix(".parquet").exists():
        return pd.read_parquet(fp.with_suffix(".parquet"))
    if fp.suffix == ".csv" and fp.exists():
        return pd.read_csv(fp, parse_dates=[0], index_col=0) if "cache" in fp.parts else pd.read_csv(fp)
    if fp.with_suffix(".csv").exists():
        alt = fp.with_suffix(".csv")
        return pd.read_csv(alt, parse_dates=[0], index_col=0) if "cache" in fp.parts else pd.read_csv(alt)
    raise FileNotFoundError(fp)

def _write_parquet_or_csv(df: pd.DataFrame, fp: Path) -> None:
    # Try Parquet first (pyarrow/fastparquet); if not available, fall back to CSV.
    try:
        fp = fp.with_suffix(".parquet")
        df.to_parquet(fp, index=True)
    except Exception:
        fp = fp.with_suffix(".csv")
        if isinstance(df.index, pd.DatetimeIndex):
            df.to_csv(fp, index=True, date_format="%Y-%m-%d")
        else:
            df.to_csv(fp, index=True)


# --------------------------- Public API: Universe -----------------------------

def fetch_sp500_universe_cached(
    cache_path: str | Path = "data/sp500_universe.csv",
    refresh: bool = False,
    min_count: int = 450,
) -> Universe:
    """
    Fetch the S&P 500 universe with on-disk caching.

    Parameters
    ----------
    cache_path : str | Path
        CSV/Parquet file for caching constituents (auto-created).
    refresh : bool
        If True, force a fresh download from Wikipedia.
    min_count : int
        Sanity check on number of rows before trusting the fresh download.

    Returns
    -------
    Universe(tickers, metadata)
    """
    p = Path(cache_path)
    if p.exists() and not refresh:
        df = _read_parquet_or_csv(p)
        if "Symbol" not in df.columns:
            # older cache: assume first column is Symbol
            first = df.columns[0]
            df = df.rename(columns={first: "Symbol"})
        tickers = df["Symbol"].astype(str).tolist()
        return Universe(tickers=tickers, metadata=df, benchmark="SPY")
    # fresh download
    try:
        df = _fetch_sp500_from_wikipedia()
        if len(df) < min_count:
            raise RuntimeError(f"Only {len(df)} rows fetched; aborting cache write.")
        p.parent.mkdir(parents=True, exist_ok=True)
        # Prefer CSV for portability here
        df.to_csv(p if p.suffix else p.with_suffix(".csv"), index=False)
        tickers = df["Symbol"].astype(str).tolist()
        return Universe(tickers=tickers, metadata=df)
    except Exception as e:
        if p.exists():
            df = _read_parquet_or_csv(p)
            tickers = df["Symbol"].astype(str).tolist()
            print(f"[fetch_sp500_universe_cached] Warning: {e}. Falling back to cached file at {p}.")
            return Universe(tickers=tickers, metadata=df, benchmark="SPY")
        raise

# Backward-compatible alias (so you can `from backtesting.data import fetch_sp500_universe`)
def fetch_sp500_universe(*args, **kwargs) -> Universe:
    return fetch_sp500_universe_cached(*args, **kwargs)


# --------------------------- Public API: Prices -------------------------------

def download_price_history(
    tickers: Iterable[str],
    start: str | pd.Timestamp,
    end: str | pd.Timestamp,
    cache_dir: str | Path = "data/cache/prices",
    refresh: bool = False,
    monthly: bool = True,
    max_retries: int = 3,
    sleep_seconds: float = 0.8,
    limit: int | None = None,
    min_ok: int = 25,
    source: str = "auto",  # "auto" | "yahoo" | "stooq"
) -> pd.DataFrame:
    """
    Robust per-ticker downloader with caching, retries, and a Stooq fallback.
    """
    import time
    import yfinance as yf
    import requests
    from pandas_datareader import data as pdr

    # --- setup ---
    tickers = [t.strip().upper() for t in _to_list(tickers)]
    if limit is not None:
        tickers = tickers[: int(limit)]
    start = pd.to_datetime(start).date()
    end = pd.to_datetime(end).date()
    cache = Path(cache_dir); cache.mkdir(parents=True, exist_ok=True)

    # persistent session with UA to reduce Yahoo “timezone” errors
    session = requests.Session()
    session.headers.update({
        "User-Agent": ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                       "AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.0 Safari/605.1.15")
    })

    frames: Dict[str, pd.DataFrame] = {}
    need: List[str] = []

    # --- load from cache first ---
    for t in tickers:
        base = cache / t
        if not refresh:
            for ext in (".parquet", ".csv"):
                fp = base.with_suffix(ext)
                if fp.exists():
                    try:
                        df = pd.read_parquet(fp) if ext == ".parquet" else pd.read_csv(fp, parse_dates=[0], index_col=0)
                        if not df.empty:
                            frames[t] = df
                            break
                    except Exception:
                        pass
            else:
                need.append(t)
        else:
            need.append(t)

    # --- single-ticker downloaders ---
    def _dl_yahoo(t: str) -> pd.DataFrame | None:
        err = None
        for k in range(max_retries):
            try:
                hist = yf.download(
                    tickers=t,
                    start=start,
                    end=end,
                    auto_adjust=True,
                    progress=False,
                    group_by="column",
                    threads=False,           # avoid tripping rate limits
                    session=session,
                )
                if isinstance(hist, pd.DataFrame) and not hist.empty:
                    s = (hist["Adj Close"] if "Adj Close" in hist else hist["Close"]).rename(t).to_frame()
                    if not s.empty:
                        return s
                err = "empty history"
            except Exception as e:
                err = str(e)
            time.sleep(sleep_seconds * (1.5 ** k))
        print(f"Yahoo failed for {t}: {err}")
        return None

    def _dl_stooq(t: str) -> pd.DataFrame | None:
        # Stooq uses US listings without suffix via pandas_datareader
        try:
            s = pdr.DataReader(t, "stooq", start=start, end=end)
            if isinstance(s, pd.DataFrame) and not s.empty:
                # Stooq columns are ['Open','High','Low','Close','Volume']
                s = s["Close"].sort_index().rename(t).to_frame()
                return s
        except Exception as e:
            print(f"Stooq failed for {t}: {e}")
        return None

    def _dl_one(t: str) -> pd.DataFrame | None:
        if source in ("auto", "yahoo"):
            df = _dl_yahoo(t)
            if df is not None:
                return df
            if source == "yahoo":
                return None
        if source in ("auto", "stooq"):
            return _dl_stooq(t)
        return None

    # --- download missing tickers ---
    for i, t in enumerate(need, 1):
        df = _dl_one(t)
        if df is not None and not df.empty:
            df = df.sort_index()
            frames[t] = df
            _write_parquet_or_csv(df, cache / t)
        else:
            print(f"Skipped {t} (no data from Yahoo/Stooq).")

    if not frames:
        raise RuntimeError("No price data downloaded or found in cache.")

    # --- combine & clean ---
    wide = pd.concat(frames.values(), axis=1)

    # Make index datetime and strip tz if present (works on all pandas versions)
    wide.index = pd.to_datetime(wide.index)
    try:
        # If tz-aware, convert to naive; if already naive this raises, so we ignore
        if getattr(wide.index, "tz", None) is not None:
            wide.index = wide.index.tz_convert(None)
    except Exception:
        try:
            wide.index = wide.index.tz_localize(None)
        except Exception:
            pass  # leave as-is if already naive

    wide = wide.sort_index()
    wide = wide.loc[:, [c for c in tickers if c in wide.columns]]

    if monthly:
        wide = wide.resample("M").last()

    wide = wide.dropna(how="all", axis=1).dropna(how="all").ffill(limit=5)

    if wide.shape[1] < min_ok:
        print(f"[download_price_history] Only {wide.shape[1]} tickers succeeded (<{min_ok}). "
              f"Re-run to fill cache or set source='stooq'.")
    return wide