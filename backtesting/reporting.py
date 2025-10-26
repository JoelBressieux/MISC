"""Turn raw backtest data into human-friendly tables."""

from __future__ import annotations

import pandas as pd
import numpy as np
from typing import Dict

from .metrics import compute_yearly_returns, summarize_performance


def build_summary_table(results: pd.Series) -> pd.DataFrame:
    """Convert summary statistics into a DataFrame for pretty printing."""

    return pd.DataFrame(results).T


def compile_performance_tables(
    strategy_returns: pd.Series,
    benchmark_returns: pd.Series,
    rf_annual: float = 0.0,   # annual risk-free (e.g. 0.02 for 2%)
) -> Dict[str, pd.DataFrame]:
    """
    Build summary and yearly performance tables from return series.

    Parameters
    ----------
    strategy_returns : pd.Series
        Periodic returns of the strategy (same frequency as benchmark).
    benchmark_returns : pd.Series
        Periodic returns of the benchmark.
    rf_annual : float
        Annualized risk-free rate used for Sharpe.

    Returns
    -------
    dict with:
      - 'summary' (raw numeric)
      - 'yearly' (raw numeric)
      - 'summary_fmt' (percents, 2 sig figs)
      - 'yearly_fmt'  (percents, 2 sig figs)
    """

    # ---------- helpers (kept local so function is self-contained) ----------
    def _align(a: pd.Series, b: pd.Series) -> tuple[pd.Series, pd.Series]:
        a = pd.Series(a).astype(float).dropna()
        b = pd.Series(b).astype(float).dropna()
        joined = a.to_frame("s").join(b.to_frame("b"), how="inner")
        return joined["s"], joined["b"]

    def _periods_per_year(idx: pd.Index) -> int:
        if not isinstance(idx, pd.DatetimeIndex):
            return 12  # assume monthly if not date-indexed
        freq = pd.infer_freq(idx)
        if freq:
            f = freq.upper()
            if f.startswith("B") or f == "D":
                return 252
            if f.startswith("W"):
                return 52
            if f.endswith("M"):
                return 12
            if f.startswith("Q"):
                return 4
        # fallback: median spacing
        if len(idx) > 1:
            days = np.median(np.diff(idx.values).astype("timedelta64[D]").astype(int))
            if days >= 25: return 12
            if days >= 5:  return 52
        return 252

    def _cagr(r: pd.Series, ppy: int) -> float:
        if r.empty: return np.nan
        total = float((1.0 + r).prod())
        years = len(r) / ppy
        if years <= 0 or total <= 0: return np.nan
        return total**(1.0 / years) - 1.0

    def _ann_vol(r: pd.Series, ppy: int) -> float:
        if r.std(ddof=0) == 0: return 0.0
        return float(r.std(ddof=0) * np.sqrt(ppy))

    def _sharpe(r: pd.Series, ppy: int, rf_annual: float) -> float:
        if r.std(ddof=0) == 0: return np.nan
        mu = r.mean()
        rf_p = rf_annual / ppy
        return float((mu - rf_p) / r.std(ddof=0) * np.sqrt(ppy))

    def _max_drawdown(r: pd.Series) -> float:
        if r.empty: return np.nan
        curve = (1.0 + r).cumprod()
        peak = curve.cummax()
        dd = curve / peak - 1.0
        return float(dd.min())

    def _resample_yearly(r: pd.Series) -> pd.Series:
        if isinstance(r.index, pd.DatetimeIndex):
            y = (1.0 + r).resample("Y").prod() - 1.0
            y.index = y.index.year
            return y
        # non-datetime: group by synthetic years every 12 periods
        years = np.arange(len(r)) // 12
        return pd.Series((1.0 + r).groupby(years).prod().values - 1.0, index=np.unique(years))

    def _fmt_pct_df(df: pd.DataFrame, cols=None) -> pd.DataFrame:
        out = df.copy()
        if cols is None:
            cols = out.select_dtypes(include=[np.number]).columns
        for c in cols:
            out[c] = (out[c] * 100.0).map(lambda v: f"{v:.2g}%"
                                          if pd.notna(v) else "")
        return out
    # -----------------------------------------------------------------------

    s, b = _align(strategy_returns, benchmark_returns)
    ppy = _periods_per_year(s.index)

    # summary stats (raw)
    summary = pd.DataFrame(
        {
            "CAGR":        [_cagr(s, ppy), _cagr(b, ppy)],
            "Volatility":  [_ann_vol(s, ppy), _ann_vol(b, ppy)],
            "Max Drawdown":[_max_drawdown(s), _max_drawdown(b)],
            "Sharpe":      [_sharpe(s, ppy, rf_annual), _sharpe(b, ppy, rf_annual)],
        },
        index=["Strategy", "Benchmark"],
    )

    # yearly (raw)
    yearly = pd.DataFrame({
        "Strategy": _resample_yearly(s),
        "Benchmark": _resample_yearly(b),
    })
    yearly.index.name = "Year"

    # formatted tables (percents with 2 significant digits)
    pct_cols_summary = [c for c in ["CAGR", "Volatility", "Max Drawdown"] if c in summary.columns]
    summary_fmt = _fmt_pct_df(summary, pct_cols_summary)

    yearly_fmt = _fmt_pct_df(yearly, cols=["Strategy", "Benchmark"])

    return {
        "summary": summary,           # raw numeric
        "yearly": yearly,             # raw numeric
        "summary_fmt": summary_fmt,   # pretty %
        "yearly_fmt": yearly_fmt,     # pretty %
    }