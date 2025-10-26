"""Performance statistics with extra sass."""

from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS_PER_YEAR = 252


def compute_cagr(returns: pd.Series) -> float:
    """Compound annual growth rate of a return series."""

    if returns.empty:
        return np.nan

    cumulative = (1 + returns).prod()
    num_years = returns.index.to_series().diff().dt.days.sum() / 365.25
    if num_years == 0:
        return np.nan

    return cumulative ** (1 / num_years) - 1


def compute_volatility(returns: pd.Series) -> float:
    """Annualised volatility of returns."""

    return returns.std() * np.sqrt(TRADING_DAYS_PER_YEAR)


def compute_sharpe_ratio(returns: pd.Series, risk_free: float = 0.0) -> float:
    """Classic Sharpe ratio assuming a flat risk-free rate."""

    excess = returns - risk_free / TRADING_DAYS_PER_YEAR
    volatility = compute_volatility(excess)
    if volatility == 0:
        return np.nan
    return excess.mean() * TRADING_DAYS_PER_YEAR / volatility


def compute_max_drawdown(returns: pd.Series) -> float:
    """Worst peak-to-trough decline of the cumulative performance curve."""

    cumulative = (1 + returns).cumprod()
    running_max = cumulative.cummax()
    drawdowns = cumulative / running_max - 1
    return drawdowns.min()


def compute_yearly_returns(returns: pd.Series) -> pd.Series:
    """Return for each calendar year."""

    return (1 + returns).resample("Y").prod() - 1


def summarize_performance(returns: pd.Series) -> pd.Series:
    """Create a neat bundle of performance stats for reporting."""

    return pd.Series(
        {
            "CAGR": compute_cagr(returns),
            "Volatility": compute_volatility(returns),
            "Sharpe": compute_sharpe_ratio(returns),
            "Max Drawdown": compute_max_drawdown(returns),
        }
    )


__all__ = [
    "compute_cagr",
    "compute_max_drawdown",
    "compute_sharpe_ratio",
    "compute_volatility",
    "compute_yearly_returns",
    "summarize_performance",
]
