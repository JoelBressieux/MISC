"""Strategy definitions.

The idea is to keep each strategy self-contained and delightfully verbose so we
can plug them into the backtest engine without needing to keep a cheat sheet.
"""

from __future__ import annotations

import pandas as pd


class Strategy:
    """Base class for all strategies.

    We keep the interface intentionally tiny: given a set of prices, the
    strategy returns a weight matrix specifying how much capital to allocate to
    each asset at each rebalance date. Child classes can get as creative as they
    like as long as they respect the signature.
    """

    def generate_weights(self, prices: pd.DataFrame) -> pd.DataFrame:
        raise NotImplementedError


class OneYearMomentum(Strategy):
    """Simple 12-month momentum with monthly rebalancing.

    The logic mirrors a classic academic momentum paper: at every month end we
    buy the 50 stocks with the strongest 12-month return (skipping the most
    recent month to avoid short-term reversal noise). Capital is spread evenly
    across the chosen stocks because equal weights make spreadsheets – and
    humans – happy.
    """

    def __init__(self, top_n: int = 50) -> None:
        self.top_n = top_n

    def generate_weights(self, prices: pd.DataFrame) -> pd.DataFrame:  # noqa: D401
        """See base class docstring."""

        # Convert daily prices to month-end prices. We use ``resample`` so that
        # pandas does the heavy lifting and we can sip our coffee instead.
        monthly_prices = prices.resample("M").last()

        # Classic momentum skips the most recent month, so we shift by one month
        # before computing the 12 month return. The +1 keeps the math friendly.
        momentum = monthly_prices.pct_change(periods=12)
        momentum = momentum.shift(1)

        # Select the top ``N`` performers at each month end. ``nlargest`` does
        # the ranking without us having to channel our inner sorting algorithm.
        top_momentum = momentum.apply(lambda row: row.nlargest(self.top_n).index, axis=1)

        # Build a weight DataFrame filled with zeros. We'll sprinkle in equal
        # weights (1/N) for the assets we actually hold each month.
        weights = pd.DataFrame(0.0, index=monthly_prices.index, columns=monthly_prices.columns)

        for date, selected in top_momentum.items():
            if len(selected) == 0:
                continue
            weights.loc[date, selected] = 1.0 / len(selected)

        return weights


__all__ = ["Strategy", "OneYearMomentum"]
