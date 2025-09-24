"""Plotting utilities.

Matplotlib may be verbose but it gives us full control, plus we can sneak in
ridiculous colour palettes if the market ever bores us.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt
import pandas as pd


def plot_cumulative_performance(
    cumulative: pd.DataFrame,
    output_path: Path,
    title: str = "Cumulative Performance",
) -> Path:
    """Plot cumulative performance curves and save the figure.

    Parameters
    ----------
    cumulative:
        DataFrame whose columns are different strategies / benchmarks and whose
        index is the date.
    output_path:
        Location to save the PNG plot. Parent directories are created if needed.
    title:
        Plot title because unnamed charts are shy charts.
    """

    output_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(12, 6))
    for column in cumulative.columns:
        plt.plot(cumulative.index, cumulative[column], label=column)

    plt.title(title)
    plt.ylabel("Growth of $1")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()
    return output_path


__all__ = ["plot_cumulative_performance"]
