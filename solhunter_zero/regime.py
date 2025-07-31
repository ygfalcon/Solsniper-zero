from __future__ import annotations
from typing import Sequence

from .regime_cluster import cluster_regime


def detect_regime(
    prices: Sequence[float], *, threshold: float = 0.02, window: int = 20, method: str = "kmeans"
) -> str:
    """Return market regime label based on price trend or clustering.

    Parameters
    ----------
    prices:
        Historical price sequence ordered oldest to newest.
    threshold:
        Minimum fractional change over the window considered a trend when not
        enough history is available for clustering.
    window:
        Rolling window size used by the clustering model.
    method:
        Clustering algorithm to apply (``"kmeans"`` or ``"dbscan"``).
    """
    label = cluster_regime(prices, window=window, method=method)
    if label:
        return label
    if len(prices) < 2:
        return "sideways"
    change = prices[-1] / prices[0] - 1
    if change > threshold:
        return "bull"
    if change < -threshold:
        return "bear"
    return "sideways"
