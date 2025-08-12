from __future__ import annotations

"""Loader for small slice of live market data used in CI tests.

The data is fetched from a public GitHub-hosted CSV containing historical
Apple stock prices.  The first ``limit`` closing prices are returned as a
list of tick dictionaries with ``timestamp`` and ``price`` keys.  The
dataset is static and therefore deterministic across runs.
"""

from typing import Any, Dict, List
from urllib.error import URLError
from urllib.request import urlopen
import csv

URL = (
    "https://raw.githubusercontent.com/plotly/datasets/master/finance-charts-apple.csv"
)

_cache: List[Dict[str, Any]] | None = None


def load_live_ticks(limit: int = 120) -> List[Dict[str, Any]]:
    """Fetch a deterministic slice of real market data.

    Returns an empty list when the data cannot be retrieved (e.g. due to
    lack of network access).
    """

    global _cache
    if _cache is not None and len(_cache) >= limit:
        return _cache[:limit]

    try:
        with urlopen(URL, timeout=10) as resp:
            text = resp.read().decode("utf-8").splitlines()
    except (URLError, OSError):
        _cache = []
        return _cache

    reader = csv.DictReader(text)
    ticks: List[Dict[str, Any]] = []
    for row in reader:
        try:
            price = float(row["AAPL.Close"])
        except (KeyError, TypeError, ValueError):
            continue
        ticks.append({"timestamp": row.get("Date", ""), "price": price})
        if len(ticks) >= limit:
            break

    _cache = ticks
    return ticks
