"""Fetch live tick data from a public market API."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List

import requests


def fetch_live_ticks(limit: int = 100, symbol: str = "BTCUSDT") -> List[Dict[str, Any]]:
    """Return recent price ticks for ``symbol``.

    The function queries Binance's public API for recent 1 minute candlesticks
    and converts them into the same structure used by
    :func:`solhunter_zero.datasets.sample_ticks.load_sample_ticks`.

    Each tick contains:

    ``timestamp``: ISO formatted UTC time of the candle open.
    ``price``: closing price for the interval.
    ``depth``: traded volume during the interval (used as a depth proxy).
    ``imbalance``: (close - open) / open as a simple imbalance metric.

    On any network or parsing error an empty list is returned so callers can
    fall back to bundled sample data.
    """

    url = "https://api.binance.com/api/v3/klines"
    params = {"symbol": symbol, "interval": "1m", "limit": limit}
    try:
        resp = requests.get(url, params=params, timeout=10)
        resp.raise_for_status()
        klines = resp.json()
    except Exception:
        return []

    ticks: List[Dict[str, Any]] = []
    for kline in klines:
        try:
            (
                open_time,
                open_price,
                _high,
                _low,
                close_price,
                volume,
                _close_time,
                _quote_asset_vol,
                _num_trades,
                _taker_base_vol,
                _taker_quote_vol,
                _ignore,
            ) = kline

            price = float(close_price)
            depth = float(volume)
            open_f = float(open_price)
            imbalance = (price - open_f) / open_f if open_f else 0.0
            timestamp = datetime.utcfromtimestamp(open_time / 1000).isoformat()

            ticks.append(
                {
                    "timestamp": timestamp,
                    "price": price,
                    "depth": depth,
                    "imbalance": imbalance,
                }
            )
        except Exception:
            # Skip malformed entries but continue processing others
            continue

    return ticks

