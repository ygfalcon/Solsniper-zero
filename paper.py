"""Minimal paper trading CLI using :mod:`solhunter_zero` strategies.

This script delegates to :func:`solhunter_zero.simple_bot.run`, the same
helper used by ``demo.py``.  It accepts either a local tick dataset or, when
``--fetch-live`` is supplied, attempts to download recent market data via a
Codex endpoint.  If the live fetch fails the bundled sample ticks are used
instead.  The underlying :mod:`solhunter_zero.investor_demo` engine writes
``summary.json``, ``trade_history.json`` and ``highlights.json`` reports so
that downstream tests can compare results across the demo and paper workflows.
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List
from urllib.request import urlopen

from solhunter_zero.cli_shared import add_shared_arguments
from solhunter_zero.datasets.sample_ticks import load_sample_ticks
from solhunter_zero.simple_bot import run as run_simple_bot


# Public Codex endpoint providing recent SOL/USD candles.  The exact source is
# not critical; the fetch is best-effort and falls back to bundled samples when
# unavailable.
CODEX_URL = (
    "https://api.coingecko.com/api/v3/coins/solana/market_chart"
    "?vs_currency=usd&days=1&interval=hourly"
)


def _ticks_to_price_file(ticks: List[Dict[str, Any]]) -> Path:
    """Convert tick entries to a temporary JSON price dataset."""

    entries: List[Dict[str, Any]] = []
    for i, tick in enumerate(ticks):
        if "price" not in tick:
            continue
        try:
            price = float(tick["price"])
        except Exception:
            continue
        date = str(tick.get("timestamp", i))
        entries.append({"date": date, "price": price})
    if not entries:
        raise ValueError("tick dataset is empty")
    tmp = tempfile.NamedTemporaryFile("w", delete=False, suffix=".json")
    json.dump(entries, tmp)
    tmp.close()
    return Path(tmp.name)


def _fetch_live_dataset() -> Path | None:
    """Fetch recent market data via Codex.

    Returns a path to a temporary JSON file or ``None`` if fetching fails.
    """

    try:
        with urlopen(CODEX_URL, timeout=10) as resp:
            data = json.load(resp)
        prices = data.get("prices") or []
        ticks = [
            {"price": p[1], "timestamp": int(p[0] // 1000)} for p in prices
        ]
        if not ticks:
            return None
        return _ticks_to_price_file(ticks)
    except Exception:
        return None


def run(argv: List[str] | None = None) -> None:
    """Execute a lightweight paper trading simulation."""

    parser = argparse.ArgumentParser(description="Run simple paper trading")
    add_shared_arguments(parser)
    parser.add_argument(
        "--ticks",
        type=Path,
        default=None,
        help="Path to JSON tick history (defaults to bundled sample)",
    )
    parser.add_argument(
        "--fetch-live",
        action="store_true",
        help="Fetch live market data via Codex, falling back to sample ticks",
    )
    args = parser.parse_args(argv)

    dataset: Path | None = None
    if args.fetch_live:
        dataset = _fetch_live_dataset()

    if dataset is None:
        ticks = load_sample_ticks(args.ticks) if args.ticks else load_sample_ticks()
        dataset = _ticks_to_price_file(ticks)

    run_simple_bot(
        dataset,
        args.reports,
        capital=args.capital,
        fee=args.fee,
        slippage=args.slippage,
    )


if __name__ == "__main__":  # pragma: no cover
    run(sys.argv[1:])


__all__ = ["run"]

