"""Paper trading CLI delegating to :mod:`solhunter_zero.simple_bot`."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from solhunter_zero.simple_bot import run as run_simple_bot
import solhunter_zero.investor_demo as investor_demo


def run(argv: list[str] | None = None) -> None:
    """Execute the investor demo using either preset or live price data.

    ``paper.py`` previously reimplemented a small wrapper around
    :func:`solhunter_zero.investor_demo.main`.  The logic now lives in
    :mod:`solhunter_zero.simple_bot` so that both the standalone demo and this
    CLI share a common implementation.  The function simply parses the dataset
    options and forwards them to :func:`simple_bot.run`.
    """

    parser = argparse.ArgumentParser(description="Run the investor demo with optional live prices")
    parser.add_argument(
        "--reports",
        type=Path,
        default=Path("reports"),
        help="Directory where demo reports will be written",
    )
    parser.add_argument(
        "--preset",
        choices=sorted(investor_demo.PRESET_DATA_FILES.keys()),
        default=None,
        help="Bundled price dataset to use (defaults to demo's built-in set)",
    )
    parser.add_argument(
        "--url",
        type=str,
        default=None,
        help="HTTP URL returning JSON price data; overrides --preset",
    )
    args = parser.parse_args(argv)
    args.reports.mkdir(parents=True, exist_ok=True)

    if args.url and args.preset:
        raise ValueError("Provide only one of --url or --preset")

    dataset = args.url or args.preset
    run_simple_bot(dataset, args.reports)


if __name__ == "__main__":  # pragma: no cover
    run(sys.argv[1:])
