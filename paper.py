"""Paper trading CLI mirroring the investor demo with live data support."""

from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path
from urllib.request import urlopen

import solhunter_zero.investor_demo as investor_demo


def run(argv: list[str] | None = None) -> None:
    """Execute the investor demo using either preset or live price data.

    This lightweight wrapper simply forwards arguments to
    :func:`solhunter_zero.investor_demo.main`.  When ``--url`` is provided the
    JSON response is downloaded and passed to the demo as a temporary data file
    so that the run uses up-to-date market information.  Otherwise the selected
    preset dataset is used.
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

    forwarded = ["--reports", str(args.reports)]

    if args.url:
        with urlopen(args.url, timeout=10) as resp:
            data = resp.read().decode("utf-8")
        with tempfile.NamedTemporaryFile("w", delete=False, suffix=".json") as fh:
            fh.write(data)
            tmp_path = fh.name
        forwarded.extend(["--data", tmp_path])
    elif args.preset:
        forwarded.extend(["--preset", args.preset])

    investor_demo.main(forwarded)


if __name__ == "__main__":  # pragma: no cover
    run(sys.argv[1:])
