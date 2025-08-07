#!/usr/bin/env python3
"""Run the investor demo across all bundled strategies."""

from solhunter_zero.investor_demo import main


def run() -> None:
    """Execute the demo with the full preset and default report path."""
    main(["--preset", "full"])


if __name__ == "__main__":  # pragma: no cover
    run()
