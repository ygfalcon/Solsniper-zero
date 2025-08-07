#!/usr/bin/env python3
"""Run the investor demo across all bundled strategies."""

from solhunter_zero.investor_demo import main


def run(argv: list[str] | None = None) -> None:
    """Execute the demo with the given CLI args.

    Defaults to the ``full`` preset if no arguments are provided so that the
    script remains a quick showcase.  Additional arguments like ``--reports``
    are forwarded to :func:`solhunter_zero.investor_demo.main`.
    """
    if argv is None or len(argv) == 0:
        argv = ["--preset", "full"]
    main(argv)


if __name__ == "__main__":  # pragma: no cover
    import sys

    run(sys.argv[1:])
