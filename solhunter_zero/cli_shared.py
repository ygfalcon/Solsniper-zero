"""Utilities for shared CLI arguments.

This helper centralizes commonly used command line flags so that scripts such
as :mod:`demo` and :mod:`paper` remain in sync.  Each consumer can extend its
own :class:`argparse.ArgumentParser` and invoke :func:`add_shared_arguments` to
populate options that ultimately feed into
``solhunter_zero.investor_demo``:

``--reports``
    Directory where output artifacts are written.
``--capital``
    Starting USD balance for the backtest.
``--fee``
    Per-trade fee expressed as a fractional cost.
``--slippage``
    Per-trade slippage expressed as a fractional cost.
"""

from __future__ import annotations

from argparse import ArgumentParser
from pathlib import Path


def add_shared_arguments(parser: ArgumentParser) -> ArgumentParser:
    """Attach common CLI options to *parser*.

    Parameters
    ----------
    parser:
        The :class:`argparse.ArgumentParser` instance to extend.

    Returns
    -------
    ArgumentParser
        The parser for convenience so callers can chain or simply ignore the
        return value.
    """

    parser.add_argument(
        "--reports",
        type=Path,
        default=Path("reports"),
        help="Directory where reports will be written",
    )
    parser.add_argument(
        "--capital",
        type=float,
        default=100.0,
        help="Starting capital for the backtest",
    )
    parser.add_argument(
        "--fee",
        type=float,
        default=0.0,
        help="Per-trade fee as a fractional cost",
    )
    parser.add_argument(
        "--slippage",
        type=float,
        default=0.0,
        help="Per-trade slippage as a fractional cost",
    )
    return parser


__all__ = ["add_shared_arguments"]

