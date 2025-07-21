"""Top level package for :mod:`solhunter_zero`.

This module originally imported several submodules on import which pulled in
heavy optional dependencies like ``solders``.  Many unit tests only need to
access lightweight utilities such as :mod:`backtester` and therefore should not
fail simply because those optional packages are missing.  To avoid that we lazily
import the heavy modules when the relevant attributes are accessed.
"""

__version__ = "0.1.0"

__all__ = [
    "load_keypair",
    "fetch_token_prices",
    "fetch_token_prices_async",
    "RiskManager",
    "covariance_matrix",
    "portfolio_cvar",
    "portfolio_variance",
]


def __getattr__(name: str):
    if name == "load_keypair":
        from .wallet import load_keypair as func

        return func
    if name == "fetch_token_prices":
        from .prices import fetch_token_prices as func

        return func
    if name == "fetch_token_prices_async":
        from .prices import fetch_token_prices_async as func

        return func
    if name == "RiskManager":
        from .risk import RiskManager as cls

        return cls
    if name == "covariance_matrix":
        from .risk import covariance_matrix as func

        return func
    if name == "portfolio_cvar":
        from .risk import portfolio_cvar as func

        return func
    if name == "portfolio_variance":
        from .risk import portfolio_variance as func

        return func
    raise AttributeError(name)
