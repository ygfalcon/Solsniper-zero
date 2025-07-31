from __future__ import annotations
from typing import Sequence

import os
import torch

from .models.regime_model import RegimeModel, load_regime_model as _load_model

_REGIME_MODEL: RegimeModel | None = None


def load_regime_model(path: str | None = None) -> None:
    """Load regime model from ``path`` or ``REGIME_MODEL_PATH`` env var."""
    global _REGIME_MODEL
    if path is None:
        path = os.getenv("REGIME_MODEL_PATH")
    if path and os.path.exists(path):
        try:
            _REGIME_MODEL = _load_model(path)
        except Exception:
            _REGIME_MODEL = None
    else:
        _REGIME_MODEL = None


def detect_regime_ml(prices: Sequence[float], features: Sequence[Sequence[float]]) -> str:
    """Detect regime using the loaded ML model."""
    if _REGIME_MODEL is None:
        return detect_regime_rule(prices)
    seq_len = _REGIME_MODEL.seq_len
    if len(prices) < seq_len or len(features) < seq_len:
        return detect_regime_rule(prices)
    seq = [ [p] + list(f) for p, f in zip(prices[-seq_len:], features[-seq_len:]) ]
    return _REGIME_MODEL.predict(seq)


def detect_regime_rule(prices: Sequence[float], *, threshold: float = 0.02) -> str:
    """Return market regime label based on price trend.

    Parameters
    ----------
    prices:
        Historical price sequence ordered oldest to newest.
    threshold:
        Minimum fractional change over the window considered a trend.
    """
    if len(prices) < 2:
        return "sideways"
    change = prices[-1] / prices[0] - 1
    if change > threshold:
        return "bull"
    if change < -threshold:
        return "bear"
    return "sideways"


def detect_regime(prices: Sequence[float], *, threshold: float = 0.02, features: Sequence[Sequence[float]] | None = None) -> str:
    """Detect regime using ML model when available."""
    if _REGIME_MODEL is not None and features is not None:
        return detect_regime_ml(prices, features)
    return detect_regime_rule(prices, threshold=threshold)
