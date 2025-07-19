from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List

import numpy as np
import requests

logger = logging.getLogger(__name__)

@dataclass
class SimulationResult:
    success_prob: float
    expected_roi: float


def fetch_token_metrics(token: str) -> dict:
    """Fetch historical return metrics for ``token``.

    The function tries to retrieve data from a remote API. When the call fails
    for any reason, default values are returned.  The metrics include the mean
    daily return and the standard deviation of daily returns.
    """

    url = f"https://api.example.com/token/{token}/metrics"
    try:
        resp = requests.get(url, timeout=5)
        resp.raise_for_status()
        data = resp.json()
        return {
            "mean": float(data.get("mean_return", 0.0)),
            "volatility": float(data.get("volatility", 0.02)),
        }
    except Exception as exc:  # pragma: no cover - network errors
        logger.warning("Failed to fetch metrics for %s: %s", token, exc)
        return {"mean": 0.0, "volatility": 0.02}


def run_simulations(token: str, count: int = 1000, days: int = 30) -> List[SimulationResult]:
    """Run Monte Carlo simulations for a given token."""

    metrics = fetch_token_metrics(token)
    mu = metrics["mean"]
    sigma = metrics["volatility"]

    results: List[SimulationResult] = []
    for _ in range(count):
        daily_returns = np.random.normal(mu, sigma, days)
        roi = float(np.prod(1 + daily_returns) - 1)
        success_prob = float(np.mean(daily_returns > 0))
        results.append(SimulationResult(success_prob, roi))

    return results
