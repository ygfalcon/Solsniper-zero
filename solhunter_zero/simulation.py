"""Monte Carlo simulation utilities."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import List

import numpy as np


@dataclass
class SimulationResult:
    """Result of a single simulation run."""

    success_prob: float
    expected_roi: float


def _simulate_single(initial_price: float) -> SimulationResult:
    """Naïve Monte Carlo price simulation."""

    # price path of 24 steps representing roughly 1h of minute candles
    volatility = np.random.uniform(0.01, 0.2)
    drift = np.random.uniform(-0.05, 0.1)
    price = initial_price
    for _ in range(24):
        shock = np.random.normal(drift, volatility)
        price *= math.exp(shock)

    roi = (price - initial_price) / max(initial_price, 1e-6)
    success = 1.0 if roi > 0 else 0.0
    return SimulationResult(success_prob=success, expected_roi=roi)


def simulation_count(previous_runs: int, base: int = 500) -> int:
    """Return a dynamic number of simulations based on prior history."""

    multiplier = 1.0 + min(previous_runs / 100.0, 1.0)
    jitter = np.random.uniform(0.8, 1.2)
    return int(base * multiplier * jitter)


def run_simulations(token: str, count: int = 500, start_price: float | None = None) -> List[SimulationResult]:
    """Run a batch of simulations for ``token``.

    Parameters
    ----------
    token:
        Token address being evaluated.
    count:
        Number of Monte Carlo runs to perform.
    start_price:
        Starting price for the simulations. If ``None`` a small random
        starting value is used.
    """

    initial = start_price or np.random.uniform(0.1, 1.0)
    results = [_simulate_single(initial) for _ in range(count)]
    return results
