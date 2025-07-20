from __future__ import annotations
from typing import List
import statistics

from .simulation import SimulationResult


def should_buy(
    sim_results: List[SimulationResult],
    *,
    min_success: float = 0.6,
    min_roi: float = 1.0,
    min_sharpe: float = 1.0,
) -> bool:
    """Decide whether to buy a token based on simulation results.

    The decision now incorporates the Sharpe ratio to account for volatility.
    Thresholds for average success probability, ROI and Sharpe ratio are
    configurable.
    """

    if not sim_results:
        return False

    successes = [r.success_prob for r in sim_results]
    rois = [r.expected_roi for r in sim_results]

    avg_success = sum(successes) / len(successes)
    avg_roi = sum(rois) / len(rois)
    roi_std = statistics.stdev(rois) if len(rois) > 1 else 0.0
    sharpe = avg_roi / roi_std if roi_std > 0 else 0.0

    return (
        avg_success >= min_success and avg_roi >= min_roi and sharpe >= min_sharpe
    )


def should_sell(
    sim_results: List[SimulationResult],
    *,
    max_success: float = 0.4,
    max_roi: float = 0.0,
) -> bool:
    """Decide whether to sell a token based on simulation results.

    The function looks at the average expected ROI and the average success
    probability.  If either indicates poor future performance we recommend
    selling.  By default a negative expected return or a success probability
    below ``max_success`` triggers a sell.
    """

    if not sim_results:
        return False

    successes = [r.success_prob for r in sim_results]
    rois = [r.expected_roi for r in sim_results]

    avg_success = sum(successes) / len(successes)
    avg_roi = sum(rois) / len(rois)

    return avg_success <= max_success or avg_roi <= max_roi
