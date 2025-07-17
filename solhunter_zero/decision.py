from __future__ import annotations
from typing import List
from .simulation import SimulationResult


def should_buy(sim_results: List[SimulationResult]) -> bool:
    if not sim_results:
        return False
    avg_success = sum(r.success_prob for r in sim_results) / len(sim_results)
    avg_roi = sum(r.expected_roi for r in sim_results) / len(sim_results)
    return avg_success > 0.6 and avg_roi > 1
