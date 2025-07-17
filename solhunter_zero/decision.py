"""Decision making utilities."""

from __future__ import annotations

from typing import List, Optional

import numpy as np

from .simulation import SimulationResult
from .analysis import sentiment_score
from .memory import Memory


def should_buy(sim_results: List[SimulationResult], token: str | None = None, memory: Optional[Memory] = None) -> bool:
    """Return ``True`` if simulation results and sentiment suggest a long position."""

    if not sim_results:
        return False

    successes = np.array([r.success_prob for r in sim_results])
    rois = np.array([r.expected_roi for r in sim_results])

    avg_success = float(successes.mean())
    avg_roi = float(rois.mean())
    std_roi = float(rois.std())

    # Dynamic threshold based on ROI volatility. The higher the dispersion,
    # the more conservative we become.
    required_roi = max(0.5, 1.5 * std_roi)
    if memory:
        history_roi = memory.average_roi(token)
        required_roi -= history_roi * 0.1

    sentiment = sentiment_score(token) if token else 0.0

    return avg_success > 0.55 and avg_roi > required_roi and sentiment >= -0.2
