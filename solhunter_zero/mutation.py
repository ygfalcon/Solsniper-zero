from __future__ import annotations

import copy
import json
import os
import random
from typing import Iterable, Dict

from .agents import BaseAgent


# ----------------------------------------------------------------------
#  Agent cloning helpers
# ----------------------------------------------------------------------

def clone_agent(agent: BaseAgent, name: str | None = None, **params) -> BaseAgent:
    """Return a deep copy of ``agent`` with optional parameter overrides."""
    cloned = copy.deepcopy(agent)
    if name:
        cloned.name = name
    for attr, val in params.items():
        if hasattr(cloned, attr):
            setattr(cloned, attr, val)
    return cloned


def mutate_agent(
    agent: BaseAgent,
    *,
    name: str | None = None,
    volatility_weight: float | None = None,
    time_horizon: int | None = None,
) -> BaseAgent:
    """Clone ``agent`` with tweaked parameters and a new ``name``."""
    params: Dict[str, object] = {}
    if volatility_weight is not None:
        for attr in ("volatility_factor", "volatility_weight"):
            if hasattr(agent, attr):
                params[attr] = volatility_weight
    if time_horizon is not None:
        for attr in ("time_horizon", "days", "count"):
            if hasattr(agent, attr):
                params[attr] = time_horizon
    if name is None:
        suffix = random.randint(1000, 9999)
        name = f"{agent.name}_{suffix}"
    return clone_agent(agent, name=name, **params)


# ----------------------------------------------------------------------
#  Mutation state persistence
# ----------------------------------------------------------------------

def load_state(path: str) -> dict:
    state = {"active": [], "roi": {}}
    try:
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        if isinstance(data, dict):
            state.update(data)
    except Exception:
        return state
    return state


def save_state(state: dict, path: str) -> None:
    """Persist mutation ``state`` to ``path``.

    Each entry in ``state['active']`` can be either a mutation record or a
    simple agent name. When saving, mutation records are normalised to include
    the mutated agent name, the base agent used for cloning and the mutation
    parameters.  If ``state['active']`` only contains names, they are written
    back as-is for backward compatibility.
    """

    active = []
    for entry in state.get("active", []):
        if isinstance(entry, dict):
            active.append(
                {
                    "name": entry.get("name"),
                    "base": entry.get("base"),
                    "params": entry.get("params", {}),
                }
            )
        else:
            active.append(entry)

    data = dict(state)
    data["active"] = active

    tmp = os.path.join(os.path.dirname(path), ".tmp_state")
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(data, fh)
    os.replace(tmp, path)
