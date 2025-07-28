"""Typed event payload schemas used with the event bus."""
from __future__ import annotations

from dataclasses import dataclass, asdict, is_dataclass
from typing import Any, Dict, Type


@dataclass
class ActionExecuted:
    """Payload for an executed trading action."""
    action: Dict[str, Any]
    result: Any


@dataclass
class WeightsUpdated:
    """Payload for updated agent weights."""
    weights: Dict[str, float]


@dataclass
class RLWeights:
    """Payload for RL-generated weights and risk parameters."""

    weights: Dict[str, float]
    risk: Dict[str, float] | None = None


@dataclass
class RLCheckpoint:
    """Payload emitted when RL daemon saves a checkpoint."""
    time: float
    path: str


@dataclass
class PortfolioUpdated:
    """Payload emitted whenever portfolio balances change."""

    balances: Dict[str, float]


_EVENT_SCHEMAS: Dict[str, Type] = {
    "action_executed": ActionExecuted,
    "weights_updated": WeightsUpdated,
    "rl_weights": RLWeights,
    "rl_checkpoint": RLCheckpoint,
    "portfolio_updated": PortfolioUpdated,
}


def validate_message(topic: str, payload: Any) -> Any:
    """Validate ``payload`` for ``topic`` returning dataclass instance.

    Unknown topics pass through unmodified.
    Raises ``ValueError`` if validation fails.
    """
    schema = _EVENT_SCHEMAS.get(topic)
    if schema is None:
        return payload
    if isinstance(payload, schema):
        return payload
    if isinstance(payload, dict):
        try:
            return schema(**payload)  # type: ignore[arg-type]
        except Exception as exc:
            raise ValueError(f"Invalid payload for {topic}: {exc}") from exc
    raise ValueError(f"Invalid payload for {topic}")


def to_dict(payload: Any) -> Any:
    """Convert dataclass ``payload`` to a plain dictionary."""
    if is_dataclass(payload):
        return asdict(payload)
    return payload

