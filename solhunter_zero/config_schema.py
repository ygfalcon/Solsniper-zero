from __future__ import annotations

from typing import Dict, List

from pydantic import (
    AnyUrl,
    BaseModel,
    ValidationError,
    root_validator,
    field_validator,
    ValidationInfo,
)


class ConfigModel(BaseModel):
    """Schema for SolHunter configuration files."""

    solana_rpc_url: AnyUrl
    dex_base_url: AnyUrl
    agents: List[str]
    agent_weights: Dict[str, float]

    # Risk management and trading parameters
    stop_loss: float | None = None
    take_profit: float | None = None
    risk_tolerance: float | None = None
    max_allocation: float | None = None
    max_risk_per_token: float | None = None
    trailing_stop: float | None = None
    max_drawdown: float | None = None
    volatility_factor: float | None = None

    class Config:
        extra = "allow"

    @field_validator("agents")
    def _agents_non_empty(cls, value: List[str]) -> List[str]:
        if not value or not all(isinstance(a, str) and a.strip() for a in value):
            raise ValueError("agents must be a list of non-empty strings")
        return value

    @root_validator(skip_on_failure=True)
    def _weights_for_agents(cls, values: Dict[str, object]) -> Dict[str, object]:
        agents = values.get("agents") or []
        weights = values.get("agent_weights") or {}
        missing = [a for a in agents if a not in weights]
        if missing:
            raise ValueError(f"missing weight for agent(s): {', '.join(missing)}")
        return values

    @field_validator(
        "stop_loss",
        "take_profit",
        "risk_tolerance",
        "max_allocation",
        "max_risk_per_token",
        "trailing_stop",
        "max_drawdown",
        "volatility_factor",
    )
    def _fraction_0_1(cls, value: float | None, info: ValidationInfo):
        if value is None:
            return value
        if not 0 <= value <= 1:
            raise ValueError(f"{info.field_name} must be between 0 and 1")
        return value


def validate_config(data: Dict[str, object]) -> Dict[str, object]:
    """Validate ``data`` against :class:`ConfigModel`.

    Returns the validated data with type normalization applied.
    Raises ``ValueError`` on validation errors.
    """
    try:
        return ConfigModel(**data).model_dump(mode="json")
    except ValidationError as exc:  # pragma: no cover - pass through as ValueError
        raise ValueError(str(exc)) from exc
