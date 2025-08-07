from __future__ import annotations

from typing import Dict, List

from pydantic import (
    BaseModel,
    AnyUrl,
    ValidationError,
    root_validator,
    validator,
    Field,
)


class ConfigModel(BaseModel):
    """Schema for SolHunter configuration files."""

    solana_rpc_url: AnyUrl
    dex_base_url: AnyUrl
    agents: List[str]
    agent_weights: Dict[str, float]
    initial_tokens: List[str] = Field(default_factory=list)

    class Config:
        extra = "allow"

    @validator("agents")
    def _agents_non_empty(cls, value: List[str]) -> List[str]:
        if not value or not all(isinstance(a, str) and a.strip() for a in value):
            raise ValueError("agents must be a list of non-empty strings")
        return value

    @validator("initial_tokens", each_item=True)
    def _initial_tokens_non_empty(cls, value: str) -> str:
        if not isinstance(value, str) or not value.strip():
            raise ValueError("initial_tokens must be a list of non-empty strings")
        return value

    @root_validator(skip_on_failure=True)
    def _weights_for_agents(cls, values: Dict[str, object]) -> Dict[str, object]:
        agents = values.get("agents") or []
        weights = values.get("agent_weights") or {}
        missing = [a for a in agents if a not in weights]
        if missing:
            raise ValueError(f"missing weight for agent(s): {', '.join(missing)}")
        return values


def validate_config(data: Dict[str, object]) -> Dict[str, object]:
    """Validate ``data`` against :class:`ConfigModel`.

    Returns the validated data with type normalization applied.
    Raises ``ValueError`` on validation errors.
    """
    try:
        return ConfigModel(**data).dict()
    except ValidationError as exc:  # pragma: no cover - pass through as ValueError
        raise ValueError(str(exc)) from exc
