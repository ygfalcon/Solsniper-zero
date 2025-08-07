from __future__ import annotations

from typing import Dict, List, Optional

from pydantic import (
    BaseModel,
    AnyUrl,
    ValidationError,
    root_validator,
    validator,
    StrictStr,
)


class ConfigModel(BaseModel):
    """Schema for SolHunter configuration files."""

    solana_rpc_url: AnyUrl
    dex_base_url: AnyUrl
    birdeye_api_key: Optional[StrictStr] = None
    use_flash_loans: Optional[bool] = None
    max_hops: Optional[int] = None
    priority_fees: Optional[List[float]] = None
    agents: List[str]
    agent_weights: Dict[str, float]

    class Config:
        extra = "allow"

    @validator("agents")
    def _agents_non_empty(cls, value: List[str]) -> List[str]:
        if not value or not all(isinstance(a, str) and a.strip() for a in value):
            raise ValueError("agents must be a list of non-empty strings")
        return value

    @validator("max_hops")
    def _max_hops_positive(cls, value: Optional[int]) -> Optional[int]:
        if value is not None and int(value) < 2:
            raise ValueError("max_hops must be >= 2")
        return value

    @validator("priority_fees", pre=True)
    def _parse_priority_fees(cls, value: object) -> Optional[List[float]]:
        if value in (None, "", []):
            return None
        if isinstance(value, str):
            parts = [p.strip() for p in value.split(",") if p.strip()]
        elif isinstance(value, (list, tuple)):
            parts = value
        else:
            raise ValueError(
                "priority_fees must be a list of floats or comma-separated string"
            )
        try:
            return [float(p) for p in parts]
        except (TypeError, ValueError) as exc:
            raise ValueError("priority_fees must contain valid floats") from exc

    @root_validator
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
