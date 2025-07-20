from __future__ import annotations

import os
from pathlib import Path

import tomllib

try:
    import yaml  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    yaml = None


ENV_VARS = {
    "birdeye_api_key": "BIRDEYE_API_KEY",
    "solana_rpc_url": "SOLANA_RPC_URL",
    "dex_base_url": "DEX_BASE_URL",
    "dex_testnet_url": "DEX_TESTNET_URL",
    "metrics_base_url": "METRICS_BASE_URL",
    "discovery_method": "DISCOVERY_METHOD",
    "risk_tolerance": "RISK_TOLERANCE",
    "max_allocation": "MAX_ALLOCATION",
    "max_risk_per_token": "MAX_RISK_PER_TOKEN",
    "trailing_stop": "TRAILING_STOP",
    "max_drawdown": "MAX_DRAWDOWN",
    "volatility_factor": "VOLATILITY_FACTOR",
    "risk_multiplier": "RISK_MULTIPLIER",
}


def _read_config_file(path: Path) -> dict:
    """Return configuration dictionary from a YAML or TOML file."""
    if path.suffix in {".yaml", ".yml"}:
        if yaml is None:
            raise RuntimeError("PyYAML is required for YAML config files")
        with path.open("r", encoding="utf-8") as fh:
            return yaml.safe_load(fh) or {}
    if path.suffix == ".toml":
        with path.open("rb") as fh:
            return tomllib.load(fh)
    raise ValueError(f"Unsupported config format: {path}")


def load_config(path: str | os.PathLike | None = None) -> dict:
    """Load configuration from ``path`` or default locations."""
    if path is None:
        env_path = os.getenv("SOLHUNTER_CONFIG")
        if env_path:
            path = env_path
        else:
            for name in ("config.yaml", "config.yml", "config.toml"):
                if Path(name).is_file():
                    path = name
                    break
    if path is None:
        return {}
    return _read_config_file(Path(path))


def apply_env_overrides(config: dict) -> dict:
    """Merge environment variable overrides into ``config``."""
    cfg = dict(config)
    for key, env in ENV_VARS.items():
        env_val = os.getenv(env)
        if env_val is not None:
            cfg[key] = env_val
    return cfg


def set_env_from_config(config: dict) -> None:
    """Set environment variables for values present in ``config``."""
    for key, env in ENV_VARS.items():
        val = config.get(key)
        if val is not None and os.getenv(env) is None:
            os.environ[env] = str(val)
