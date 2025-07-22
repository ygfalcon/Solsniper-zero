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
    "orca_api_url": "ORCA_API_URL",
    "raydium_api_url": "RAYDIUM_API_URL",
    "orca_ws_url": "ORCA_WS_URL",
    "raydium_ws_url": "RAYDIUM_WS_URL",
    "jupiter_ws_url": "JUPITER_WS_URL",
    "orca_dex_url": "ORCA_DEX_URL",
    "raydium_dex_url": "RAYDIUM_DEX_URL",
    "metrics_base_url": "METRICS_BASE_URL",
    "discovery_method": "DISCOVERY_METHOD",
    "risk_tolerance": "RISK_TOLERANCE",
    "max_allocation": "MAX_ALLOCATION",
    "max_risk_per_token": "MAX_RISK_PER_TOKEN",
    "risk_multiplier": "RISK_MULTIPLIER",
    "trailing_stop": "TRAILING_STOP",
    "max_drawdown": "MAX_DRAWDOWN",
    "volatility_factor": "VOLATILITY_FACTOR",
    "arbitrage_threshold": "ARBITRAGE_THRESHOLD",
    "arbitrage_amount": "ARBITRAGE_AMOUNT",

    "min_portfolio_value": "MIN_PORTFOLIO_VALUE",
    "min_delay": "MIN_DELAY",
    "max_delay": "MAX_DELAY",

    "strategies": "STRATEGIES",

    "token_suffix": "TOKEN_SUFFIX",
    "token_keywords": "TOKEN_KEYWORDS",
    "volume_threshold": "VOLUME_THRESHOLD",

    "agents": "AGENTS",
    "agent_weights": "AGENT_WEIGHTS",
    "weights_path": "WEIGHTS_PATH",
    "dex_priorities": "DEX_PRIORITIES",
    "dex_fees": "DEX_FEES",
    "dex_gas": "DEX_GAS",
    "dex_latency": "DEX_LATENCY",
    "order_book_ws_url": "ORDER_BOOK_WS_URL",

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


# ---------------------------------------------------------------------------
#  Configuration file management helpers
# ---------------------------------------------------------------------------

CONFIG_DIR = os.getenv("CONFIG_DIR", "configs")
ACTIVE_CONFIG_FILE = os.path.join(CONFIG_DIR, "active")
os.makedirs(CONFIG_DIR, exist_ok=True)


def list_configs() -> list[str]:
    """Return all saved configuration file names."""
    return [
        f
        for f in os.listdir(CONFIG_DIR)
        if os.path.isfile(os.path.join(CONFIG_DIR, f))
        and not f.startswith(".")
    ]


def save_config(name: str, data: bytes) -> None:
    """Persist configuration ``data`` under ``name``.

    The name must not contain path traversal components.
    """
    if os.path.sep in name or (os.path.altsep and os.path.altsep in name) or ".." in name:
        raise ValueError("invalid config name")
    path = os.path.join(CONFIG_DIR, name)
    with open(path, "wb") as fh:
        fh.write(data)


def select_config(name: str) -> None:
    """Mark ``name`` as the active configuration."""
    path = os.path.join(CONFIG_DIR, name)
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    with open(ACTIVE_CONFIG_FILE, "w", encoding="utf-8") as fh:
        fh.write(name)


def get_active_config_name() -> str | None:
    try:
        with open(ACTIVE_CONFIG_FILE, "r", encoding="utf-8") as fh:
            return fh.read().strip() or None
    except FileNotFoundError:
        return None


def load_selected_config() -> dict:
    """Load the currently selected configuration file."""
    name = get_active_config_name()
    if not name:
        return {}
    path = os.path.join(CONFIG_DIR, name)
    if not os.path.exists(path):
        return {}
    return load_config(path)
