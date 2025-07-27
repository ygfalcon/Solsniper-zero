from __future__ import annotations

import os
import json
import ast
from typing import Mapping, Any
from pathlib import Path

from .dex_config import DEXConfig
from .event_bus import publish

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
    "phoenix_api_url": "PHOENIX_API_URL",
    "meteora_api_url": "METEORA_API_URL",
    "orca_ws_url": "ORCA_WS_URL",
    "raydium_ws_url": "RAYDIUM_WS_URL",
    "phoenix_ws_url": "PHOENIX_WS_URL",
    "meteora_ws_url": "METEORA_WS_URL",
    "jupiter_ws_url": "JUPITER_WS_URL",
    "orca_dex_url": "ORCA_DEX_URL",
    "raydium_dex_url": "RAYDIUM_DEX_URL",
    "phoenix_dex_url": "PHOENIX_DEX_URL",
    "meteora_dex_url": "METEORA_DEX_URL",
    "metrics_base_url": "METRICS_BASE_URL",
    "news_feeds": "NEWS_FEEDS",
    "twitter_feeds": "TWITTER_FEEDS",
    "discord_feeds": "DISCORD_FEEDS",
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
    "offline_data_limit_gb": "OFFLINE_DATA_LIMIT_GB",
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
    "priority_fees": "PRIORITY_FEES",
    "priority_rpc": "PRIORITY_RPC",
    "jito_rpc_url": "JITO_RPC_URL",
    "jito_auth": "JITO_AUTH",
    "jito_ws_url": "JITO_WS_URL",
    "jito_ws_auth": "JITO_WS_AUTH",
    "order_book_ws_url": "ORDER_BOOK_WS_URL",
    "depth_service": "DEPTH_SERVICE",
    "use_depth_stream": "USE_DEPTH_STREAM",
    "use_rust_exec": "USE_RUST_EXEC",
    "use_service_exec": "USE_SERVICE_EXEC",
    "mempool_score_threshold": "MEMPOOL_SCORE_THRESHOLD",
    "mempool_stats_window": "MEMPOOL_STATS_WINDOW",
    "use_flash_loans": "USE_FLASH_LOANS",
    "max_flash_amount": "MAX_FLASH_AMOUNT",
    "use_mev_bundles": "USE_MEV_BUNDLES",
    "mempool_threshold": "MEMPOOL_THRESHOLD",
    "bundle_size": "BUNDLE_SIZE",
    "depth_threshold": "DEPTH_THRESHOLD",
    "max_hops": "MAX_HOPS",
    "path_algorithm": "PATH_ALGORITHM",
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
        if os.path.isfile(os.path.join(CONFIG_DIR, f)) and not f.startswith(".")
    ]


def save_config(name: str, data: bytes) -> None:
    """Persist configuration ``data`` under ``name``.

    The name must not contain path traversal components.
    """
    if (
        os.path.sep in name
        or (os.path.altsep and os.path.altsep in name)
        or ".." in name
    ):
        raise ValueError("invalid config name")
    path = os.path.join(CONFIG_DIR, name)
    with open(path, "wb") as fh:
        fh.write(data)
    cfg = {}
    try:
        cfg = _read_config_file(Path(path))
    except Exception:
        pass
    publish("config_updated", cfg)


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


def load_dex_config(config: Mapping[str, Any] | None = None) -> DEXConfig:
    """Return :class:`DEXConfig` populated from ``config`` and environment."""

    cfg = apply_env_overrides(config or {})

    base = str(cfg.get("dex_base_url", "https://quote-api.jup.ag"))
    testnet = str(cfg.get("dex_testnet_url", base))

    def _url(name: str) -> str:
        return str(cfg.get(name, base))

    venue_urls = {
        "jupiter": base,
        "raydium": _url("raydium_dex_url"),
        "orca": _url("orca_dex_url"),
        "phoenix": _url("phoenix_dex_url"),
        "meteora": _url("meteora_dex_url"),
    }

    def _parse_map(val: Any) -> dict[str, float]:
        if not val:
            return {}
        if isinstance(val, Mapping):
            data = val
        elif isinstance(val, str):
            try:
                data = json.loads(val)
            except Exception:
                try:
                    data = ast.literal_eval(val)
                except Exception:
                    return {}
        else:
            return {}
        return {str(k): float(v) for k, v in data.items()}

    fees = _parse_map(cfg.get("dex_fees"))
    gas = _parse_map(cfg.get("dex_gas"))
    latency = _parse_map(cfg.get("dex_latency"))

    return DEXConfig(
        base_url=base,
        testnet_url=testnet,
        venue_urls=venue_urls,
        fees=fees,
        gas=gas,
        latency=latency,
    )
