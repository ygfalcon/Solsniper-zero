from __future__ import annotations

import os
import sys
from .jsonutil import loads, dumps
import ast
from typing import Mapping, Any, Sequence
from pathlib import Path
from .dex_config import DEXConfig
from importlib import import_module

import tomllib
from pydantic import ValidationError

from .config_schema import ConfigModel

try:
    import yaml  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    yaml = None

try:  # pragma: no cover - optional dependency
    from dotenv import dotenv_values
except Exception:  # fallback when python-dotenv is unavailable
    dotenv_values = None


ENV_VARS = {
    "birdeye_api_key": "BIRDEYE_API_KEY",
    "solana_rpc_url": "SOLANA_RPC_URL",
    "solana_keypair": "SOLANA_KEYPAIR",
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
    "metrics_url": "METRICS_URL",
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
    "llm_model": "LLM_MODEL",
    "llm_context_length": "LLM_CONTEXT_LENGTH",
    "agents": "AGENTS",
    "agent_weights": "AGENT_WEIGHTS",
    "weights_path": "WEIGHTS_PATH",
    "dex_priorities": "DEX_PRIORITIES",
    "dex_fees": "DEX_FEES",
    "dex_gas": "DEX_GAS",
    "dex_latency": "DEX_LATENCY",
    "dex_latency_refresh_interval": "DEX_LATENCY_REFRESH_INTERVAL",
    "priority_fees": "PRIORITY_FEES",
    "priority_rpc": "PRIORITY_RPC",
    "jito_rpc_url": "JITO_RPC_URL",
    "jito_auth": "JITO_AUTH",
    "jito_ws_url": "JITO_WS_URL",
    "jito_ws_auth": "JITO_WS_AUTH",
    "event_bus_url": "EVENT_BUS_URL",
    "event_bus_peers": "EVENT_BUS_PEERS",
    "broker_url": "BROKER_URL",
    "broker_urls": "BROKER_URLS",
    "compress_events": "COMPRESS_EVENTS",
    "event_serialization": "EVENT_SERIALIZATION",
    "event_batch_ms": "EVENT_BATCH_MS",
    "event_mmap_batch_ms": "EVENT_MMAP_BATCH_MS",
    "event_mmap_batch_size": "EVENT_MMAP_BATCH_SIZE",
    "order_book_ws_url": "ORDER_BOOK_WS_URL",
    "depth_service": "DEPTH_SERVICE",
    "use_depth_stream": "USE_DEPTH_STREAM",
    "use_depth_feed": "USE_DEPTH_FEED",
    "use_rust_exec": "USE_RUST_EXEC",
    "use_service_exec": "USE_SERVICE_EXEC",
    "use_service_route": "USE_SERVICE_ROUTE",
    "mempool_score_threshold": "MEMPOOL_SCORE_THRESHOLD",
    "mempool_stats_window": "MEMPOOL_STATS_WINDOW",
    "use_flash_loans": "USE_FLASH_LOANS",
    "max_flash_amount": "MAX_FLASH_AMOUNT",
    "flash_loan_ratio": "FLASH_LOAN_RATIO",
    "use_mev_bundles": "USE_MEV_BUNDLES",
    "mempool_threshold": "MEMPOOL_THRESHOLD",
    "bundle_size": "BUNDLE_SIZE",
    "depth_threshold": "DEPTH_THRESHOLD",
    "depth_update_threshold": "DEPTH_UPDATE_THRESHOLD",
    "depth_min_send_interval": "DEPTH_MIN_SEND_INTERVAL",
    "cpu_low_threshold": "CPU_LOW_THRESHOLD",
    "cpu_high_threshold": "CPU_HIGH_THRESHOLD",
    "max_concurrency": "MAX_CONCURRENCY",
    "cpu_usage_threshold": "CPU_USAGE_THRESHOLD",
    "concurrency_smoothing": "CONCURRENCY_SMOOTHING",
    "concurrency_kp": "CONCURRENCY_KP",
    "concurrency_ki": "CONCURRENCY_KI",
    "min_rate": "MIN_RATE",
    "max_rate": "MAX_RATE",
    "depth_freq_low": "DEPTH_FREQ_LOW",
    "depth_freq_high": "DEPTH_FREQ_HIGH",
    "max_hops": "MAX_HOPS",
    "path_algorithm": "PATH_ALGORITHM",
    "use_gnn_routing": "USE_GNN_ROUTING",
    "gnn_model_path": "GNN_MODEL_PATH",
    "offline_data_interval": "OFFLINE_DATA_INTERVAL",
    "gpu_memory_index": "GPU_MEMORY_INDEX",
    "memory_sync_interval": "MEMORY_SYNC_INTERVAL",
    "use_gpu_sim": "USE_GPU_SIM",
    "rl_build_mmap_dataset": "RL_BUILD_MMAP_DATASET",
    "rl_prefetch_buffer": "RL_PREFETCH_BUFFER",
}


def load_env_file(path: str | os.PathLike) -> dict[str, str]:
    """Load environment variables from a ``.env`` style file.

    Uses :mod:`python-dotenv` when available for robust parsing. Existing
    variables in :data:`os.environ` are not overwritten.  Returns a mapping
    of variables that were loaded.
    """

    p = Path(path)
    if not p.exists():
        return {}

    data: dict[str, str]
    if dotenv_values is not None:
        data = {k: v for k, v in dotenv_values(p).items() if v is not None}
    else:  # fallback simple parser
        data = {}
        for raw_line in p.read_text().splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            data[key.strip()] = value.strip().strip("'\"")

    for key, value in data.items():
        os.environ.setdefault(key, value)
    return data


def _publish(topic: str, payload: Any) -> None:
    """Proxy to :func:`event_bus.publish` imported lazily."""
    ev = import_module("solhunter_zero.event_bus")
    ev.publish(topic, payload)


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
        path = find_config_file()
    if path is None:
        return {}
    cfg = _read_config_file(Path(path))
    try:
        cfg = ConfigModel(**cfg).dict()
    except ValidationError as exc:
        raise ValueError(f"Invalid configuration: {exc}") from exc
    return cfg


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
def validate_config(cfg: Mapping[str, Any]) -> dict:
    """Validate configuration against :class:`ConfigModel` schema.

    Returns the normalized configuration dictionary.
    """
    try:
        return ConfigModel(**cfg).dict()
    except ValidationError as exc:
        raise ValueError(f"Invalid configuration: {exc}") from exc


# ---------------------------------------------------------------------------
#  Configuration file management helpers
# ---------------------------------------------------------------------------

CONFIG_DIR = os.getenv("CONFIG_DIR", "configs")
ACTIVE_CONFIG_FILE = os.path.join(CONFIG_DIR, "active")
os.makedirs(CONFIG_DIR, exist_ok=True)


def find_config_file() -> str | None:
    """Return the first existing configuration file path.

    The search order is ``config.toml``, ``config.yaml`` and ``config.yml``.
    The ``SOLHUNTER_CONFIG`` environment variable, when set, takes precedence.
    """

    path = os.getenv("SOLHUNTER_CONFIG")
    if path and Path(path).is_file():
        return path
    for name in ("config.toml", "config.yaml", "config.yml"):
        p = Path(name)
        if p.is_file():
            return str(p)
    return None


def ensure_config_file() -> str | None:
    """Ensure a configuration file exists, generating a default if needed."""
    path = find_config_file()
    if path:
        return path
    try:
        from scripts import quick_setup

        quick_setup.main(["--auto"])
    except Exception:
        return None
    return find_config_file()


def validate_env(required: Sequence[str], cfg_path: str | None = None) -> dict:
    """Ensure required environment variables are set.

    Missing values are filled from the configuration file when possible.
    Returns the loaded configuration dictionary.
    """
    cfg_data: dict[str, Any] = {}
    if cfg_path is None:
        cfg_path = ensure_config_file()
    if cfg_path:
        cfg_data = apply_env_overrides(load_config(cfg_path))
    env_to_key = {v: k for k, v in ENV_VARS.items()}
    missing: list[str] = []
    for name in required:
        if not os.getenv(name):
            val = None
            key = env_to_key.get(name)
            if key:
                val = cfg_data.get(key)
            if val is None:
                val = cfg_data.get(name)
            if val is not None:
                os.environ[name] = str(val)
            if not os.getenv(name):
                missing.append(name)
    if missing:
        for name in missing:
            print(f"Required env var {name} is not set", file=sys.stderr)
        raise SystemExit(1)
    return cfg_data


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
    _publish("config_updated", cfg)


def select_config(name: str) -> None:
    """Mark ``name`` as the active configuration."""
    path = os.path.join(CONFIG_DIR, name)
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    with open(ACTIVE_CONFIG_FILE, "w", encoding="utf-8") as fh:
        fh.write(name)
    reload_active_config()


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
                data = loads(val)
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


# ---------------------------------------------------------------------------
#  Active configuration helpers
# ---------------------------------------------------------------------------

_ACTIVE_CONFIG: dict[str, Any] = apply_env_overrides(load_selected_config())
set_env_from_config(_ACTIVE_CONFIG)


def _update_active(cfg: Mapping[str, Any] | None) -> None:
    global _ACTIVE_CONFIG
    if cfg is None:
        cfg = {}
    _ACTIVE_CONFIG = apply_env_overrides(dict(cfg))
    set_env_from_config(_ACTIVE_CONFIG)


def reload_active_config() -> dict:
    """Reload the currently selected configuration and broadcast an update."""
    cfg = load_selected_config()
    _update_active(cfg)
    _publish("config_updated", cfg)
    return _ACTIVE_CONFIG


from . import event_bus as _event_bus

_sub = _event_bus.subscription("config_updated", _update_active)
_sub.__enter__()
try:
    _event_bus._reload_bus(None)
    _event_bus._reload_broker(None)
    _event_bus._reload_serialization(None)
except Exception:
    pass


def get_event_bus_url(cfg: Mapping[str, Any] | None = None) -> str | None:
    """Return websocket URL of the external event bus if configured."""
    cfg = cfg or _ACTIVE_CONFIG
    url = os.getenv("EVENT_BUS_URL") or str(cfg.get("event_bus_url", ""))
    return url or None


def get_event_bus_peers(cfg: Mapping[str, Any] | None = None) -> list[str]:
    """Return list of peer URLs for the event bus."""
    cfg = cfg or _ACTIVE_CONFIG
    env_val = os.getenv("EVENT_BUS_PEERS")
    if env_val:
        peers = [u.strip() for u in env_val.split(",") if u.strip()]
    else:
        raw = cfg.get("event_bus_peers")
        if isinstance(raw, str):
            peers = [u.strip() for u in raw.split(",") if u.strip()]
        elif isinstance(raw, (list, tuple, set)):
            peers = [str(u).strip() for u in raw if str(u).strip()]
        else:
            peers = []
    return peers


def get_broker_url(cfg: Mapping[str, Any] | None = None) -> str | None:
    """Return message broker URL if configured (first of :func:`get_broker_urls`)."""
    urls = get_broker_urls(cfg)
    return urls[0] if urls else None


def get_broker_urls(cfg: Mapping[str, Any] | None = None) -> list[str]:
    """Return list of message broker URLs."""
    cfg = cfg or _ACTIVE_CONFIG
    env_val = os.getenv("BROKER_URLS")
    if env_val:
        urls = [u.strip() for u in env_val.split(",") if u.strip()]
    else:
        raw = cfg.get("broker_urls")
        if isinstance(raw, str):
            urls = [u.strip() for u in raw.split(",") if u.strip()]
        elif isinstance(raw, (list, tuple, set)):
            urls = [str(u).strip() for u in raw if str(u).strip()]
        else:
            urls = []
    if not urls:
        url = os.getenv("BROKER_URL") or str(cfg.get("broker_url", ""))
        if url:
            urls = [url]
    return urls
    

def get_event_serialization(cfg: Mapping[str, Any] | None = None) -> str | None:
    """Return configured event serialization format."""
    cfg = cfg or _ACTIVE_CONFIG
    val = os.getenv("EVENT_SERIALIZATION") or str(cfg.get("event_serialization", ""))
    return val or None


def get_event_batch_ms(cfg: Mapping[str, Any] | None = None) -> int:
    """Return event batching interval in milliseconds."""
    cfg = cfg or _ACTIVE_CONFIG
    val = os.getenv("EVENT_BATCH_MS")
    if val is None or val == "":
        val = cfg.get("event_batch_ms", 0)
    return int(val or 0)


def get_event_mmap_batch_ms(cfg: Mapping[str, Any] | None = None) -> int:
    """Return mmap event batching interval in milliseconds."""
    cfg = cfg or _ACTIVE_CONFIG
    val = os.getenv("EVENT_MMAP_BATCH_MS")
    if val is None or val == "":
        val = cfg.get("event_mmap_batch_ms", 0)
    return int(val or 0)


def get_event_mmap_batch_size(cfg: Mapping[str, Any] | None = None) -> int:
    """Return number of events to buffer before mmap flush."""
    cfg = cfg or _ACTIVE_CONFIG
    val = os.getenv("EVENT_MMAP_BATCH_SIZE")
    if val is None or val == "":
        val = cfg.get("event_mmap_batch_size", 0)
    return int(val or 0)


def get_depth_ws_addr(cfg: Mapping[str, Any] | None = None) -> tuple[str, int]:
    """Return address and port of the depth websocket server."""
    cfg = cfg or _ACTIVE_CONFIG
    addr = os.getenv("DEPTH_WS_ADDR") or str(cfg.get("depth_ws_addr", "127.0.0.1"))
    port = int(os.getenv("DEPTH_WS_PORT") or cfg.get("depth_ws_port", 8765))
    return addr, port
