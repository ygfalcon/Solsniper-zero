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
from .config_model import Config, ENV_VARS
from dataclasses import fields

try:
    import yaml  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    yaml = None


# Commonly required environment variables
REQUIRED_ENV_VARS = (
    "EVENT_BUS_URL",
    "SOLANA_RPC_URL",
    "SOLANA_KEYPAIR",
)


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


def load_config(path: str | os.PathLike | None = None) -> Config:
    """Load configuration from ``path`` or default locations."""
    if path is None:
        path = find_config_file()
    data: dict[str, Any] = {}
    if path and Path(path).is_file():
        data = _read_config_file(Path(path))
    cfg = Config.from_dict(data)
    try:
        ConfigModel(**cfg.to_dict())
    except ValidationError as exc:
        raise ValueError(f"Invalid configuration: {exc}") from exc
    return cfg


def apply_env_overrides(config: Mapping[str, Any] | Config) -> Config:
    """Merge environment variable overrides into ``config``."""
    if isinstance(config, Config):
        cfg = Config.from_dict(config.to_dict())
    else:
        cfg = Config.from_dict(dict(config))
    return cfg


def set_env_from_config(config: Mapping[str, Any] | Config) -> None:
    """Set environment variables for values present in ``config``."""
    if not isinstance(config, Config):
        config = Config.from_dict(dict(config))
    for f in fields(Config):
        if f.name == "extra":
            continue
        env = f.metadata.get("env")
        if not env:
            continue
        val = getattr(config, f.name, None)
        if val is not None and os.getenv(env) is None:
            os.environ[env] = str(val)


def validate_config(cfg: Mapping[str, Any] | Config) -> dict:
    """Validate configuration against :class:`ConfigModel` schema.

    Returns the normalized configuration dictionary.
    """
    data = cfg.to_dict() if isinstance(cfg, Config) else dict(cfg)
    try:
        return ConfigModel(**data).dict()
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


def validate_env(required: Sequence[str], cfg_path: str | None = None) -> Config:
    """Ensure required environment variables are set.

    Missing values are filled from the configuration file when possible.
    Returns the loaded configuration dataclass.
    """
    cfg_data: Config = Config()
    if cfg_path is None:
        cfg_path = ensure_config_file()
    if cfg_path:
        cfg_data = load_config(cfg_path)
    env_to_key = {v: k for k, v in ENV_VARS.items()}
    missing: list[str] = []
    for name in required:
        if not os.getenv(name):
            val = None
            key = env_to_key.get(name)
            if key:
                val = getattr(cfg_data, key, None)
            if val is None:
                val = getattr(cfg_data, name, None)
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


def load_selected_config() -> Config:
    """Load the currently selected configuration file."""
    name = get_active_config_name()
    if not name:
        return Config()
    path = os.path.join(CONFIG_DIR, name)
    if not os.path.exists(path):
        return Config()
    return load_config(path)


def load_dex_config(config: Mapping[str, Any] | Config | None = None) -> DEXConfig:
    """Return :class:`DEXConfig` populated from ``config`` and environment."""

    cfg = apply_env_overrides(config if config is not None else {})

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

_ACTIVE_CONFIG: Config = load_selected_config()
set_env_from_config(_ACTIVE_CONFIG)


def _update_active(cfg: Mapping[str, Any] | Config | None) -> None:
    global _ACTIVE_CONFIG
    if cfg is None:
        cfg = Config()
    elif not isinstance(cfg, Config):
        cfg = Config.from_dict(dict(cfg))
    _ACTIVE_CONFIG = cfg
    set_env_from_config(_ACTIVE_CONFIG)


def reload_active_config() -> Config:
    """Reload the currently selected configuration and broadcast an update."""
    cfg = load_selected_config()
    _update_active(cfg)
    _publish("config_updated", cfg.to_dict())
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
