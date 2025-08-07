from __future__ import annotations

from dataclasses import dataclass, field, fields
from typing import Any, Dict, Iterator, MutableMapping
import os

# Mapping of configuration field names to environment variable names.
ENV_VARS: Dict[str, str] = {
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
    "broker_heartbeat_interval": "BROKER_HEARTBEAT_INTERVAL",
    "broker_retry_limit": "BROKER_RETRY_LIMIT",
    "compress_events": "COMPRESS_EVENTS",
    "event_serialization": "EVENT_SERIALIZATION",
    "event_batch_ms": "EVENT_BATCH_MS",
    "event_mmap_batch_ms": "EVENT_MMAP_BATCH_MS",
    "event_mmap_batch_size": "EVENT_MMAP_BATCH_SIZE",
    "order_book_ws_url": "ORDER_BOOK_WS_URL",
    "depth_service": "DEPTH_SERVICE",
    "depth_max_restarts": "DEPTH_MAX_RESTARTS",
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
    "memory_snapshot_path": "MEMORY_SNAPSHOT_PATH",
    "use_gpu_sim": "USE_GPU_SIM",
    "rl_build_mmap_dataset": "RL_BUILD_MMAP_DATASET",
    "rl_prefetch_buffer": "RL_PREFETCH_BUFFER",
}


@dataclass
class Config(MutableMapping[str, Any]):
    """Dataclass representing all configuration values.

    The fields mirror the keys in :data:`ENV_VARS`.  Environment overrides are
    applied using the ``env`` metadata on each field.
    """

    birdeye_api_key: Any | None = field(
        default=None, metadata={"env": "BIRDEYE_API_KEY"}
    )
    solana_rpc_url: Any | None = field(
        default=None, metadata={"env": "SOLANA_RPC_URL"}
    )
    solana_keypair: Any | None = field(
        default=None, metadata={"env": "SOLANA_KEYPAIR"}
    )
    dex_base_url: Any | None = field(
        default=None, metadata={"env": "DEX_BASE_URL"}
    )
    dex_testnet_url: Any | None = field(
        default=None, metadata={"env": "DEX_TESTNET_URL"}
    )
    orca_api_url: Any | None = field(default=None, metadata={"env": "ORCA_API_URL"})
    raydium_api_url: Any | None = field(
        default=None, metadata={"env": "RAYDIUM_API_URL"}
    )
    phoenix_api_url: Any | None = field(
        default=None, metadata={"env": "PHOENIX_API_URL"}
    )
    meteora_api_url: Any | None = field(
        default=None, metadata={"env": "METEORA_API_URL"}
    )
    orca_ws_url: Any | None = field(default=None, metadata={"env": "ORCA_WS_URL"})
    raydium_ws_url: Any | None = field(
        default=None, metadata={"env": "RAYDIUM_WS_URL"}
    )
    phoenix_ws_url: Any | None = field(
        default=None, metadata={"env": "PHOENIX_WS_URL"}
    )
    meteora_ws_url: Any | None = field(
        default=None, metadata={"env": "METEORA_WS_URL"}
    )
    jupiter_ws_url: Any | None = field(
        default=None, metadata={"env": "JUPITER_WS_URL"}
    )
    orca_dex_url: Any | None = field(
        default=None, metadata={"env": "ORCA_DEX_URL"}
    )
    raydium_dex_url: Any | None = field(
        default=None, metadata={"env": "RAYDIUM_DEX_URL"}
    )
    phoenix_dex_url: Any | None = field(
        default=None, metadata={"env": "PHOENIX_DEX_URL"}
    )
    meteora_dex_url: Any | None = field(
        default=None, metadata={"env": "METEORA_DEX_URL"}
    )
    metrics_base_url: Any | None = field(
        default=None, metadata={"env": "METRICS_BASE_URL"}
    )
    metrics_url: Any | None = field(default=None, metadata={"env": "METRICS_URL"})
    news_feeds: Any | None = field(default=None, metadata={"env": "NEWS_FEEDS"})
    twitter_feeds: Any | None = field(
        default=None, metadata={"env": "TWITTER_FEEDS"}
    )
    discord_feeds: Any | None = field(
        default=None, metadata={"env": "DISCORD_FEEDS"}
    )
    discovery_method: Any | None = field(
        default=None, metadata={"env": "DISCOVERY_METHOD"}
    )
    risk_tolerance: Any | None = field(
        default=None, metadata={"env": "RISK_TOLERANCE"}
    )
    max_allocation: Any | None = field(
        default=None, metadata={"env": "MAX_ALLOCATION"}
    )
    max_risk_per_token: Any | None = field(
        default=None, metadata={"env": "MAX_RISK_PER_TOKEN"}
    )
    risk_multiplier: Any | None = field(
        default=None, metadata={"env": "RISK_MULTIPLIER"}
    )
    trailing_stop: Any | None = field(
        default=None, metadata={"env": "TRAILING_STOP"}
    )
    max_drawdown: Any | None = field(
        default=None, metadata={"env": "MAX_DRAWDOWN"}
    )
    volatility_factor: Any | None = field(
        default=None, metadata={"env": "VOLATILITY_FACTOR"}
    )
    arbitrage_threshold: Any | None = field(
        default=None, metadata={"env": "ARBITRAGE_THRESHOLD"}
    )
    arbitrage_amount: Any | None = field(
        default=None, metadata={"env": "ARBITRAGE_AMOUNT"}
    )
    min_portfolio_value: Any | None = field(
        default=None, metadata={"env": "MIN_PORTFOLIO_VALUE"}
    )
    min_delay: Any | None = field(default=None, metadata={"env": "MIN_DELAY"})
    max_delay: Any | None = field(default=None, metadata={"env": "MAX_DELAY"})
    offline_data_limit_gb: Any | None = field(
        default=None, metadata={"env": "OFFLINE_DATA_LIMIT_GB"}
    )
    strategies: Any | None = field(default=None, metadata={"env": "STRATEGIES"})
    token_suffix: Any | None = field(
        default=None, metadata={"env": "TOKEN_SUFFIX"}
    )
    token_keywords: Any | None = field(
        default=None, metadata={"env": "TOKEN_KEYWORDS"}
    )
    volume_threshold: Any | None = field(
        default=None, metadata={"env": "VOLUME_THRESHOLD"}
    )
    llm_model: Any | None = field(default=None, metadata={"env": "LLM_MODEL"})
    llm_context_length: Any | None = field(
        default=None, metadata={"env": "LLM_CONTEXT_LENGTH"}
    )
    agents: Any | None = field(default=None, metadata={"env": "AGENTS"})
    agent_weights: Any | None = field(
        default=None, metadata={"env": "AGENT_WEIGHTS"}
    )
    weights_path: Any | None = field(
        default=None, metadata={"env": "WEIGHTS_PATH"}
    )
    dex_priorities: Any | None = field(
        default=None, metadata={"env": "DEX_PRIORITIES"}
    )
    dex_fees: Any | None = field(default=None, metadata={"env": "DEX_FEES"})
    dex_gas: Any | None = field(default=None, metadata={"env": "DEX_GAS"})
    dex_latency: Any | None = field(default=None, metadata={"env": "DEX_LATENCY"})
    dex_latency_refresh_interval: Any | None = field(
        default=None, metadata={"env": "DEX_LATENCY_REFRESH_INTERVAL"}
    )
    priority_fees: Any | None = field(
        default=None, metadata={"env": "PRIORITY_FEES"}
    )
    priority_rpc: Any | None = field(default=None, metadata={"env": "PRIORITY_RPC"})
    jito_rpc_url: Any | None = field(
        default=None, metadata={"env": "JITO_RPC_URL"}
    )
    jito_auth: Any | None = field(default=None, metadata={"env": "JITO_AUTH"})
    jito_ws_url: Any | None = field(
        default=None, metadata={"env": "JITO_WS_URL"}
    )
    jito_ws_auth: Any | None = field(
        default=None, metadata={"env": "JITO_WS_AUTH"}
    )
    event_bus_url: Any | None = field(
        default=None, metadata={"env": "EVENT_BUS_URL"}
    )
    event_bus_peers: Any | None = field(
        default=None, metadata={"env": "EVENT_BUS_PEERS"}
    )
    broker_url: Any | None = field(default=None, metadata={"env": "BROKER_URL"})
    broker_urls: Any | None = field(default=None, metadata={"env": "BROKER_URLS"})
    broker_heartbeat_interval: Any | None = field(
        default=None, metadata={"env": "BROKER_HEARTBEAT_INTERVAL"}
    )
    broker_retry_limit: Any | None = field(
        default=None, metadata={"env": "BROKER_RETRY_LIMIT"}
    )
    compress_events: Any | None = field(
        default=None, metadata={"env": "COMPRESS_EVENTS"}
    )
    event_serialization: Any | None = field(
        default=None, metadata={"env": "EVENT_SERIALIZATION"}
    )
    event_batch_ms: Any | None = field(
        default=None, metadata={"env": "EVENT_BATCH_MS"}
    )
    event_mmap_batch_ms: Any | None = field(
        default=None, metadata={"env": "EVENT_MMAP_BATCH_MS"}
    )
    event_mmap_batch_size: Any | None = field(
        default=None, metadata={"env": "EVENT_MMAP_BATCH_SIZE"}
    )
    order_book_ws_url: Any | None = field(
        default=None, metadata={"env": "ORDER_BOOK_WS_URL"}
    )
    depth_service: Any | None = field(
        default=None, metadata={"env": "DEPTH_SERVICE"}
    )
    depth_max_restarts: Any | None = field(
        default=None, metadata={"env": "DEPTH_MAX_RESTARTS"}
    )
    use_depth_stream: Any | None = field(
        default=None, metadata={"env": "USE_DEPTH_STREAM"}
    )
    use_depth_feed: Any | None = field(
        default=None, metadata={"env": "USE_DEPTH_FEED"}
    )
    use_rust_exec: Any | None = field(
        default=None, metadata={"env": "USE_RUST_EXEC"}
    )
    use_service_exec: Any | None = field(
        default=None, metadata={"env": "USE_SERVICE_EXEC"}
    )
    use_service_route: Any | None = field(
        default=None, metadata={"env": "USE_SERVICE_ROUTE"}
    )
    mempool_score_threshold: Any | None = field(
        default=None, metadata={"env": "MEMPOOL_SCORE_THRESHOLD"}
    )
    mempool_stats_window: Any | None = field(
        default=None, metadata={"env": "MEMPOOL_STATS_WINDOW"}
    )
    use_flash_loans: Any | None = field(
        default=None, metadata={"env": "USE_FLASH_LOANS"}
    )
    max_flash_amount: Any | None = field(
        default=None, metadata={"env": "MAX_FLASH_AMOUNT"}
    )
    flash_loan_ratio: Any | None = field(
        default=None, metadata={"env": "FLASH_LOAN_RATIO"}
    )
    use_mev_bundles: Any | None = field(
        default=None, metadata={"env": "USE_MEV_BUNDLES"}
    )
    mempool_threshold: Any | None = field(
        default=None, metadata={"env": "MEMPOOL_THRESHOLD"}
    )
    bundle_size: Any | None = field(default=None, metadata={"env": "BUNDLE_SIZE"})
    depth_threshold: Any | None = field(
        default=None, metadata={"env": "DEPTH_THRESHOLD"}
    )
    depth_update_threshold: Any | None = field(
        default=None, metadata={"env": "DEPTH_UPDATE_THRESHOLD"}
    )
    depth_min_send_interval: Any | None = field(
        default=None, metadata={"env": "DEPTH_MIN_SEND_INTERVAL"}
    )
    cpu_low_threshold: Any | None = field(
        default=None, metadata={"env": "CPU_LOW_THRESHOLD"}
    )
    cpu_high_threshold: Any | None = field(
        default=None, metadata={"env": "CPU_HIGH_THRESHOLD"}
    )
    max_concurrency: Any | None = field(
        default=None, metadata={"env": "MAX_CONCURRENCY"}
    )
    cpu_usage_threshold: Any | None = field(
        default=None, metadata={"env": "CPU_USAGE_THRESHOLD"}
    )
    concurrency_smoothing: Any | None = field(
        default=None, metadata={"env": "CONCURRENCY_SMOOTHING"}
    )
    concurrency_kp: Any | None = field(
        default=None, metadata={"env": "CONCURRENCY_KP"}
    )
    concurrency_ki: Any | None = field(
        default=None, metadata={"env": "CONCURRENCY_KI"}
    )
    min_rate: Any | None = field(default=None, metadata={"env": "MIN_RATE"})
    max_rate: Any | None = field(default=None, metadata={"env": "MAX_RATE"})
    depth_freq_low: Any | None = field(
        default=None, metadata={"env": "DEPTH_FREQ_LOW"}
    )
    depth_freq_high: Any | None = field(
        default=None, metadata={"env": "DEPTH_FREQ_HIGH"}
    )
    max_hops: Any | None = field(default=None, metadata={"env": "MAX_HOPS"})
    path_algorithm: Any | None = field(
        default=None, metadata={"env": "PATH_ALGORITHM"}
    )
    use_gnn_routing: Any | None = field(
        default=None, metadata={"env": "USE_GNN_ROUTING"}
    )
    gnn_model_path: Any | None = field(
        default=None, metadata={"env": "GNN_MODEL_PATH"}
    )
    offline_data_interval: Any | None = field(
        default=None, metadata={"env": "OFFLINE_DATA_INTERVAL"}
    )
    gpu_memory_index: Any | None = field(
        default=None, metadata={"env": "GPU_MEMORY_INDEX"}
    )
    memory_sync_interval: Any | None = field(
        default=None, metadata={"env": "MEMORY_SYNC_INTERVAL"}
    )
    memory_snapshot_path: Any | None = field(
        default=None, metadata={"env": "MEMORY_SNAPSHOT_PATH"}
    )
    use_gpu_sim: Any | None = field(
        default=None, metadata={"env": "USE_GPU_SIM"}
    )
    rl_build_mmap_dataset: Any | None = field(
        default=None, metadata={"env": "RL_BUILD_MMAP_DATASET"}
    )
    rl_prefetch_buffer: Any | None = field(
        default=None, metadata={"env": "RL_PREFETCH_BUFFER"}
    )

    # store unknown fields from config files
    extra: Dict[str, Any] = field(default_factory=dict, repr=False)

    # ------------------------------------------------------------------
    # Mapping protocol implementation
    # ------------------------------------------------------------------
    def __getitem__(self, key: str) -> Any:  # type: ignore[override]
        if key in self.__dataclass_fields__ and key != "extra":
            return getattr(self, key)
        return self.extra[key]

    def __setitem__(self, key: str, value: Any) -> None:  # type: ignore[override]
        if key in self.__dataclass_fields__ and key != "extra":
            setattr(self, key, value)
        else:
            self.extra[key] = value

    def __delitem__(self, key: str) -> None:  # type: ignore[override]
        if key in self.__dataclass_fields__ and key != "extra":
            setattr(self, key, None)
        else:
            del self.extra[key]

    def __iter__(self) -> Iterator[str]:  # type: ignore[override]
        for f in self.__dataclass_fields__:
            if f != "extra":
                yield f
        for k in self.extra:
            yield k

    def __len__(self) -> int:  # type: ignore[override]
        return sum(1 for _ in self.__iter__())

    def __getattr__(self, name: str) -> Any:
        try:
            return self.extra[name]
        except KeyError as exc:
            raise AttributeError(name) from exc

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def to_dict(self) -> Dict[str, Any]:
        data = {f: getattr(self, f) for f in self.__dataclass_fields__ if f != "extra"}
        data.update(self.extra)
        return data

    def apply_env_overrides(self) -> None:
        """Override fields with values from the environment."""
        for f in fields(self):
            if f.name == "extra":
                continue
            env = f.metadata.get("env")
            if env:
                val = os.getenv(env)
                if val is not None:
                    setattr(self, f.name, val)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "Config":
        cfg = cls()
        known = set(cfg.__dataclass_fields__) - {"extra"}
        for k, v in data.items():
            if k in known:
                setattr(cfg, k, v)
            else:
                cfg.extra[k] = v
        cfg.apply_env_overrides()
        return cfg
