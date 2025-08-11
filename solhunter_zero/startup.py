"""Startup helpers for SolHunter Zero."""

from __future__ import annotations

import asyncio
import logging
import os
import time

from .config import (
    apply_env_overrides,
    load_config,
    set_env_from_config,
)
from .config_runtime import Config
from . import metrics_aggregator


def ensure_connectivity(*, offline: bool = False) -> None:
    """Verify Solana RPC and DEX websocket connectivity."""
    if offline:
        return

    from solhunter_zero.rpc_utils import ensure_rpc as _ensure_rpc
    from .dex_ws import stream_listed_tokens

    _ensure_rpc()

    url = os.getenv("DEX_LISTING_WS_URL", "")
    if not url:
        return

    raise_on_ws_fail = os.getenv("RAISE_ON_WS_FAIL", "").lower() in {
        "1",
        "true",
        "yes",
    }

    async def _check_ws() -> None:
        gen = stream_listed_tokens(url)
        try:
            await asyncio.wait_for(gen.__anext__(), timeout=1)
        except asyncio.TimeoutError:
            msg = "No data received from DEX listing websocket"
            logging.getLogger(__name__).warning(msg)
            if raise_on_ws_fail:
                raise RuntimeError(msg)
        finally:
            with contextlib.suppress(Exception):
                await gen.aclose()

    import contextlib

    asyncio.run(_check_ws())


def prepare_environment(
    config_path: str | None,
    *,
    offline: bool = False,
    dry_run: bool = False,
) -> tuple[dict, Config]:
    """Load configuration, set environment variables and verify connectivity."""
    start = time.perf_counter()
    cfg = apply_env_overrides(load_config(config_path))
    set_env_from_config(cfg)
    runtime_cfg = Config.from_env(cfg)
    metrics_aggregator.publish(
        "startup_config_load_duration", time.perf_counter() - start
    )

    start = time.perf_counter()
    ensure_connectivity(offline=offline or dry_run)
    metrics_aggregator.publish(
        "startup_connectivity_check_duration", time.perf_counter() - start
    )

    return cfg, runtime_cfg

