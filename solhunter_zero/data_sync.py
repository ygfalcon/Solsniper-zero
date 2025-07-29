import asyncio
import logging
import os
from pathlib import Path
from typing import Sequence

import aiohttp

from .offline_data import OfflineData, MarketSnapshot
from .token_scanner import scan_tokens_async
from .simulation import DEFAULT_METRICS_BASE_URL
from .news import fetch_sentiment_async

logger = logging.getLogger(__name__)

DEFAULT_LIMIT_GB = 50.0


def _prune_db(data: OfflineData, db_path: str, limit_gb: float) -> None:
    path = Path(db_path)
    limit_bytes = limit_gb * 1024 ** 3
    while path.exists() and path.stat().st_size > limit_bytes:
        with data.Session() as session:
            snap = (
                session.query(MarketSnapshot)
                .order_by(MarketSnapshot.timestamp)
                .first()
            )
            if snap is None:
                break
            session.delete(snap)
            session.commit()


async def sync_snapshots(
    tokens: Sequence[str],
    *,
    days: int = 3,
    db_path: str = "offline_data.db",
    base_url: str | None = None,
    limit_gb: float | None = None,
    concurrency: int = 5,
) -> None:
    """Download order-book snapshots and insert them into ``db_path``."""

    base_url = base_url or os.getenv("METRICS_BASE_URL", DEFAULT_METRICS_BASE_URL)
    limit_gb = (
        float(os.getenv("OFFLINE_DATA_LIMIT_GB", DEFAULT_LIMIT_GB))
        if limit_gb is None
        else limit_gb
    )
    data = OfflineData(f"sqlite:///{db_path}")

    feeds = [u for u in os.getenv("NEWS_FEEDS", "").split(",") if u]
    twitter_feeds = [u for u in os.getenv("TWITTER_FEEDS", "").split(",") if u]
    discord_feeds = [u for u in os.getenv("DISCORD_FEEDS", "").split(",") if u]
    sentiment = 0.0
    if feeds or twitter_feeds or discord_feeds:
        try:
            sentiment = await fetch_sentiment_async(
                feeds,
                twitter_urls=twitter_feeds,
                discord_urls=discord_feeds,
            )
        except Exception as exc:  # pragma: no cover - network errors
            logger.warning("failed to fetch sentiment: %s", exc)

    sem = asyncio.Semaphore(concurrency)
    async with aiohttp.ClientSession() as session:
        async def fetch_and_log(token: str) -> None:
            url = f"{base_url.rstrip('/')}/token/{token}/history?days={days}"
            try:
                async with sem:
                    async with session.get(url, timeout=10) as resp:
                        resp.raise_for_status()
                        resp_data = await resp.json()
            except Exception as exc:  # pragma: no cover - network errors
                logger.warning("failed to fetch snapshots for %s: %s", token, exc)
                return
            for snap in resp_data.get("snapshots", []):
                try:
                    data.log_snapshot(
                        token=token,
                        price=float(snap.get("price", 0.0)),
                        depth=float(snap.get("depth", 0.0)),
                        total_depth=float(snap.get("total_depth", 0.0)),
                        imbalance=float(snap.get("imbalance", 0.0)),
                        slippage=float(snap.get("slippage", 0.0)),
                        volume=float(snap.get("volume", 0.0)),
                        sentiment=sentiment,
                    )
                except Exception as exc:  # pragma: no cover - bad data
                    logger.warning("invalid snapshot for %s: %s", token, exc)

        await asyncio.gather(*(fetch_and_log(token) for token in tokens))

    _prune_db(data, db_path, limit_gb)


async def sync_recent(days: int = 3, db_path: str = "offline_data.db") -> None:
    """Discover tokens and sync recent snapshots."""

    tokens = await scan_tokens_async(
        offline=False, token_file=None, method=os.getenv("DISCOVERY_METHOD", "websocket")
    )
    if tokens:
        await sync_snapshots(tokens, days=days, db_path=db_path)

