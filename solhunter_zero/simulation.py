from __future__ import annotations

import logging
import os
from dataclasses import dataclass
import aiohttp
import asyncio
from typing import List

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor

try:  # pragma: no cover - optional dependency
    from xgboost import XGBRegressor  # type: ignore
except Exception:  # pragma: no cover - when xgboost is missing
    XGBRegressor = None

import numpy as np
from . import onchain_metrics, models
from .http import get_session
from .lru import TTLCache

logger = logging.getLogger(__name__)

# Default base URL for the metrics API. Can be overridden by the
# ``METRICS_BASE_URL`` environment variable or via a configuration file that
# sets this variable before the module is imported.
DEFAULT_METRICS_BASE_URL = "https://api.example.com"

# Recent trade ROI history used for bias adjustment
_TRADE_ROIS: list[float] = []

# Stored bias values updated via :func:`bias_correction`
_BIAS: dict[str, float] = {"mean": 0.0, "volatility": 0.0}

# Cached price model and modification time for fast reloads
_PRICE_MODEL = None
_PRICE_MTIME = 0.0

# module level cache for token metrics
TOKEN_METRICS_CACHE_TTL = 30  # seconds
TOKEN_METRICS_CACHE = TTLCache(maxsize=256, ttl=TOKEN_METRICS_CACHE_TTL)


def get_price_model(model_path: str | None = None):
    """Return cached price model reloading when the file changes."""
    global _PRICE_MODEL, _PRICE_MTIME
    model_path = model_path or os.getenv("PRICE_MODEL_PATH")
    if not model_path or not os.path.exists(model_path):
        _PRICE_MODEL = None
        _PRICE_MTIME = 0.0
        return None
    try:
        mtime = os.path.getmtime(model_path)
    except OSError:
        return None
    if _PRICE_MODEL is None or mtime > _PRICE_MTIME:
        _PRICE_MODEL = models.get_model(model_path, reload=True)
        _PRICE_MTIME = mtime
    return _PRICE_MODEL


def log_trade_outcome(roi: float) -> None:
    """Record a realized trade ROI for later bias correction."""
    _TRADE_ROIS.append(float(roi))


def bias_correction(window: int = 20) -> dict[str, float]:
    """Recompute prediction bias from recent trade outcomes."""
    if not _TRADE_ROIS:
        return _BIAS

    recent = _TRADE_ROIS[-window:]
    _BIAS["mean"] = float(np.mean(recent))
    _BIAS["volatility"] = float(np.std(recent))
    return _BIAS


@dataclass
class SimulationResult:
    success_prob: float
    expected_roi: float
    volume: float = 0.0
    liquidity: float = 0.0

    slippage: float = 0.0
    volatility: float = 0.0
    volume_spike: float = 1.0

    sentiment: float = 0.0
    order_book_strength: float = 0.0

    token_age: float = 0.0
    initial_liquidity: float = 0.0
    tx_trend: float = 0.0

    depth_change: float = 0.0
    tx_rate: float = 0.0
    whale_activity: float = 0.0



async def fetch_token_metrics_async(token: str) -> dict:
    """Asynchronously fetch historical return metrics for ``token``."""

    cached = TOKEN_METRICS_CACHE.get(token)
    if cached is not None:
        return cached

    base_url = os.getenv("METRICS_BASE_URL", DEFAULT_METRICS_BASE_URL)
    url = f"{base_url.rstrip('/')}/token/{token}/metrics"
    session = await get_session()
    try:
        async with session.get(url, timeout=5) as resp:
            resp.raise_for_status()
            data = await resp.json()
    except Exception as exc:  # pragma: no cover - network errors
        logger.warning("Failed to fetch metrics for %s: %s", token, exc)
        metrics = {
            "mean": 0.0,
            "volatility": 0.02,
            "volume": 0.0,
            "liquidity": 0.0,
            "slippage": 0.0,
            "depth": 0.0,
            "price_history": [],
            "liquidity_history": [],
            "depth_history": [],
            "slippage_history": [],
            "volume_history": [],
            "token_age": 0.0,
            "initial_liquidity": 0.0,
            "tx_count_history": [],
        }
    else:
        metrics = {
            "mean": float(data.get("mean_return", 0.0)),
            "volatility": float(data.get("volatility", 0.02)),
            "volume": float(data.get("volume_24h", 0.0)),
            "liquidity": float(data.get("liquidity", 0.0)),
            "slippage": float(data.get("slippage", 0.0)),
            "depth": float(data.get("depth", 0.0)),
            "price_history": data.get("price_history", []),
            "liquidity_history": data.get("liquidity_history", []),
            "depth_history": data.get("depth_history", []),
            "slippage_history": data.get("slippage_history", []),
            "volume_history": data.get("volume_history", []),
            "token_age": float(data.get("token_age", 0.0)),
            "initial_liquidity": float(data.get("initial_liquidity", 0.0)),
            "tx_count_history": data.get("tx_count_history", []),
        }

    dex_urls = [u.strip() for u in os.getenv("DEX_METRIC_URLS", "").split(",") if u.strip()]
    depth_vals = []
    slip_vals = []
    session = await get_session()
    for base in dex_urls:
        d_url = f"{base.rstrip('/')}/v1/depth?token={token}"
        try:
            async with session.get(d_url, timeout=5) as resp:
                resp.raise_for_status()
                val = (await resp.json()).get("depth")
            if isinstance(val, (int, float)):
                depth_vals.append(float(val))
        except Exception as exc:  # pragma: no cover - network errors
            logger.warning("Failed to fetch depth from %s: %s", base, exc)
        s_url = f"{base.rstrip('/')}/v1/slippage?token={token}"
        try:
            async with session.get(s_url, timeout=5) as resp:
                resp.raise_for_status()
                val = (await resp.json()).get("slippage")
            if isinstance(val, (int, float)):
                slip_vals.append(float(val))
        except Exception as exc:  # pragma: no cover - network errors
            logger.warning("Failed to fetch slippage from %s: %s", base, exc)

    if depth_vals:
        metrics["depth_per_dex"] = depth_vals
        metrics["depth"] = float(sum(depth_vals) / len(depth_vals))
    else:
        metrics["depth_per_dex"] = []
    if slip_vals:
        metrics["slippage_per_dex"] = slip_vals
        metrics["slippage"] = float(sum(slip_vals) / len(slip_vals))
    else:
        metrics["slippage_per_dex"] = []

    TOKEN_METRICS_CACHE.set(token, metrics)
    return metrics


async def async_fetch_token_metrics(token: str) -> dict:
    """Deprecated compatibility wrapper for :func:`fetch_token_metrics_async`."""
    return await fetch_token_metrics_async(token)


def fetch_token_metrics(token: str) -> dict:
    """Synchronous wrapper for :func:`fetch_token_metrics_async`."""
    return asyncio.run(fetch_token_metrics_async(token))


def predict_price_movement(
    token: str,
    *,
    min_volume: float = 0.0,
    recent_volume: float | None = None,
    recent_slippage: float | None = None,
    sentiment: float | None = None,
    order_book_strength: float | None = None,
    model_path: str | None = None,
) -> float:
    """Predict short term price change using ML models when available."""

    model = get_price_model(model_path)
    if model:
        metrics = fetch_token_metrics(token)
        ph = metrics.get("price_history") or []
        lh = metrics.get("liquidity_history") or []
        dh = metrics.get("depth_history") or []
        sh = metrics.get("slippage_history") or []
        vh = metrics.get("volume_history") or []
        th = metrics.get("tx_count_history") or []
        n = min(len(ph), len(lh), len(dh), len(sh or ph), len(vh or ph), len(th or ph))
        if n >= 30:
            seq = np.column_stack(
                [
                    ph[-30:],
                    lh[-30:],
                    dh[-30:],
                    (sh or [0] * n)[-30:],
                    (vh or [0] * n)[-30:],
                    (th or [0] * n)[-30:],
                ]
            )
            try:
                return float(model.predict(seq))
            except Exception:
                pass

    sims = run_simulations(
        token,
        count=1,
        days=1,
        min_volume=min_volume,
        recent_volume=recent_volume,
        recent_slippage=recent_slippage,
        sentiment=sentiment,
        order_book_strength=order_book_strength,
    )
    return sims[0].expected_roi if sims else 0.0


def run_simulations(
    token: str,
    count: int = 1000,
    days: int = 30,
    *,
    min_volume: float = 0.0,
    recent_volume: float | None = None,
    recent_slippage: float | None = None,
    gas_cost: float = 0.0,
    sentiment: float | None = None,
    order_book_strength: float | None = None,
) -> List[SimulationResult]:
    """Run ROI simulations using a simple regression-based model."""

    metrics = fetch_token_metrics(token)
    bias = bias_correction()
    depth_features = metrics.get("depth_per_dex", [])
    slip_features = metrics.get("slippage_per_dex", [])

    depth_features = metrics.get("depth_per_dex", []) or []
    slip_features = metrics.get("slippage_per_dex", []) or []

    results: List[SimulationResult] = []

    dex_metrics = onchain_metrics.fetch_dex_metrics(token)
    for key in ("volume", "liquidity", "depth"):
        val = dex_metrics.get(key)
        if isinstance(val, (int, float)):
            metrics[key] = float(val)

    depth_change = 0.0
    tx_rate = 0.0
    whale_activity = 0.0

    rpc_url = os.getenv("SOLANA_RPC_URL")
    if rpc_url:
        try:
            metrics["liquidity"] = onchain_metrics.fetch_liquidity_onchain(
                token, rpc_url
            )
            metrics["volume"] = onchain_metrics.fetch_volume_onchain(token, rpc_url)
            metrics["slippage"] = onchain_metrics.fetch_slippage_onchain(
                token, rpc_url
            )
            insights = onchain_metrics.collect_onchain_insights(token, rpc_url)
            depth_change = insights.get("depth_change", 0.0)
            tx_rate = insights.get("tx_rate", 0.0)
            whale_activity = insights.get("whale_activity", 0.0)
        except Exception as exc:  # pragma: no cover - unexpected errors
            logger.warning("On-chain metric fetch failed: %s", exc)

    
    depth_features = metrics.get("depth_per_dex", [])
    slip_features = metrics.get("slippage_per_dex", [])
    if metrics.get("volume", 0.0) < min_volume:
        return []

    mu = metrics.get("mean", 0.0)
    sigma = metrics.get("volatility", 0.02)
    base_volume = metrics.get("volume", 0.0)
    volume = base_volume
    volume_spike = 1.0
    if recent_volume is not None:
        volume = float(recent_volume)
        volume_spike = volume / base_volume if base_volume else 1.0

    liquidity = metrics.get("liquidity", 0.0)
    slippage = metrics.get("slippage", 0.0)
    if recent_slippage is not None:
        slippage = float(recent_slippage)

    token_age = float(metrics.get("token_age", 0.0))
    tx_hist = metrics.get("tx_count_history") or []
    tx_trend = float(tx_hist[-1] - tx_hist[-2]) if len(tx_hist) >= 2 else 0.0
    tx_trend_hist = list(np.diff(tx_hist)) if len(tx_hist) >= 2 else []

    initial_liquidity = metrics.get("initial_liquidity")
    if initial_liquidity is None:
        liq_hist_default = metrics.get("liquidity_history") or []
        initial_liquidity = float(liq_hist_default[0]) if liq_hist_default else float(liquidity)
    else:
        initial_liquidity = float(initial_liquidity)

    depth_features = metrics.get("depth_per_dex", [])[:2]
    slip_features = metrics.get("slippage_per_dex", [])[:2]




    depth = metrics.get("depth", 0.0)


    sentiment_val = float(sentiment) if sentiment is not None else float(
        metrics.get("sentiment", 0.0)
    )
    order_strength = float(order_book_strength) if order_book_strength is not None else float(
        metrics.get("order_book_strength", 0.0)
    )


    results: List[SimulationResult] = []

    price_hist = metrics.get("price_history")
    liq_hist = metrics.get("liquidity_history")
    depth_hist = metrics.get("depth_history")
    slip_hist = metrics.get("slippage_history")

    predicted_mean = mu
    used_ml = False

    model = get_price_model()
    if model and price_hist and liq_hist and depth_hist and tx_hist:
        try:
            seq = np.column_stack(
                [price_hist, liq_hist, depth_hist, tx_hist]
            )[-30:]
            predicted_mean = float(model.predict(seq))
            used_ml = True
        except Exception as exc:  # pragma: no cover - model errors
            logger.warning("Failed to load ML model: %s", exc)

    results: List[SimulationResult] = []

    if (
        not used_ml
        and price_hist
        and liq_hist
        and depth_hist
        and slip_hist
        and tx_hist
        and len(price_hist) >= 2
        and len(liq_hist) >= 2
        and len(depth_hist) >= 2
        and len(slip_hist) >= 2
        and len(tx_hist) >= 2
    ):
        try:
            returns = np.diff(price_hist) / price_hist[:-1]
            n = min(
                len(returns),
                len(liq_hist) - 1,
                len(depth_hist) - 1,
                len(slip_hist) - 1,
                len(tx_hist) - 1,
            )
            cols = [liq_hist[:n], depth_hist[:n], slip_hist[:n], tx_trend_hist[:n]]
            for val in depth_features:
                cols.append(np.full(n, val))
            for val in slip_features:
                cols.append(np.full(n, val))
            cols.append(np.full(n, token_age))
            cols.append(np.full(n, initial_liquidity))
            X = np.column_stack(cols)
            if XGBRegressor is not None:
                model = XGBRegressor(
                    n_estimators=50,
                    learning_rate=0.1,
                    max_depth=3,
                    objective="reg:squarederror",
                )
            else:
                model = RandomForestRegressor(n_estimators=50, random_state=42)
            model = model.fit(X, returns[:n])
            feat = (
                [liquidity, depth, slippage, tx_trend]
                + depth_features
                + slip_features
                + [token_age, initial_liquidity]
            )
            predicted_mean = float(model.predict([feat])[0])
        except Exception as exc:  # pragma: no cover - numeric issues
            logger.warning("ROI model training failed: %s", exc)
    elif (
        not used_ml
        and price_hist
        and liq_hist
        and depth_hist
        and slip_hist
        and len(price_hist) >= 2
        and len(liq_hist) >= 2
        and len(depth_hist) >= 2
        and len(slip_hist) >= 2
    ):
        try:
            returns = np.diff(price_hist) / price_hist[:-1]
            n = min(
                len(returns), len(liq_hist) - 1, len(depth_hist) - 1, len(slip_hist) - 1
            )
            cols = [liq_hist[:n], depth_hist[:n], slip_hist[:n]]
            for val in depth_features:
                cols.append(np.full(n, val))
            for val in slip_features:
                cols.append(np.full(n, val))
            X = np.column_stack(cols)
            model = GradientBoostingRegressor().fit(X, returns[:n])
            feat = [liquidity, depth, slippage] + depth_features + slip_features
            predicted_mean = float(model.predict([feat])[0])
        except Exception as exc:  # pragma: no cover - numeric issues
            logger.warning("ROI model training failed: %s", exc)
    elif (
        not used_ml
        and price_hist
        and liq_hist
        and depth_hist
        and len(price_hist) >= 2
        and len(liq_hist) >= 2
        and len(depth_hist) >= 2
    ):
        try:
            returns = np.diff(price_hist) / price_hist[:-1]
            n = min(len(returns), len(liq_hist) - 1, len(depth_hist) - 1)
            cols = [liq_hist[:n], depth_hist[:n], np.full(n, sigma)]
            for val in depth_features:
                cols.append(np.full(n, val))
            for val in slip_features:
                cols.append(np.full(n, val))
            X = np.column_stack(cols)
            model = LinearRegression().fit(X, returns[:n])
            feat = [liquidity, depth, sigma] + depth_features + slip_features
            predicted_mean = float(model.predict([feat])[0])
        except Exception as exc:  # pragma: no cover - numeric issues
            logger.warning("ROI model training failed: %s", exc)

    # Apply bias adjustments after model prediction
    predicted_mean += bias.get("mean", 0.0)
    sigma = max(0.0, sigma + bias.get("volatility", 0.0))





    for _ in range(count):
        daily_returns = np.random.normal(predicted_mean, sigma, days)
        roi = float(np.prod(1 + daily_returns) - 1)
        roi -= gas_cost
        success_prob = float(np.mean(daily_returns > 0))

        results.append(
            SimulationResult(

                success_prob=success_prob,
                expected_roi=roi,
                volume=volume,
                liquidity=liquidity,
                slippage=slippage,
                volatility=sigma,
                volume_spike=volume_spike,
                sentiment=sentiment_val,
                order_book_strength=order_strength,
                token_age=token_age,
                initial_liquidity=initial_liquidity,
                tx_trend=tx_trend,
                depth_change=depth_change,
                tx_rate=tx_rate,
                whale_activity=whale_activity,

            )
        )


    return results
