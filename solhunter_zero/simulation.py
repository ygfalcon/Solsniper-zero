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
from solhunter_zero.lru import TTLCache

# Optional GPU acceleration for simulations
USE_GPU_SIM = os.getenv("USE_GPU_SIM", "0").lower() in {"1", "true", "yes"}
_GPU_BACKEND = None
if USE_GPU_SIM:
    try:  # pragma: no cover - optional dependency
        import torch  # type: ignore
        if torch.cuda.is_available():
            _GPU_BACKEND = "torch"
    except Exception:
        torch = None  # type: ignore
    if _GPU_BACKEND is None:
        try:  # pragma: no cover - optional dependency
            import cupy as cp  # type: ignore
            if cp.cuda.runtime.getDeviceCount() > 0:
                _GPU_BACKEND = "cupy"
        except Exception:
            cp = None  # type: ignore

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
_ACTIVITY_MODEL = None
_ACTIVITY_MTIME = 0.0

# module level cache for token metrics
TOKEN_METRICS_CACHE_TTL = float(os.getenv("TOKEN_METRICS_CACHE_TTL", "30") or 30)
TOKEN_METRICS_CACHE = TTLCache(maxsize=256, ttl=TOKEN_METRICS_CACHE_TTL)

# Cache for trained ROI models keyed by (token, model_type, params)
SIM_MODEL_CACHE_TTL = float(os.getenv("SIM_MODEL_CACHE_TTL", "300") or 300)
SIM_MODEL_CACHE = TTLCache(maxsize=64, ttl=SIM_MODEL_CACHE_TTL)


def invalidate_simulation_models(token: str | None = None) -> None:
    """Remove cached simulation models.

    Parameters
    ----------
    token:
        If provided, only models for the given token are removed. Otherwise the
        entire cache is cleared.
    """

    if token is None:
        SIM_MODEL_CACHE.clear()
        return
    for key in list(SIM_MODEL_CACHE.keys()):
        if isinstance(key, tuple) and key and key[0] == token:
            SIM_MODEL_CACHE.pop(key, None)


def _metrics_signature(metrics: dict) -> tuple:
    """Return a short signature describing metric history state."""

    def _last(val: list) -> float:
        return float(val[-1]) if val else 0.0

    return (
        len(metrics.get("price_history") or []),
        _last(metrics.get("price_history") or []),
        len(metrics.get("liquidity_history") or []),
        _last(metrics.get("liquidity_history") or []),
        len(metrics.get("depth_history") or []),
        _last(metrics.get("depth_history") or []),
        len(metrics.get("slippage_history") or []),
        _last(metrics.get("slippage_history") or []),
        len(metrics.get("tx_count_history") or []),
        _last(metrics.get("tx_count_history") or []),
        float(metrics.get("token_age", 0.0)),
        float(metrics.get("initial_liquidity", 0.0)),
    )


def _model_cache_key(token: str, model_type: str, metrics: dict, days: int) -> tuple:
    """Return hashable key for ``SIM_MODEL_CACHE``."""

    return (token, model_type, days, _metrics_signature(metrics))


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


def get_activity_model(model_path: str | None = None):
    """Return cached activity model reloading when file changes."""
    global _ACTIVITY_MODEL, _ACTIVITY_MTIME
    model_path = model_path or os.getenv("ACTIVITY_MODEL_PATH")
    if not model_path or not os.path.exists(model_path):
        _ACTIVITY_MODEL = None
        _ACTIVITY_MTIME = 0.0
        return None
    try:
        mtime = os.path.getmtime(model_path)
    except OSError:
        return None
    if _ACTIVITY_MODEL is None or mtime > _ACTIVITY_MTIME:
        _ACTIVITY_MODEL = models.get_model(model_path, reload=True)
        _ACTIVITY_MTIME = mtime
    return _ACTIVITY_MODEL


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
    activity_score: float = 0.0



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

    async def _fetch(url: str, key: str, base: str) -> float | None:
        try:
            async with session.get(url, timeout=5) as resp:
                resp.raise_for_status()
                val = (await resp.json()).get(key)
            return float(val) if isinstance(val, (int, float)) else None
        except Exception as exc:  # pragma: no cover - network errors
            logger.warning("Failed to fetch %s from %s: %s", key, base, exc)
            return None

    tasks: list[asyncio.Future] = []
    kinds: list[str] = []

    for base in dex_urls:
        d_url = f"{base.rstrip('/')}/v1/depth?token={token}"
        s_url = f"{base.rstrip('/')}/v1/slippage?token={token}"

        tasks.append(_fetch(d_url, "depth", base))
        kinds.append("depth")
        tasks.append(_fetch(s_url, "slippage", base))
        kinds.append("slippage")

    for kind, result in zip(kinds, await asyncio.gather(*tasks)):
        if result is None:
            continue
        if kind == "depth":
            depth_vals.append(result)
        else:
            slip_vals.append(result)

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
    invalidate_simulation_models(token)
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
        if isinstance(
            model,
            (models.TransformerModel, models.DeepTransformerModel, models.XLTransformerModel),
        ):
            tdh = metrics.get("total_depth_history") or [0.0] * len(ph)
            sh = metrics.get("slippage_history") or [0.0] * len(ph)
            vh = metrics.get("volume_history") or [0.0] * len(ph)
            th = metrics.get("tx_count_history") or [0.0] * len(ph)
            mr = metrics.get("mempool_rate_history") or [0.0] * len(ph)
            ws = metrics.get("whale_share_history") or [0.0] * len(ph)
            sp = metrics.get("spread_history") or [0.0] * len(ph)
            n = min(
                len(ph),
                len(lh),
                len(dh),
                len(tdh),
                len(sh),
                len(vh),
                len(th),
                len(mr),
                len(ws),
                len(sp),
            )
            if n >= 30:
                seq = np.column_stack(
                    [
                        ph[-30:],
                        lh[-30:],
                        dh[-30:],
                        tdh[-30:],
                        sh[-30:],
                        vh[-30:],
                        th[-30:],
                        mr[-30:],
                        ws[-30:],
                        sp[-30:],
                    ]
                )
                try:
                    return float(model.predict(seq))
                except Exception:
                    pass
        else:
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


def predict_token_activity(
    token: str,
    *,
    metrics: dict | None = None,
    model_path: str | None = None,
) -> float:
    """Return activity score using the optional ML model."""

    model = get_activity_model(model_path)
    if model is None:
        return 0.0

    if metrics is None:
        rpc = os.getenv("SOLANA_RPC_URL")
        if not rpc:
            return 0.0
        metrics = onchain_metrics.collect_onchain_insights(token, rpc)

    feat = [
        float(metrics.get("depth_change", 0.0)),
        float(metrics.get("tx_rate", 0.0)),
        float(metrics.get("whale_activity", 0.0)),
        float(metrics.get("avg_swap_size", 0.0)),
    ]

    try:
        return model.predict(feat)
    except Exception:
        return 0.0


async def run_simulations_async(
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
    """Asynchronously run ROI simulations using a simple regression-based model."""

    metrics = await fetch_token_metrics_async(token)
    bias = bias_correction()
    depth_features = metrics.get("depth_per_dex", []) or []
    slip_features = metrics.get("slippage_per_dex", []) or []

    results: List[SimulationResult] = []

    dex_metrics = await onchain_metrics.fetch_dex_metrics_async(token)
    for key in ("volume", "liquidity", "depth"):
        val = dex_metrics.get(key)
        if isinstance(val, (int, float)):
            metrics[key] = float(val)

    depth_change = 0.0
    tx_rate = 0.0
    whale_activity = 0.0
    avg_swap_size = 0.0

    rpc_url = os.getenv("SOLANA_RPC_URL")
    if rpc_url:
        try:
            metrics["liquidity"] = await onchain_metrics.fetch_liquidity_onchain_async(
                token, rpc_url
            )
            metrics["volume"] = await onchain_metrics.fetch_volume_onchain_async(
                token, rpc_url
            )
            metrics["slippage"] = onchain_metrics.fetch_slippage_onchain(
                token, rpc_url
            )
            insights = await onchain_metrics.collect_onchain_insights_async(token, rpc_url)
            depth_change = insights.get("depth_change", 0.0)
            tx_rate = insights.get("tx_rate", 0.0)
            whale_activity = insights.get("whale_activity", 0.0)
            avg_swap_size = insights.get("avg_swap_size", 0.0)
        except Exception as exc:  # pragma: no cover - unexpected errors
            logger.warning("On-chain metric fetch failed: %s", exc)

    depth_features = metrics.get("depth_per_dex", [])
    slip_features = metrics.get("slippage_per_dex", [])
    activity_score = predict_token_activity(
        token,
        metrics={
            "depth_change": depth_change,
            "tx_rate": tx_rate,
            "whale_activity": whale_activity,
            "avg_swap_size": avg_swap_size,
        },
    )
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

    price_hist = metrics.get("price_history")
    liq_hist = metrics.get("liquidity_history")
    depth_hist = metrics.get("depth_history")
    slip_hist = metrics.get("slippage_history")

    predicted_mean = mu
    used_ml = False

    model = get_price_model()
    if model and price_hist and liq_hist and depth_hist and tx_hist:
        try:
            seq = np.column_stack([price_hist, liq_hist, depth_hist, tx_hist])[-30:]
            predicted_mean = float(model.predict(seq))
            used_ml = True
        except Exception as exc:  # pragma: no cover - model errors
            logger.warning("Failed to load ML model: %s", exc)

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
            model_type = "xgb" if XGBRegressor is not None else "rf"
            cache_key = _model_cache_key(token, model_type, metrics, days)
            model = SIM_MODEL_CACHE.get(cache_key)
            if model is None:
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
                SIM_MODEL_CACHE.set(cache_key, model)
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
            cache_key = _model_cache_key(token, "gbr", metrics, days)
            model = SIM_MODEL_CACHE.get(cache_key)
            if model is None:
                model = GradientBoostingRegressor().fit(X, returns[:n])
                SIM_MODEL_CACHE.set(cache_key, model)
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
            cache_key = _model_cache_key(token, "lin", metrics, days)
            model = SIM_MODEL_CACHE.get(cache_key)
            if model is None:
                model = LinearRegression().fit(X, returns[:n])
                SIM_MODEL_CACHE.set(cache_key, model)
            feat = [liquidity, depth, sigma] + depth_features + slip_features
            predicted_mean = float(model.predict([feat])[0])
        except Exception as exc:  # pragma: no cover - numeric issues
            logger.warning("ROI model training failed: %s", exc)

    predicted_mean += bias.get("mean", 0.0)
    sigma = max(0.0, sigma + bias.get("volatility", 0.0))

    if USE_GPU_SIM and _GPU_BACKEND == "torch":
        t_mean = torch.full((count, days), predicted_mean, device="cuda")
        t_std = torch.full((count, days), sigma, device="cuda")
        daily_returns = torch.normal(t_mean, t_std)
        rois = torch.prod(1 + daily_returns, dim=1) - 1
        rois = rois - gas_cost
        success_probs = torch.mean(daily_returns > 0, dim=1)
        success_probs = success_probs.cpu().numpy()
        rois = rois.cpu().numpy()
    elif USE_GPU_SIM and _GPU_BACKEND == "cupy":
        daily_returns = cp.random.normal(predicted_mean, sigma, (count, days))
        rois = cp.prod(1 + daily_returns, axis=1) - 1
        rois = rois - gas_cost
        success_probs = cp.mean(daily_returns > 0, axis=1)
        success_probs = cp.asnumpy(success_probs)
        rois = cp.asnumpy(rois)
    else:
        daily_returns = np.random.normal(predicted_mean, sigma, (count, days))
        rois = np.prod(1 + daily_returns, axis=1) - 1
        rois = rois - gas_cost
        success_probs = np.mean(daily_returns > 0, axis=1)

    for prob, roi in zip(success_probs, rois):
        results.append(
            SimulationResult(
                success_prob=float(prob),
                expected_roi=float(roi),
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
                activity_score=activity_score,
            )
        )

    return results



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
    """Synchronous wrapper for :func:`run_simulations_async`."""

    return asyncio.run(
        run_simulations_async(
            token,
            count=count,
            days=days,
            min_volume=min_volume,
            recent_volume=recent_volume,
            recent_slippage=recent_slippage,
            gas_cost=gas_cost,
            sentiment=sentiment,
            order_book_strength=order_book_strength,
        )
    )
