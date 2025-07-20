from __future__ import annotations

import logging
import os
from dataclasses import dataclass
import aiohttp
from typing import List

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor

try:  # pragma: no cover - optional dependency
    from xgboost import XGBRegressor  # type: ignore
except Exception:  # pragma: no cover - when xgboost is missing
    XGBRegressor = None

import numpy as np
import requests
from . import onchain_metrics

logger = logging.getLogger(__name__)

# Default base URL for the metrics API. Can be overridden by the
# ``METRICS_BASE_URL`` environment variable or via a configuration file that
# sets this variable before the module is imported.
DEFAULT_METRICS_BASE_URL = "https://api.example.com"


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



def fetch_token_metrics(token: str) -> dict:
    """Fetch historical return metrics for ``token``.

    The function tries to retrieve data from a remote API. When the call fails
    for any reason, default values are returned.  The metrics include the mean
    daily return and the standard deviation of daily returns.
    """

    base_url = os.getenv("METRICS_BASE_URL", DEFAULT_METRICS_BASE_URL)
    url = f"{base_url.rstrip('/')}/token/{token}/metrics"
    try:
        resp = requests.get(url, timeout=5)
        resp.raise_for_status()
        data = resp.json()
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
            "token_age": float(data.get("token_age", 0.0)),
            "initial_liquidity": float(data.get("initial_liquidity", 0.0)),
            "tx_count_history": data.get("tx_count_history", []),
        }
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
            "token_age": 0.0,
            "initial_liquidity": 0.0,
            "tx_count_history": [],
        }

    dex_urls = [u.strip() for u in os.getenv("DEX_METRIC_URLS", "").split(",") if u.strip()]
    depth_vals = []
    slip_vals = []
    for base in dex_urls:
        d_url = f"{base.rstrip('/')}/v1/depth?token={token}"
        try:
            resp = requests.get(d_url, timeout=5)
            resp.raise_for_status()
            val = resp.json().get("depth")
            if isinstance(val, (int, float)):
                depth_vals.append(float(val))
        except Exception as exc:  # pragma: no cover - network errors
            logger.warning("Failed to fetch depth from %s: %s", base, exc)
        s_url = f"{base.rstrip('/')}/v1/slippage?token={token}"
        try:
            resp = requests.get(s_url, timeout=5)
            resp.raise_for_status()
            val = resp.json().get("slippage")
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

    return metrics


async def async_fetch_token_metrics(token: str) -> dict:
    """Asynchronously fetch token metrics via ``aiohttp``."""

    base_url = os.getenv("METRICS_BASE_URL", DEFAULT_METRICS_BASE_URL)
    url = f"{base_url.rstrip('/')}/token/{token}/metrics"
    async with aiohttp.ClientSession() as session:
        try:
            async with session.get(url, timeout=5) as resp:
                resp.raise_for_status()
                data = await resp.json()
        except Exception as exc:  # pragma: no cover - network errors
            logger.warning("Failed to fetch metrics for %s: %s", token, exc)
            return {
                "mean": 0.0,
                "volatility": 0.02,
                "volume": 0.0,
                "liquidity": 0.0,
                "slippage": 0.0,
                "slippage_history": [],
                "token_age": 0.0,
                "initial_liquidity": 0.0,
                "tx_count_history": [],
            }

    return {
        "mean": float(data.get("mean_return", 0.0)),
        "volatility": float(data.get("volatility", 0.02)),
        "volume": float(data.get("volume_24h", 0.0)),
        "liquidity": float(data.get("liquidity", 0.0)),
        "slippage": float(data.get("slippage", 0.0)),
        "slippage_history": data.get("slippage_history", []),
        "token_age": float(data.get("token_age", 0.0)),
        "initial_liquidity": float(data.get("initial_liquidity", 0.0)),
        "tx_count_history": data.get("tx_count_history", []),
    }


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

    results: List[SimulationResult] = []

    if (
        price_hist
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
        price_hist
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
        price_hist
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

            )
        )


    return results
