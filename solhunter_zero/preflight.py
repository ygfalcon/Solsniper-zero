from __future__ import annotations

"""Reusable preflight helpers for SolHunter Zero."""

from pathlib import Path
import os
import shutil
import asyncio
import time
from typing import Dict

ROOT = Path(__file__).resolve().parent.parent

__all__ = [
    "ensure_endpoints",
    "check_disk_space",
    "check_internet",
    "ensure_rpc",
]


def ensure_endpoints(cfg: dict) -> None:
    """Ensure HTTP endpoints referenced in ``cfg`` are reachable.

    The configuration may specify several service URLs such as ``DEX_BASE_URL``
    or custom metrics endpoints. This function attempts a ``HEAD`` request to
    each HTTP(S) URL and aborts startup if any service is unreachable. BirdEye
    is only checked when an API key is configured.
    """

    import urllib.error
    from solhunter_zero.http import check_endpoint

    urls: Dict[str, str] = {}
    if cfg.get("birdeye_api_key"):
        urls["BirdEye"] = "https://public-api.birdeye.so/defi/tokenlist"
    for key, val in cfg.items():
        if not isinstance(val, str):
            continue
        if not val.startswith("http://") and not val.startswith("https://"):
            continue
        urls[key] = val

    async def _check(name: str, url: str) -> tuple[str, Exception] | None:
        # Each URL is checked with its own exponential backoff.
        for attempt in range(3):
            try:
                # ``check_endpoint`` is synchronous; run it in a thread to avoid blocking.
                await asyncio.to_thread(check_endpoint, url, retries=1)
                return None
            except urllib.error.URLError as exc:  # pragma: no cover - network failure
                if attempt == 2:
                    return name, exc
                wait = 2 ** attempt
                print(
                    f"Attempt {attempt + 1} failed for {name} at {url}: {exc}. "
                    f"Retrying in {wait} seconds...",
                )
                await asyncio.sleep(wait)

    async def _run() -> list[tuple[str, Exception] | None]:
        tasks = [_check(name, url) for name, url in urls.items()]
        return await asyncio.gather(*tasks)

    results = asyncio.run(_run())
    failures: list[tuple[str, str, Exception]] = []
    for name_exc in results:
        if name_exc is None:
            continue
        name, exc = name_exc
        url = urls[name]
        failures.append((name, url, exc))

    if failures:
        details = "; ".join(
            f"{name} at {url} ({exc})" for name, url, exc in failures
        )
        print(
            "Failed to reach the following endpoints: "
            f"{details}. Check your network connection or configuration."
        )
        raise SystemExit(1)


def check_disk_space(min_bytes: int) -> None:
    """Ensure there is at least ``min_bytes`` free on the current filesystem."""

    try:
        _, _, free = shutil.disk_usage(ROOT)
    except OSError as exc:  # pragma: no cover - unexpected failure
        print(f"Unable to determine free disk space: {exc}")
        raise SystemExit(1)

    if free < min_bytes:
        required_gb = min_bytes / (1024 ** 3)
        free_gb = free / (1024 ** 3)
        print(
            f"Insufficient disk space: {free_gb:.2f} GB available," f" {required_gb:.2f} GB required."
        )
        print("Please free up disk space and try again.")
        raise SystemExit(1)


def check_internet(url: str = "https://example.com") -> None:
    """Ensure basic internet connectivity by reaching a known host."""

    import urllib.request
    import urllib.error

    for attempt in range(3):
        try:
            with urllib.request.urlopen(url, timeout=5) as resp:  # nosec B310
                resp.read()
                return
        except Exception as exc:  # pragma: no cover - network failure
            if attempt == 2:
                print(
                    f"Failed to reach {url} after 3 attempts: {exc}. "
                    "Check your internet connection."
                )
                raise SystemExit(1)
            wait = 2 ** attempt
            print(
                f"Attempt {attempt + 1} failed to reach {url}: {exc}. "
                f"Retrying in {wait} seconds...",
            )
            time.sleep(wait)


def ensure_rpc(*, warn_only: bool = False) -> None:
    """Send a simple JSON-RPC request to ensure the Solana RPC is reachable."""

    rpc_url = os.environ.get("SOLANA_RPC_URL", "https://api.mainnet-beta.solana.com")
    if not os.environ.get("SOLANA_RPC_URL"):
        print(f"Using default RPC URL {rpc_url}")

    import json
    import urllib.request

    payload = json.dumps({"jsonrpc": "2.0", "id": 1, "method": "getHealth"}).encode()
    req = urllib.request.Request(
        rpc_url, data=payload, headers={"Content-Type": "application/json"}
    )
    for attempt in range(3):
        try:
            with urllib.request.urlopen(req, timeout=5) as resp:  # nosec B310
                resp.read()
                break
        except Exception as exc:  # pragma: no cover - network failure
            if attempt == 2:
                msg = (
                    f"Failed to contact Solana RPC at {rpc_url} after 3 attempts: {exc}."
                    " Please ensure the endpoint is reachable or set SOLANA_RPC_URL to a valid RPC."
                )
                if warn_only:
                    print(f"Warning: {msg}")
                    return
                print(msg)
                raise SystemExit(1)
            wait = 2 ** attempt
            print(
                f"Attempt {attempt + 1} failed to contact Solana RPC at {rpc_url}: {exc}."
                f" Retrying in {wait} seconds...",
            )
            time.sleep(wait)
