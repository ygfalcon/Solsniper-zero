from __future__ import annotations

import time
import urllib.error
import urllib.request


def check_internet(url: str) -> None:
    """Ensure basic internet connectivity by reaching ``url``.

    The function performs a simple ``GET`` request with exponential backoff.
    ``SystemExit`` is raised with a helpful message if all attempts fail.
    """

    for attempt in range(3):
        try:
            with urllib.request.urlopen(url, timeout=5) as resp:  # nosec B310
                resp.read()
            return
        except Exception as exc:  # pragma: no cover - network failure
            if attempt == 2:
                msg = (
                    f"Failed to reach {url} after 3 attempts: {exc}. "
                    "Check your internet connection."
                )
                print(msg)
                raise SystemExit(msg)
            wait = 2 ** attempt
            print(
                f"Attempt {attempt + 1} failed to reach {url}: {exc}. "
                f"Retrying in {wait} seconds..."
            )
            time.sleep(wait)
