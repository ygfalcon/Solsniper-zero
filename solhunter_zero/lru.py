from typing import Any, Hashable
from cachetools import LRUCache as _LRUCache, TTLCache as _TTLCache

# Re-export :class:`cachetools.LRUCache` for external modules.
LRUCache = _LRUCache


class TTLCache(_TTLCache):
    """TTL cache implemented using :class:`cachetools.TTLCache`."""

    def set(self, key: Hashable, value: Any) -> None:
        """Set ``key`` to ``value`` in the cache."""
        self[key] = value

