from collections import OrderedDict
from typing import Any, Hashable
import time

class LRUCache:
    """Simple LRU cache for arbitrary keys."""

    def __init__(self, maxsize: int = 128) -> None:
        self.maxsize = maxsize
        self._cache: OrderedDict[Hashable, Any] = OrderedDict()

    def get(self, key: Hashable):
        if key not in self._cache:
            return None
        value = self._cache.pop(key)
        self._cache[key] = value
        return value

    def set(self, key: Hashable, value: Any) -> None:
        if key in self._cache:
            self._cache.pop(key)
        elif len(self._cache) >= self.maxsize:
            self._cache.popitem(last=False)
        self._cache[key] = value

    def clear(self) -> None:
        self._cache.clear()

    def __len__(self) -> int:  # pragma: no cover - introspection only
        return len(self._cache)


class TTLCache:
    """LRU cache with expiration time for each entry."""

    def __init__(self, maxsize: int = 128, ttl: float = 30.0) -> None:
        self.maxsize = maxsize
        self.ttl = ttl
        self._cache: OrderedDict[Hashable, tuple[float, Any]] = OrderedDict()

    def get(self, key: Hashable):
        entry = self._cache.get(key)
        if entry is None:
            return None
        ts, value = entry
        if time.time() - ts > self.ttl:
            self._cache.pop(key, None)
            return None
        self._cache.move_to_end(key)
        return value

    def set(self, key: Hashable, value: Any) -> None:
        now = time.time()
        if key in self._cache:
            self._cache.pop(key)
        elif len(self._cache) >= self.maxsize:
            self._cache.popitem(last=False)
        self._cache[key] = (now, value)

    def clear(self) -> None:
        self._cache.clear()

    def __len__(self) -> int:  # pragma: no cover - introspection only
        return len(self._cache)
