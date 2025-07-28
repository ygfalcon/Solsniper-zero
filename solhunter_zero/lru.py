from collections import OrderedDict
from typing import Any, Hashable

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
