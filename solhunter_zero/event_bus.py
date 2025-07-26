import asyncio
import inspect
from collections import defaultdict
from typing import Any, Awaitable, Callable, Dict, List

# mapping of topic -> list of handlers
_subscribers: Dict[str, List[Callable[[Any], Awaitable[None] | None]]] = defaultdict(list)

def subscribe(topic: str, handler: Callable[[Any], Awaitable[None] | None]):
    """Register ``handler`` for ``topic`` events.

    Returns a callable that will remove the handler when invoked.
    """
    _subscribers[topic].append(handler)

    def _unsub() -> None:
        unsubscribe(topic, handler)

    return _unsub

def unsubscribe(topic: str, handler: Callable[[Any], Awaitable[None] | None]):
    """Remove ``handler`` from ``topic`` subscriptions."""
    handlers = _subscribers.get(topic)
    if not handlers:
        return
    try:
        handlers.remove(handler)
    except ValueError:
        return
    if not handlers:
        _subscribers.pop(topic, None)

def publish(topic: str, payload: Any) -> None:
    """Publish ``payload`` to all subscribers of ``topic``."""
    handlers = list(_subscribers.get(topic, []))
    if not handlers:
        return
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None
    for h in handlers:
        if inspect.iscoroutinefunction(h):
            if loop:
                loop.create_task(h(payload))
            else:
                asyncio.run(h(payload))
        else:
            h(payload)

