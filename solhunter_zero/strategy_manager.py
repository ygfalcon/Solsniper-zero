import importlib
import asyncio
from typing import Iterable, Any, List, Dict


class StrategyManager:
    """Load and execute multiple trading strategy modules."""

    def __init__(self, strategies: Iterable[str] | None = None) -> None:
        if strategies is None:
            strategies = ["solhunter_zero.sniper", "solhunter_zero.arbitrage"]
        self._modules = []
        for name in strategies:
            try:
                mod = importlib.import_module(name)
            except Exception:  # pragma: no cover - optional strategies
                continue
            if hasattr(mod, "evaluate"):
                self._modules.append(mod)

    async def evaluate(self, token: str, portfolio: Any) -> List[Dict[str, Any]]:
        """Run all strategies on ``token`` and return combined actions."""
        tasks = []
        results: List[Any] = []
        for mod in self._modules:
            func = getattr(mod, "evaluate", None)
            if func is None:
                continue
            if asyncio.iscoroutinefunction(func):
                tasks.append(func(token, portfolio))
            else:
                results.append(func(token, portfolio))
        if tasks:
            results.extend(await asyncio.gather(*tasks))
        actions: List[Dict[str, Any]] = []
        for res in results:
            if not res:
                continue
            if isinstance(res, list):
                actions.extend(res)
            else:
                actions.append(res)
        return actions
