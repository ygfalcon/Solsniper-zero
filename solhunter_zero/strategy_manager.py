import importlib
import asyncio
import os
from typing import Iterable, Any, List, Dict


class StrategyManager:
    """Load and execute multiple trading strategy modules."""

    DEFAULT_STRATEGIES = ["solhunter_zero.sniper", "solhunter_zero.arbitrage"]

    def __init__(
        self,
        strategies: Iterable[str] | None = None,
        *,
        env_var: str = "STRATEGIES",
    ) -> None:
        if strategies is None:
            env = os.getenv(env_var)
            if env:
                strategies = [s.strip() for s in env.split(",") if s.strip()]
        if not strategies:
            strategies = self.DEFAULT_STRATEGIES

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

        return self._merge_actions(actions)

    @staticmethod
    def _merge_actions(actions: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Combine actions on the same token and side by summing amounts."""
        merged: dict[tuple[str, str], Dict[str, Any]] = {}
        for action in actions:
            token = action.get("token")
            side = action.get("side")
            if not token or not side:
                continue
            amt = float(action.get("amount", 0))
            price = float(action.get("price", 0))
            key = (token, side)
            m = merged.setdefault(key, {"token": token, "side": side, "amount": 0.0, "price": 0.0})
            old_amt = m["amount"]
            if old_amt + amt > 0:
                m["price"] = (m["price"] * old_amt + price * amt) / (old_amt + amt)
            m["amount"] += amt
        return list(merged.values())

