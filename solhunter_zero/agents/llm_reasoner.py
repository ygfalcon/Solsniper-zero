from __future__ import annotations

from typing import Any, Dict, Iterable, List

try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
except ImportError as exc:  # pragma: no cover - optional dependency

    class _Missing:
        def __init__(self, *a, **k) -> None:
            raise ImportError("transformers is required for LLMReasoner")

        @classmethod
        def from_pretrained(cls, *a, **k):  # type: ignore[return-value]
            raise ImportError("transformers is required for LLMReasoner")

    AutoTokenizer = AutoModelForCausalLM = _Missing  # type: ignore

from ..news import compute_sentiment, fetch_headlines_async
from ..portfolio import Portfolio
from . import BaseAgent


class LLMReasoner(BaseAgent):
    """Use a language model to summarize news and predict market bias."""

    name = "llm_reasoner"

    def __init__(
        self,
        llm_model: str = "gpt2",
        llm_context_length: int = 128,
        feeds: Iterable[str] | None = None,
        twitter_feeds: Iterable[str] | None = None,
        discord_feeds: Iterable[str] | None = None,
    ) -> None:
        self.llm_model = llm_model
        self.context_length = int(llm_context_length)
        self.feeds = list(feeds) if feeds else []
        self.twitter_feeds = list(twitter_feeds) if twitter_feeds else []
        self.discord_feeds = list(discord_feeds) if discord_feeds else []
        self._tokenizer: AutoTokenizer | None = None
        self._model: AutoModelForCausalLM | None = None

    # ------------------------------------------------------------------
    def _load_model(self) -> tuple[AutoTokenizer, AutoModelForCausalLM]:
        if self._model is None or self._tokenizer is None:
            self._tokenizer = AutoTokenizer.from_pretrained(self.llm_model)
            self._model = AutoModelForCausalLM.from_pretrained(self.llm_model)
        return self._tokenizer, self._model

    async def _summarize(self) -> str:
        if not (self.feeds or self.twitter_feeds or self.discord_feeds):
            return ""
        headlines = await fetch_headlines_async(
            self.feeds,
            None,
            twitter_urls=self.twitter_feeds,
            discord_urls=self.discord_feeds,
        )
        if not headlines:
            return ""
        text = " ".join(headlines)
        tokenizer, model = self._load_model()
        inputs = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=self.context_length,
        )
        out = model.generate(
            inputs["input_ids"],
            max_length=self.context_length,
            num_beams=2,
            do_sample=False,
        )
        return tokenizer.decode(out[0], skip_special_tokens=True)

    async def propose_trade(
        self,
        token: str,
        portfolio: Portfolio,
        *,
        depth: float | None = None,
        imbalance: float | None = None,
    ) -> List[Dict[str, Any]]:
        summary = await self._summarize()
        bias = compute_sentiment(summary) if summary else 0.0
        side = "buy" if bias >= 0 else "sell"
        amount = abs(bias)
        return [
            {
                "token": token,
                "side": side,
                "amount": amount,
                "price": 0.0,
                "bias": bias,
            }
        ]
