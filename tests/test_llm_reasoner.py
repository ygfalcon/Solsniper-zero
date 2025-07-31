import asyncio
import types
import sys
import importlib.util

import pytest

# Stub heavy optional dependencies
if importlib.util.find_spec("transformers") is None:
    sys.modules.setdefault("transformers", types.ModuleType("transformers"))
if importlib.util.find_spec("sentence_transformers") is None:
    sys.modules.setdefault("sentence_transformers", types.ModuleType("sentence_transformers"))
if importlib.util.find_spec("faiss") is None:
    sys.modules.setdefault("faiss", types.ModuleType("faiss"))
if importlib.util.find_spec("torch") is None:
    sys.modules.setdefault("torch", types.ModuleType("torch"))
    sys.modules.setdefault("torch.nn", types.ModuleType("torch.nn"))
    sys.modules.setdefault("torch.optim", types.ModuleType("torch.optim"))
if importlib.util.find_spec("pytorch_lightning") is None:
    pl = types.ModuleType("pytorch_lightning")
    pl.callbacks = types.SimpleNamespace(Callback=object)
    pl.LightningModule = type("LightningModule", (), {})
    pl.LightningDataModule = type("LightningDataModule", (), {})
    pl.Trainer = type("Trainer", (), {"fit": lambda *a, **k: None})
    sys.modules.setdefault("pytorch_lightning", pl)

from solhunter_zero.agent_manager import AgentManager
from solhunter_zero.agents.llm_reasoner import LLMReasoner
from solhunter_zero.agents.execution import ExecutionAgent
from solhunter_zero.portfolio import Portfolio


class DummyPortfolio(Portfolio):
    def __init__(self):
        super().__init__(path=None)
        self.balances = {}


class DummyModel:
    def generate(self, ids, max_length=None, num_beams=2, do_sample=False):
        return ids


class DummyTokenizer:
    def __call__(self, text, return_tensors=None, truncation=True, max_length=None):
        return {"input_ids": [[0]]}

    def decode(self, ids, skip_special_tokens=True):
        return "good news"


def test_bias_output(monkeypatch):
    monkeypatch.setattr(
        "solhunter_zero.agents.llm_reasoner.AutoTokenizer.from_pretrained",
        lambda m: DummyTokenizer(),
    )
    monkeypatch.setattr(
        "solhunter_zero.agents.llm_reasoner.AutoModelForCausalLM.from_pretrained",
        lambda m: DummyModel(),
    )
    async def fake_headlines(*a, **k):
        return ["headline"]

    monkeypatch.setattr(
        "solhunter_zero.agents.llm_reasoner.fetch_headlines_async", fake_headlines
    )
    monkeypatch.setattr(
        "solhunter_zero.agents.llm_reasoner.compute_sentiment", lambda t: 0.8
    )

    agent = LLMReasoner(feeds=["f"])
    mgr = AgentManager([agent], executor=ExecutionAgent(rate_limit=0), memory_agent=None)
    pf = DummyPortfolio()
    actions = asyncio.run(mgr.evaluate("TOK", pf))
    assert actions and actions[0]["bias"] == pytest.approx(0.8)
