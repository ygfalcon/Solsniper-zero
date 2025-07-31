import asyncio
import types
import sys
import importlib.util

import pytest
import importlib.machinery

# Stub heavy optional dependencies
if importlib.util.find_spec("transformers") is None:
    trans_mod = types.ModuleType("transformers")
    trans_mod.__spec__ = importlib.machinery.ModuleSpec("transformers", None)
    sys.modules.setdefault("transformers", trans_mod)
if importlib.util.find_spec("sentence_transformers") is None:
    st_mod = types.ModuleType("sentence_transformers")
    st_mod.__spec__ = importlib.machinery.ModuleSpec("sentence_transformers", None)
    sys.modules.setdefault("sentence_transformers", st_mod)
if importlib.util.find_spec("faiss") is None:
    faiss_mod = types.ModuleType("faiss")
    faiss_mod.__spec__ = importlib.machinery.ModuleSpec("faiss", None)
    sys.modules.setdefault("faiss", faiss_mod)
if importlib.util.find_spec("torch") is None:
    tmod = types.ModuleType("torch")
    tmod.__spec__ = importlib.machinery.ModuleSpec("torch", None)
    sys.modules.setdefault("torch", tmod)
    nn_mod = types.ModuleType("torch.nn")
    nn_mod.__spec__ = importlib.machinery.ModuleSpec("torch.nn", None)
    sys.modules.setdefault("torch.nn", nn_mod)
    opt_mod = types.ModuleType("torch.optim")
    opt_mod.__spec__ = importlib.machinery.ModuleSpec("torch.optim", None)
    sys.modules.setdefault("torch.optim", opt_mod)
if importlib.util.find_spec("pytorch_lightning") is None:
    pl = types.ModuleType("pytorch_lightning")
    pl.__spec__ = importlib.machinery.ModuleSpec("pytorch_lightning", None)
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
