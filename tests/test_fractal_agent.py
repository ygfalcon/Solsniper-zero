import asyncio
import numpy as np

from solhunter_zero.agents.fractal_agent import FractalAgent
from solhunter_zero.portfolio import Portfolio


class DummyPortfolio(Portfolio):
    def __init__(self):
        super().__init__(path=None)
        self.balances = {}


def test_fractal_agent_trade(monkeypatch):
    class DummyPywt:
        @staticmethod
        def cwt(data, scales, wavelet):
            return np.ones((len(scales), len(data))), None

    monkeypatch.setattr(
        "solhunter_zero.agents.fractal_agent.pywt",
        DummyPywt,
    )

    monkeypatch.setattr(
        FractalAgent,
        "_roi_history",
        lambda self, token: [1.0, 1.0, 1.0, 1.0],
    )
    monkeypatch.setattr(FractalAgent, "_hurst", lambda self, s: 0.4)

    def fake_past(self, exclude=None):
        return [self._fractal_fingerprint([1.0, 1.0, 1.0, 1.0])]

    monkeypatch.setattr(FractalAgent, "_past_fingerprints", fake_past)

    agent = FractalAgent(similarity_threshold=0.8)
    pf = DummyPortfolio()
    actions = asyncio.run(agent.propose_trade("tok", pf))
    assert actions and actions[0]["side"] == "buy"
