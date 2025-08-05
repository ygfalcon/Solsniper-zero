import argparse
import pytest
from solhunter_zero import investor_demo


def test_investor_demo_invalid_preset(tmp_path, monkeypatch):
    def fake_parse_args(self, argv=None):
        return argparse.Namespace(
            data=None,
            preset="nope",
            reports=tmp_path,
            capital=100.0,
            fee=0.0,
            slippage=0.0,
            full_system=False,
            rl_demo=False,
        )

    monkeypatch.setattr(argparse.ArgumentParser, "parse_args", fake_parse_args)

    with pytest.raises(ValueError) as exc:
        investor_demo.main(["--preset", "nope", "--reports", str(tmp_path)])
    assert "Unknown preset" in str(exc.value)
