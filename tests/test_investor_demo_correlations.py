import json

import pytest

from solhunter_zero import investor_demo


@pytest.mark.timeout(60)
def test_investor_demo_correlations(tmp_path):
    """Run investor_demo and verify correlations and hedged weights."""
    reports = tmp_path / "reports"
    investor_demo.main(["--reports", str(reports)])

    corr_path = reports / "correlations.json"
    hedge_path = reports / "hedged_weights.json"
    assert corr_path.exists(), "correlations.json missing"
    assert hedge_path.exists(), "hedged_weights.json missing"

    correlations = json.loads(corr_path.read_text())
    hedged = json.loads(hedge_path.read_text())

    assert correlations["('buy_hold', 'momentum')"] == pytest.approx(
        0.9153898166961075
    )
    assert correlations["('buy_hold', 'mean_reversion')"] == pytest.approx(
        -0.8498685240167787
    )

    assert hedged["buy_hold"] == pytest.approx(0.08461018330389247)
    assert hedged["momentum"] == pytest.approx(0.9153898166961075)
