import json

import pytest

from solhunter_zero import investor_demo


@pytest.mark.timeout(60)
def test_investor_demo_correlations(tmp_path):
    """Run investor_demo in staging mode and verify saved reports."""
    reports = tmp_path / "reports"
    investor_demo.main(["--reports", str(reports)])

    corr_path = reports / "correlations.json"
    hedge_path = reports / "hedged_weights.json"
    highlights_path = reports / "highlights.json"

    assert corr_path.exists(), "correlations.json missing"
    assert hedge_path.exists(), "hedged_weights.json missing"
    assert highlights_path.exists(), "highlights.json missing"

    correlations = json.loads(corr_path.read_text())
    hedged = json.loads(hedge_path.read_text())
    highlights = json.loads(highlights_path.read_text())

    # Verify full correlation matrix and hedged weights
    assert correlations["('buy_hold', 'momentum')"] == pytest.approx(0.9153898166961075)
    assert correlations["('buy_hold', 'mean_reversion')"] == pytest.approx(
        -0.8498685240167787
    )
    assert correlations["('momentum', 'mean_reversion')"] == pytest.approx(
        -0.5658094402299682
    )

    assert hedged["buy_hold"] == pytest.approx(0.08461018330389247)
    assert hedged["momentum"] == pytest.approx(0.9153898166961075)

    # Highlights should surface the same deterministic values
    key_corr = highlights.get("key_correlations")
    hedged_hl = highlights.get("hedged_weights")
    assert key_corr is not None and hedged_hl is not None
    assert key_corr["buy_hold-momentum"] == pytest.approx(0.9153898166961075)
    assert key_corr["buy_hold-mean_reversion"] == pytest.approx(-0.8498685240167787)
    assert key_corr["momentum-mean_reversion"] == pytest.approx(-0.5658094402299682)
    assert hedged_hl["buy_hold"] == pytest.approx(0.08461018330389247)
    assert hedged_hl["momentum"] == pytest.approx(0.9153898166961075)
