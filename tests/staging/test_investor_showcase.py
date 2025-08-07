from solhunter_zero import investor_showcase


def test_investor_showcase_runs():
    actions = investor_showcase.run_showcase()
    assert actions, "no actions produced"
    sides = {a.get("side") for a in actions}
    assert "buy" in sides and "sell" in sides

