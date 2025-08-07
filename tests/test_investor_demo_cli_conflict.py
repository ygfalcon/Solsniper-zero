import pytest

from solhunter_zero import investor_demo


def test_data_and_preset_conflict(tmp_path):
    data_file = tmp_path / "prices.json"
    data_file.write_text("[]")
    with pytest.raises(ValueError) as exc:
        investor_demo.main(["--data", str(data_file), "--preset", "short"])
    assert "Cannot specify both --data and --preset" in str(exc.value)
