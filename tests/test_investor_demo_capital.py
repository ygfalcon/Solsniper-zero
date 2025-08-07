import json
from pathlib import Path

import pytest

from solhunter_zero import investor_demo


@pytest.mark.timeout(30)
def test_capital_scaling(tmp_path: Path) -> None:
    data_path = Path(__file__).parent / "data" / "prices_short.json"

    reports1 = tmp_path / "run1"
    reports2 = tmp_path / "run2"

    investor_demo.main(
        [
            "--reports",
            str(reports1),
            "--data",
            str(data_path),
        ]
    )
    investor_demo.main(
        [
            "--reports",
            str(reports2),
            "--data",
            str(data_path),
            "--capital",
            "200",
        ]
    )

    summary1 = json.loads((reports1 / "summary.json").read_text())
    summary2 = json.loads((reports2 / "summary.json").read_text())

    caps1 = {entry["config"]: entry["final_capital"] for entry in summary1}
    caps2 = {entry["config"]: entry["final_capital"] for entry in summary2}

    assert caps1.keys() == caps2.keys()
    for name, cap in caps1.items():
        assert caps2[name] == pytest.approx(cap * 2)
