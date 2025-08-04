import json

from solhunter_zero import investor_demo


def test_investor_demo(tmp_path):
    investor_demo.main(
        [
            "--data",
            "tests/data/prices.json",
            "--reports",
            str(tmp_path),
            "--capital",
            "100",
        ]
    )

    summary_json = tmp_path / "summary.json"
    assert summary_json.exists(), "Summary JSON not generated"
    content = json.loads(summary_json.read_text())
    assert content and "roi" in content[0]

