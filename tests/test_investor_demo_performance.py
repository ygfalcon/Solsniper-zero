import time
from pathlib import Path

import pytest

from solhunter_zero import investor_demo

MAX_SECONDS = 5


@pytest.mark.timeout(30)
def test_investor_demo_performance(tmp_path):
    """Ensure investor_demo runs quickly on the short dataset."""
    data_path = Path(__file__).resolve().parent / "data" / "prices_short.json"
    reports_dir = tmp_path / "reports"
    argv = ["--data", str(data_path), "--reports", str(reports_dir)]

    start = time.perf_counter()
    investor_demo.main(argv)
    duration = time.perf_counter() - start

    assert (
        duration < MAX_SECONDS
    ), f"Investor demo took {duration:.2f}s"  # pragma: no cover
