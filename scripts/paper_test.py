from __future__ import annotations

"""Run the paper trading smoke test.

The script invokes :func:`paper.run` with the ``--test`` flag, which fetches a
small slice of live market data and exercises the live trading path in dry-run
mode.  Reports are written to the specified directory (``reports/`` by default)
and the script exits with a non-zero status on failure.
"""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:  # pragma: no cover - simple path fix
    sys.path.insert(0, str(ROOT))

import paper


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run paper trading smoke test")
    parser.add_argument(
        "--reports",
        type=Path,
        default=Path("reports"),
        help="Directory to write reports",
    )
    args = parser.parse_args(argv)

    try:
        paper.run(["--test", "--reports", str(args.reports)])
    except Exception as exc:  # pragma: no cover - exercised in tests
        print(f"paper test failed: {exc}", file=sys.stderr)
        return 1

    if not (args.reports / "trade_history.json").exists():
        print("paper test failed: trade_history.json not produced", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

