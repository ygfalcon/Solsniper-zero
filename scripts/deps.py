#!/usr/bin/env python3
"""Dependency checking utilities for SolHunter Zero."""

from __future__ import annotations

import json
import pkgutil
import re
from pathlib import Path
import sys

try:
    import tomllib  # Python 3.11+
except ModuleNotFoundError:  # pragma: no cover - should not happen
    import tomli as tomllib  # type: ignore

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))
from solhunter_zero.paths import ROOT

OPTIONAL_DEPS = [
    "faiss",
    "sentence_transformers",
    "torch",
    "orjson",
    "lz4",
    "zstandard",
    "msgpack",
]


def check_deps() -> tuple[list[str], list[str]]:
    """Return lists of missing required and optional modules."""
    with open(ROOT / "pyproject.toml", "rb") as fh:
        data = tomllib.load(fh)
    deps = data.get("project", {}).get("dependencies", [])
    missing_required: list[str] = []
    for dep in deps:
        mod = re.split("[<=>]", dep)[0].replace("-", "_")
        if pkgutil.find_loader(mod) is None:
            missing_required.append(mod)
    missing_optional = [m for m in OPTIONAL_DEPS if pkgutil.find_loader(m) is None]
    return missing_required, missing_optional


def main(argv: list[str] | None = None) -> int:
    req, opt = check_deps()
    print(json.dumps({"required": req, "optional": opt}))
    return 1 if req or opt else 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
