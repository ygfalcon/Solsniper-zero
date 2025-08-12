#!/usr/bin/env python3
"""Dependency checking utilities for SolHunter Zero."""

from __future__ import annotations

import argparse
import pkgutil
import re

try:
    import tomllib  # Python 3.11+
except ModuleNotFoundError:  # pragma: no cover - should not happen
    import tomli as tomllib  # type: ignore

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

# Mapping of distribution names to importable module names when they differ.
# Keys are normalised to lowercase for case-insensitive lookups.
DIST_TO_MODULE = {
    "scikit-learn": "sklearn",
    "pyyaml": "yaml",
    "pillow": "PIL",
    "opencv-python": "cv2",
    "beautifulsoup4": "bs4",
    "python-dateutil": "dateutil",
    "python-dotenv": "dotenv",
    "grpcio": "grpc",
    "psycopg2-binary": "psycopg2",
    "mysqlclient": "MySQLdb",
}


def check_deps() -> tuple[list[str], list[str]]:
    """Return lists of missing required and optional modules."""
    with open(ROOT / "pyproject.toml", "rb") as fh:
        data = tomllib.load(fh)
    deps = data.get("project", {}).get("dependencies", [])
    missing_required: list[str] = []
    for dep in deps:
        dist_name = re.split("[<=>]", dep)[0]
        mod = DIST_TO_MODULE.get(
            dist_name.lower(), dist_name
        ).replace("-", "_")
        if pkgutil.find_loader(mod) is None:
            missing_required.append(mod)
    missing_optional = [
        m for m in OPTIONAL_DEPS if pkgutil.find_loader(m) is None
    ]
    return missing_required, missing_optional


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Install project dependencies"
    )
    parser.add_argument(
        "--install-optional",
        action="store_true",
        help="Install optional dependencies",
    )
    parser.add_argument(
        "--extras",
        nargs="*",
        help="Extras to install from the local package",
    )
    args = parser.parse_args(argv)

    from solhunter_zero.bootstrap_utils import DepsConfig, ensure_deps

    cfg = DepsConfig(
        install_optional=args.install_optional,
        extras=args.extras if args.extras else ("uvloop",),
    )
    ensure_deps(cfg)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
