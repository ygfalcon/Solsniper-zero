from __future__ import annotations

import json
from argparse import ArgumentParser
from pathlib import Path
from random import Random


def generate_dataset(tokens: list[str], seed: int, out: str) -> None:
    """Create a dataset of glyph-derived coefficients."""
    rng = Random(seed)
    data = {
        t: {"r": round(rng.uniform(3.5, 4.0), 1), "iterations": rng.randint(10, 20)}
        for t in tokens
    }
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2)


def main(argv: list[str] | None = None) -> int:
    p = ArgumentParser(description="Generate alien cipher coefficient table")
    p.add_argument("--tokens", nargs="+", default=["SOL", "ETH"], help="Token symbols")
    p.add_argument("--seed", type=int, default=0, help="Random seed")
    p.add_argument("--out", default="datasets/alien_cipher.json", help="Output JSON file")
    args = p.parse_args(argv)
    generate_dataset(args.tokens, args.seed, args.out)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
