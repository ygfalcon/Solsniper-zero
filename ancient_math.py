"""Decoding utilities for mysterious sequences.

This module demonstrates how to decode integer sequences found in
``datasets/alien_artifacts.json``. Each artifact entry contains an
``artifact_id`` and a ``sequence`` of integers representing ASCII codes.
"""

from __future__ import annotations

import json
import os
from typing import Dict, List


DATASET_PATH = os.path.join(os.path.dirname(__file__), "datasets", "alien_artifacts.json")


def load_artifacts(path: str = DATASET_PATH) -> List[Dict[str, object]]:
    """Load alien artifact sequences from the dataset."""
    with open(path, "r", encoding="utf-8") as fh:
        lines = [line for line in fh if not line.lstrip().startswith("#")]
    return json.loads("".join(lines))


def decode_sequence(sequence: List[int]) -> str:
    """Decode a list of ASCII codes into a string."""
    chars = [chr(n) for n in sequence if 0 <= n < 128]
    return "".join(chars)


if __name__ == "__main__":
    for artifact in load_artifacts():
        text = decode_sequence(artifact["sequence"])
        print(f"Artifact {artifact['artifact_id']}: {text}")
