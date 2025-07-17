import json
import os
from pathlib import Path
from typing import Optional

from solders.keypair import Keypair


def load_keypair(path: str | Path) -> Keypair:
    """Load a Solana ``Keypair`` from a JSON file.

    The file must contain an array of integers as produced by
    ``solana-keygen``.
    """
    data = json.loads(Path(path).read_text())
    if not isinstance(data, list) or not all(isinstance(i, int) for i in data):
        raise ValueError("Keypair file must be an array of integers")
    return Keypair.from_bytes(bytes(data))


def load_keypair_from_env(path: Optional[str] = None) -> Optional[Keypair]:
    """Return a ``Keypair`` from ``path`` or the ``KEYPAIR_PATH`` env var."""
    final = path or os.getenv("KEYPAIR_PATH")
    if not final:
        return None
    return load_keypair(final)
