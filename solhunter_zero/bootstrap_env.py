from __future__ import annotations

"""Project-wide environment bootstrap helper."""

import os
import sys
from pathlib import Path

from . import env

ROOT = Path(__file__).resolve().parent.parent


def init_env() -> None:
    """Initialize runtime environment for entry scripts.

    This adds the project root to ``sys.path``, changes the working directory
    to the project root and loads variables from the project's ``.env`` file.
    """
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    os.chdir(ROOT)
    env.load_env_file(ROOT / ".env")
