#!/usr/bin/env python3
"""Generate gRPC protobuf classes for Solhunter Zero."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from solhunter_zero.build_utils import ensure_event_proto


if __name__ == "__main__":
    ensure_event_proto()
