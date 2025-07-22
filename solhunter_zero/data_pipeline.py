from __future__ import annotations

import json
from pathlib import Path
from typing import Any, List

from .offline_data import OfflineData


class Snapshot:
    """Simple container for order book snapshot attributes."""

    def __init__(self, price: float, depth: float, imbalance: float, *, slippage: float = 0.0, volume: float = 0.0, tx_rate: float = 0.0) -> None:
        self.price = price
        self.depth = depth
        self.slippage = slippage
        self.volume = volume
        self.imbalance = imbalance
        self.tx_rate = tx_rate


def load_high_freq_snapshots(
    db_url: str = "sqlite:///offline_data.db", dataset_dir: str | Path = "datasets"
) -> List[Any]:
    """Load snapshots from ``offline_data.db`` and JSON datasets."""

    data = OfflineData(db_url)
    snaps = list(data.list_snapshots())

    dir_path = Path(dataset_dir)
    for file in dir_path.glob("*.json"):
        try:
            with open(file, "r", encoding="utf-8") as fh:
                items = json.load(fh)
        except Exception:
            continue
        for it in items:
            snaps.append(
                Snapshot(
                    price=float(it.get("price", 0.0)),
                    depth=float(it.get("depth", 0.0)),
                    imbalance=float(it.get("imbalance", 0.0)),
                )
            )
    return snaps
