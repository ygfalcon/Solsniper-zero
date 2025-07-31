import argparse
import asyncio

from solhunter_zero.offline_data import OfflineData
from solhunter_zero.models.regime_model import train_regime_model, save_regime_model


async def _load(data: OfflineData, token: str | None):
    snaps = await data.list_snapshots(token)
    return snaps


def main() -> None:
    p = argparse.ArgumentParser(description="Train regime classification model")
    p.add_argument("--db", default="offline_data.db", help="Offline data database")
    p.add_argument("--out", default="models/regime.pt", help="Output model path")
    p.add_argument("--token", help="Optional token filter")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--seq-len", type=int, default=30)
    args = p.parse_args()

    data = OfflineData(f"sqlite:///{args.db}")
    snaps = asyncio.run(_load(data, args.token))
    model = train_regime_model(snaps, seq_len=args.seq_len, epochs=args.epochs)
    save_regime_model(model, args.out)
    print(f"Model saved to {args.out}")


if __name__ == "__main__":  # pragma: no cover
    main()
