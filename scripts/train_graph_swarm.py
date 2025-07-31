import argparse

import numpy as np
import torch
import torch.nn as nn

from solhunter_zero.advanced_memory import AdvancedMemory
from solhunter_zero.graph_swarm import SimpleGAT, build_interaction_graph, save_model


def main() -> None:
    p = argparse.ArgumentParser(description="Train graph-based swarm model")
    p.add_argument("--db", default="memory.db", help="AdvancedMemory database")
    p.add_argument("--out", required=True, help="Output model path")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--hidden-dim", type=int, default=16)
    args = p.parse_args()

    mem = AdvancedMemory(f"sqlite:///{args.db}")
    trades = mem.list_trades()
    agents = sorted({t.reason or "" for t in trades})
    if not agents:
        raise SystemExit("no trades in memory")

    feats, adj, roi = build_interaction_graph(mem, agents)
    if roi.max() != roi.min():
        target = (roi - roi.min()) / (roi.max() - roi.min())
    else:
        target = np.ones_like(roi)
    target = np.exp(target) / np.sum(np.exp(target))

    X = torch.tensor(feats, dtype=torch.float32)
    A = torch.tensor(adj, dtype=torch.float32)
    y = torch.tensor(target, dtype=torch.float32)

    model = SimpleGAT(X.shape[1], hidden_dim=args.hidden_dim)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()
    for _ in range(args.epochs):
        opt.zero_grad()
        pred = model(X, A)
        loss = loss_fn(pred, y)
        loss.backward()
        opt.step()

    save_model(model, args.out)
    print(f"Model saved to {args.out}")


if __name__ == "__main__":  # pragma: no cover
    main()

