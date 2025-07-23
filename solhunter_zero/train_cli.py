import argparse
import asyncio
from .rl_daemon import RLDaemon

async def main() -> None:
    p = argparse.ArgumentParser(description="Run RL training daemon")
    p.add_argument("--memory", default="sqlite:///memory.db")
    p.add_argument("--data", default="offline_data.db")
    p.add_argument("--model", default="ppo_model.pt")
    p.add_argument("--algo", default="ppo", choices=["ppo", "dqn"])
    p.add_argument("--interval", type=float, default=3600.0)
    p.add_argument("--device", default=None)
    args = p.parse_args()

    daemon = RLDaemon(
        memory_path=args.memory,
        data_path=args.data,
        model_path=args.model,
        algo=args.algo,
        device=args.device,
    )
    daemon.start(args.interval)
    await asyncio.Event().wait()

if __name__ == "__main__":
    asyncio.run(main())
