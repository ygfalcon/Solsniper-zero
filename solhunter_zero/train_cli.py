import argparse
import asyncio
from .rl_daemon import RLDaemon
from .http import close_session
from .util import install_uvloop

install_uvloop()

async def main() -> None:
    p = argparse.ArgumentParser(description="Run RL training daemon")
    p.add_argument("--memory", default="sqlite:///memory.db")
    p.add_argument("--data", default="offline_data.db")
    p.add_argument("--model", default="ppo_model.pt")
    p.add_argument("--algo", default="ppo", choices=["ppo", "dqn"])
    p.add_argument("--interval", type=float, default=3600.0)
    p.add_argument("--num-workers", type=int, default=None, help="Data loader workers")
    p.add_argument("--device", default=None)
    p.add_argument("--daemon", action="store_true", help="Run in background")
    args = p.parse_args()

    if args.num_workers is not None:
        import os
        os.environ["RL_NUM_WORKERS"] = str(args.num_workers)
    daemon = RLDaemon(
        memory_path=args.memory,
        data_path=args.data,
        model_path=args.model,
        algo=args.algo,
        device=args.device,
    )
    daemon.start(args.interval)
    if args.daemon:
        try:
            while True:
                await asyncio.sleep(3600)
        except KeyboardInterrupt:
            pass
    else:
        await asyncio.Event().wait()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    finally:
        asyncio.run(close_session())
