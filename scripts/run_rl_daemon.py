import argparse
import asyncio
import os

import solhunter_zero.device as device
from solhunter_zero.agents.dqn import DQNAgent
from solhunter_zero.agents.memory import MemoryAgent
from solhunter_zero.agents.ppo_agent import PPOAgent
from solhunter_zero.memory import Memory
from solhunter_zero.rl_daemon import RLDaemon


async def main() -> None:
    parser = argparse.ArgumentParser(description="Run RL training daemon")
    parser.add_argument("--memory", default="sqlite:///memory.db")
    parser.add_argument("--data", default="offline_data.db")
    parser.add_argument("--replay", default="sqlite:///replay.db")
    parser.add_argument("--dqn-model", default="dqn_model.pt")
    parser.add_argument("--ppo-model", default="ppo_model.pt")
    parser.add_argument("--interval", type=float, default=3600.0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--event-bus")
    parser.add_argument("--distributed-rl", action="store_true")
    parser.add_argument("--no-hierarchical-rl", action="store_true")
    args = parser.parse_args()

    dev = device.get_default_device(args.device)
    if args.event_bus:
        os.environ["EVENT_BUS_URL"] = args.event_bus

    mem = Memory(args.memory)
    mem_agent = MemoryAgent(mem)

    dqn_agent = DQNAgent(
        memory_agent=mem_agent,
        model_path=args.dqn_model,
        replay_url=args.replay,
        device=dev,
    )
    ppo_agent = PPOAgent(
        memory_agent=mem_agent,
        data_url=f"sqlite:///{args.data}",
        model_path=args.ppo_model,
        device=dev,
    )

    dqn_daemon = RLDaemon(
        memory_path=args.memory,
        data_path=args.data,
        model_path=args.dqn_model,
        algo="dqn",
        device=dev,
        agents=[dqn_agent],
        distributed_rl=args.distributed_rl,
        hierarchical_rl=not args.no_hierarchical_rl,
    )
    ppo_daemon = RLDaemon(
        memory_path=args.memory,
        data_path=args.data,
        model_path=args.ppo_model,
        algo="ppo",
        device=dev,
        agents=[ppo_agent],
        distributed_rl=args.distributed_rl,
        hierarchical_rl=not args.no_hierarchical_rl,
    )

    dqn_daemon.start(args.interval)
    ppo_daemon.start(args.interval)

    await asyncio.Event().wait()


if __name__ == "__main__":  # pragma: no cover
    asyncio.run(main())
