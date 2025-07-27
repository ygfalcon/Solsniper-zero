import argparse
import asyncio
import os

from solhunter_zero.rl_daemon import RLDaemon
from solhunter_zero.agents.dqn import DQNAgent
from solhunter_zero.agents.ppo_agent import PPOAgent
from solhunter_zero.agents.memory import MemoryAgent
from solhunter_zero.memory import Memory


async def main() -> None:
    parser = argparse.ArgumentParser(description="Run RL training daemon")
    parser.add_argument("--memory", default="sqlite:///memory.db")
    parser.add_argument("--data", default="offline_data.db")
    parser.add_argument("--replay", default="sqlite:///replay.db")
    parser.add_argument("--dqn-model", default="dqn_model.pt")
    parser.add_argument("--ppo-model", default="ppo_model.pt")
    parser.add_argument("--interval", type=float, default=3600.0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--event-bus")
    args = parser.parse_args()
    if args.event_bus:
        os.environ["EVENT_BUS_URL"] = args.event_bus

    mem = Memory(args.memory)
    mem_agent = MemoryAgent(mem)

    dqn_agent = DQNAgent(
        memory_agent=mem_agent,
        model_path=args.dqn_model,
        replay_url=args.replay,
        device=args.device,
    )
    ppo_agent = PPOAgent(
        memory_agent=mem_agent,
        data_url=f"sqlite:///{args.data}",
        model_path=args.ppo_model,
        device=args.device,
    )

    dqn_daemon = RLDaemon(
        memory_path=args.memory,
        data_path=args.data,
        model_path=args.dqn_model,
        algo="dqn",
        device=args.device,
        agents=[dqn_agent],
    )
    ppo_daemon = RLDaemon(
        memory_path=args.memory,
        data_path=args.data,
        model_path=args.ppo_model,
        algo="ppo",
        device=args.device,
        agents=[ppo_agent],
    )

    dqn_daemon.start(args.interval)
    ppo_daemon.start(args.interval)

    await asyncio.Event().wait()


if __name__ == "__main__":  # pragma: no cover
    asyncio.run(main())
