import argparse
import asyncio

from solhunter_zero.agents.ppo_agent import PPOAgent
from solhunter_zero.agents.memory import MemoryAgent
from solhunter_zero.memory import Memory


async def main() -> None:
    parser = argparse.ArgumentParser(description="Continuous PPO training")
    parser.add_argument("--memory", default="sqlite:///memory.db")
    parser.add_argument("--data", default="sqlite:///offline_data.db")
    parser.add_argument("--model", default="ppo_model.pt")
    parser.add_argument("--interval", type=float, default=60.0)
    args = parser.parse_args()

    mem = Memory(args.memory)
    mem_agent = MemoryAgent(mem)
    agent = PPOAgent(memory_agent=mem_agent, data_url=args.data, model_path=args.model)
    agent.start_online_learning(interval=args.interval)
    await asyncio.Event().wait()


if __name__ == "__main__":  # pragma: no cover
    asyncio.run(main())
