import logging
import time
from argparse import ArgumentParser

from .scanner import scan_tokens
from .simulation import run_simulations
from .decision import should_buy
from .memory import Memory
from .portfolio import Portfolio

logging.basicConfig(level=logging.INFO)


def main(memory_path: str = "sqlite:///memory.db", loop_delay: int = 60) -> None:
    """Run the trading loop.

    Parameters
    ----------
    memory_path:
        Database URL for storing trades.
    loop_delay:
        Delay between iterations in seconds.
    """

    memory = Memory(memory_path)
    portfolio = Portfolio()

    while True:
        tokens = scan_tokens()
        for token in tokens:
            sims = run_simulations(token, count=100)
            if should_buy(sims):
                logging.info("Buying %s", token)
                memory.log_trade(token=token, direction='buy', amount=1, price=0)
                portfolio.add(token, 1, 0)
        time.sleep(loop_delay)


if __name__ == "__main__":
    parser = ArgumentParser(description="Run SolHunter Zero bot")
    parser.add_argument(
        "--memory-path",
        default="sqlite:///memory.db",
        help="Database URL for storing trades",
    )
    parser.add_argument(
        "--loop-delay",
        type=int,
        default=60,
        help="Delay between iterations in seconds",
    )
    args = parser.parse_args()
    main(memory_path=args.memory_path, loop_delay=args.loop_delay)
