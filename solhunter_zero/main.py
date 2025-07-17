import logging
import time
from argparse import ArgumentParser

from .scanner import scan_tokens
from .simulation import run_simulations
from .decision import should_buy
from .memory import Memory
from .portfolio import Portfolio
from .exchange import place_order

logging.basicConfig(level=logging.INFO)


def main(
    memory_path: str = "sqlite:///memory.db",
    loop_delay: int = 60,
    *,
    iterations: int | None = None,
    testnet: bool = False,
    dry_run: bool = False,
) -> None:
    """Run the trading loop.

    Parameters
    ----------
    memory_path:
        Database URL for storing trades.
    loop_delay:
        Delay between iterations in seconds.
    iterations:
        Number of iterations to run before exiting. ``None`` runs forever.
    """

    memory = Memory(memory_path)
    portfolio = Portfolio()

    def _run_iteration() -> None:
        tokens = scan_tokens()
        for token in tokens:
            sims = run_simulations(token, count=100)
            if should_buy(sims):
                logging.info("Buying %s", token)
                place_order(
                    token,
                    side="buy",
                    amount=1,
                    price=0,
                    testnet=testnet,
                    dry_run=dry_run,
                )
                if not dry_run:
                    memory.log_trade(token=token, direction="buy", amount=1, price=0)
                    portfolio.add(token, 1, 0)

    if iterations is None:
        while True:
            _run_iteration()
            time.sleep(loop_delay)
    else:
        for i in range(iterations):
            _run_iteration()
            if i < iterations - 1:
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
    parser.add_argument(
        "--iterations",
        type=int,
        default=None,
        help="Number of iterations to run before exiting",
    )
    parser.add_argument(
        "--testnet",
        action="store_true",
        help="Use testnet DEX endpoints",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Do not submit orders, just simulate",
    )
    args = parser.parse_args()
    main(
        memory_path=args.memory_path,
        loop_delay=args.loop_delay,
        iterations=args.iterations,
        testnet=args.testnet,
        dry_run=args.dry_run,
    )
