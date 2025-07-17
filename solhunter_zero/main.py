"""Main entry point for SolHunter Zero."""

import logging
import time
from typing import List

from .scanner import scan_tokens
from .simulation import run_simulations, simulation_count
from .decision import should_buy
from .memory import Memory
from .portfolio import Portfolio

logging.basicConfig(level=logging.INFO)


def main(loop_delay: float = 60.0) -> None:
    memory = Memory()
    portfolio = Portfolio()

    delay = loop_delay
    while True:
        tokens = scan_tokens()
        logging.info("Scanning complete: %d tokens", len(tokens))

        if tokens:
            delay = max(30.0, delay * 0.8)  # speed up when activity high
        else:
            delay = min(900.0, delay * 1.1)  # slow down when nothing found

        for token in tokens:
            prev_runs = len(memory.recent_simulations(token, limit=100))
            count = simulation_count(prev_runs)
            sims = run_simulations(token, count=count)
            memory.log_simulations(
                token,
                ({"success": s.success_prob, "roi": s.expected_roi} for s in sims),
            )

            if should_buy(sims, token, memory=memory):
                logging.info("Buying %s", token)
                memory.log_trade(token=token, direction="buy", amount=1, price=0.0)
                portfolio.add(token, 1, 0.0)
                # store a simple conviction metric
                avg_success = sum(s.success_prob for s in sims) / len(sims)
                memory.log_conviction(token, avg_success)

        logging.info("Sleeping %.1f seconds", delay)
        time.sleep(delay)


if __name__ == "__main__":
    main()
