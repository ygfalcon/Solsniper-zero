import logging
import time
from .scanner import scan_tokens
from .simulation import run_simulations
from .decision import should_buy
from .memory import Memory
from .portfolio import Portfolio

logging.basicConfig(level=logging.INFO)


def main():
    memory = Memory()
    portfolio = Portfolio()

    while True:
        tokens = scan_tokens()
        for token in tokens:
            sims = run_simulations(token, count=100)
            if should_buy(sims):
                logging.info("Buying %s", token)
                memory.log_trade(token=token, direction='buy', amount=1, price=0)
                portfolio.add(token, 1, 0)
        time.sleep(60)


if __name__ == "__main__":
    main()
