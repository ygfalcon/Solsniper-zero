import argparse
import time
import psutil

from solhunter_zero.memory import Memory


def benchmark(count: int, commit_interval=None, batch_size=None) -> tuple[float, float]:
    mem = Memory('sqlite:///:memory:', commit_interval=commit_interval, batch_size=batch_size)
    proc = psutil.Process()
    proc.cpu_percent(None)
    start = time.perf_counter()
    for _ in range(count):
        mem.log_trade(token='TOK', direction='buy', amount=1.0, price=1.0, _broadcast=False)
    mem.close()
    duration = time.perf_counter() - start
    cpu = proc.cpu_percent(None)
    return cpu, count / duration


def main() -> None:
    p = argparse.ArgumentParser(description="Benchmark Memory inserts")
    p.add_argument("--count", type=int, default=1000)
    p.add_argument("--commit-interval", type=float)
    p.add_argument("--batch-size", type=int)
    args = p.parse_args()

    cpu, rate = benchmark(args.count)
    print(f"baseline inserts/s={rate:.0f} cpu={cpu:.1f}%")

    cpu2, rate2 = benchmark(args.count, args.commit_interval, args.batch_size)
    print(f"buffered inserts/s={rate2:.0f} cpu={cpu2:.1f}%")


if __name__ == "__main__":
    main()
