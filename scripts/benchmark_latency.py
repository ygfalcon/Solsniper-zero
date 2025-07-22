import argparse
import asyncio
import base64
import time

from solhunter_zero.depth_client import stream_depth, submit_signed_tx


async def _run(token: str, tx_b64: str, updates: int) -> None:
    count = 0
    async for _ in stream_depth(token, max_updates=updates):
        start = time.perf_counter()
        await submit_signed_tx(tx_b64)
        end = time.perf_counter()
        print(f"latency: {end - start:.3f}s")
        count += 1
        if count >= updates:
            break


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark transaction latency")
    parser.add_argument("token", help="Token symbol")
    parser.add_argument("tx", help="Base64 transaction")
    parser.add_argument("--updates", type=int, default=1)
    args = parser.parse_args()

    asyncio.run(_run(args.token, args.tx, args.updates))


if __name__ == "__main__":  # pragma: no cover
    main()
