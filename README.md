# SolHunter Zero

SolHunter Zero is an autonomous AI-driven trading bot for the Solana blockchain. This project is a starting point based on the provided high‑level specifications.

## Requirements
- Python 3.11+
- [Poetry](https://python-poetry.org/) (optional but recommended)

Install Python dependencies:
```bash
pip install -r requirements.txt
```

## Usage
Run the bot with:
```bash
python -m solhunter_zero.main
```

The current implementation is a minimal scaffold that demonstrates the system architecture. Further development is required to implement real trading logic.

## RPC reliability
`scan_tokens()` now detects HTTP 429 responses from the Birdeye API and retries with an exponential backoff. The delay doubles after each 429 up to 60 seconds and resets once a request succeeds.
