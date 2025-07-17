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
You can customize the database path and the delay between iterations:
```bash
python -m solhunter_zero.main --memory-path sqlite:///my.db --loop-delay 30
```

The scanner uses the Birdeye API for token discovery. Set the `BIRDEYE_API_KEY`
environment variable with your API key so requests are authenticated:

```bash
export BIRDEYE_API_KEY=your_key_here
```
