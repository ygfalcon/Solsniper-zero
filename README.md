# SolHunter Zero

SolHunter Zero is an autonomous AI-driven trading bot for the Solana blockchain. This project is a starting point based on the provided high‑level specifications.

## Quick Start

1. **Install Python 3.11+**
   Ensure that Python 3.11 or newer is installed. Tools like
   [pyenv](https://github.com/pyenv/pyenv) or your system package manager can help
   with installation.

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure API access**
   The scanner can use the public BirdEye service or query the Solana blockchain
   directly. If you want to use BirdEye, export the `BIRDEYE_API_KEY`
   environment variable:
   ```bash
   export BIRDEYE_API_KEY=<your-api-key>
   ```
   To scan the blockchain yourself, provide a Solana RPC endpoint instead:
   ```bash
   export SOLANA_RPC_URL=https://api.mainnet-beta.solana.com
   ```

4. **Run the bot**
   ```bash
   ./run.sh
   # or
   python -m solhunter_zero.main
   ```
   Use the `--testnet` flag to submit orders to a testnet DEX endpoint or
   `--dry-run` to skip order submission entirely.

## Requirements
- Python 3.11+
- [Poetry](https://python-poetry.org/) (optional but recommended)

Install Python dependencies:
```bash
pip install -r requirements.txt
```

For development you can also install the package in editable mode so changes are
picked up automatically:
```bash
pip install -e .
```

## Usage
Run the bot with:
```bash
python -m solhunter_zero.main
```
Or simply use the helper script which installs any missing dependencies:
```bash
./run.sh
```
You can customize the database path and the delay between iterations.  The bot
can also run against a DEX testnet or in dry‑run mode where orders are not
submitted:
```bash
python -m solhunter_zero.main \
  --memory-path sqlite:///my.db --loop-delay 30 \
  --testnet --dry-run
```

The scanner can pull token information from BirdEye or directly from the
blockchain. For BirdEye, set the `BIRDEYE_API_KEY` environment variable so
requests are authenticated:

```bash
export BIRDEYE_API_KEY=your_key_here
```

If you prefer to scan the Solana blockchain without BirdEye, provide a Solana
RPC URL instead:

```bash
export SOLANA_RPC_URL=https://api.mainnet-beta.solana.com
```
