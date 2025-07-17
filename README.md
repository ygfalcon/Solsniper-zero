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

3. **Set the BirdEye API key**
   The bot relies on the `BIRDEYE_API_KEY` environment variable. Export it before
   running:
   ```bash
   export BIRDEYE_API_KEY=<your-api-key>
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

The scanner uses the Birdeye API for token discovery. Set the `BIRDEYE_API_KEY`
environment variable with your API key so requests are authenticated:

```bash
export BIRDEYE_API_KEY=your_key_here
```
