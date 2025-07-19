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

3. **Create a configuration file**
   Create a `config.yaml` or `config.toml` file in the project directory with
   your API keys, RPC URL and DEX endpoints:

   ```yaml
   birdeye_api_key: YOUR_BIRDEYE_KEY
   solana_rpc_url: https://api.mainnet-beta.solana.com
   dex_base_url: https://dex.example/api
   dex_testnet_url: https://dex.testnet/api
   metrics_base_url: https://api.example.com
   ```

   Environment variables with the same names override values from the file.

4. **Configure API access**
   The scanner uses the BirdEye API when `BIRDEYE_API_KEY` is set.  If the key
   is missing, it will fall back to scanning the blockchain directly using the
   RPC endpoint specified by `SOLANA_RPC_URL`.
   To use BirdEye, export the API key:

   ```bash
   export BIRDEYE_API_KEY=<your-api-key>
   ```
   If this variable is unset, the bot logs a warning and automatically falls back
   to on-chain scanning.
   To scan the blockchain yourself, provide a Solana RPC endpoint instead:


   Or provide a Solana RPC endpoint for on-chain scanning

   ```bash
   export SOLANA_RPC_URL=https://api.mainnet-beta.solana.com
   ```

5. **Configure DEX endpoints**
   Set the base URL of the DEX API for mainnet and (optionally) the testnet
   endpoint. The defaults are placeholders, so you should provide your own:
   ```bash
   export DEX_BASE_URL=https://dex.example/api
   export DEX_TESTNET_URL=https://dex.testnet/api
   ```
6. **Set the metrics API endpoint**
   Specify the base URL used by the simulator to fetch historical return
   metrics:
   ```bash
   export METRICS_BASE_URL=https://api.example.com
   ```
7. **Provide a keypair for signing**
   Generate a keypair with `solana-keygen new` if you don't already have one and
   point the bot to it using `KEYPAIR_PATH` or the `--keypair` flag:
   ```bash
   export KEYPAIR_PATH=/path/to/your/keypair.json
   ```
8. **Run the bot**
   ```bash
   ./run.sh
   # or
   python -m solhunter_zero.main

   ```
   Use the `--testnet` flag to submit orders to a testnet DEX endpoint,
   `--dry-run` to skip order submission entirely, or `--offline` to avoid
   network requests and use a static token list


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
  --testnet --dry-run --offline
```
Set the keypair path with the `--keypair` flag or the `KEYPAIR_PATH`
environment variable if you want to sign orders.

The scanner can pull token information from BirdEye or directly from the
blockchain. When `BIRDEYE_API_KEY` is set, requests are sent to BirdEye.
If the key is absent, the scanner queries the blockchain using `SOLANA_RPC_URL`.
Set the API key like this:

```bash
export BIRDEYE_API_KEY=your_key_here
```
If the key is not provided, a warning is emitted and on-chain scanning is used
instead.

To scan the Solana blockchain directly, provide a Solana RPC URL instead:

```bash
export SOLANA_RPC_URL=https://api.mainnet-beta.solana.com
```

For testing or development without any network access, pass the `--offline`
flag to the bot. In this mode the scanner returns a small fixed list of tokens
without making HTTP requests.

## Web UI

Run a simple Flask-based UI with:

```bash
python -m solhunter_zero.ui
```

The UI provides **Start** and **Stop** buttons to control the trading loop. It
is served on `http://localhost:5000` by default.
