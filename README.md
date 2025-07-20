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
   This installs all required Python packages, including
   [PyYAML](https://pyyaml.org/) and
   [solders](https://pypi.org/project/solders/) which are necessary when using
   YAML configuration files and Solana keypair functionality.

3. **Create a configuration file**
   Create a `config.yaml` or `config.toml` file in the project directory with
   your API keys, RPC URL and DEX endpoints:

   ```yaml
birdeye_api_key: YOUR_BIRDEYE_KEY
solana_rpc_url: https://api.mainnet-beta.solana.com
dex_base_url: https://dex.example/api
dex_testnet_url: https://dex.testnet/api
metrics_base_url: https://api.example.com
risk_tolerance: 0.1
max_allocation: 0.2
max_risk_per_token: 0.05
trailing_stop: 0.1
max_drawdown: 0.5
volatility_factor: 1.0
risk_multiplier: 1.0
arbitrage_threshold: 0.05
arbitrage_amount: 1.0
```

   An example configuration file named `config.example.toml` is included in
   the project root. Copy it to `config.toml` (or `config.yaml`) and edit the
   values as needed. A high risk preset called `config.highrisk.toml` is also
   provided.

   Environment variables with the same names override values from the file.
   You can specify an alternative file with the `--config` command line option
   or by setting the `SOLHUNTER_CONFIG` environment variable.

   ```bash
   export SOLHUNTER_CONFIG=/path/to/config.yaml
   ```

4. **Configure API access**
   The scanner uses the BirdEye API when `BIRDEYE_API_KEY` is set.  If the key
   is missing, it will fall back to scanning the blockchain directly using the
   RPC endpoint specified by `SOLANA_RPC_URL` and will query on-chain volume
   and liquidity metrics for discovered tokens.
   To use BirdEye, export the API key:

   ```bash
   export BIRDEYE_API_KEY=<your-api-key>
   ```
   If this variable is unset, the bot logs a warning and automatically falls back
   to on-chain scanning.
   To scan the blockchain yourself, provide a Solana RPC endpoint instead:

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
   ./run.sh --auto
   ```
   This automatically loads the selected configuration (or the `config.highrisk.toml`
   preset when none is selected), selects the only available keypair if there is
   just one, and begins trading with the default strategies.

   You can still run the bot manually with:
   ```bash
   ./run.sh
   # or
   python -m solhunter_zero.main

   ```
Use the `--testnet` flag to submit orders to a testnet DEX endpoint,
`--dry-run` to skip order submission entirely, `--offline` to avoid
network requests and use a static token list, or `--token-list <file>`
to load token addresses from a file.


## Requirements
- Python 3.11+
- [Poetry](https://python-poetry.org/) (optional but recommended)
- [PyYAML](https://pyyaml.org/) for YAML configuration files
- [solders](https://pypi.org/project/solders/) for Solana keypair operations

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
./run.sh --auto
```
The `--auto` flag performs the automatic configuration and keypair selection
described above. To run manually without automation use:
```bash
./run.sh
```
You can customize the database path and the delay between iterations.  The bot
can also run against a DEX testnet or in dry‑run mode where orders are not
submitted:
```bash
python -m solhunter_zero.main \
  --memory-path sqlite:///my.db --loop-delay 30 \
  --testnet --dry-run --offline \
  --discovery-method websocket
  --config myconfig.yaml
```
Set the keypair path with the `--keypair` flag or the `KEYPAIR_PATH`
environment variable if you want to sign orders.

Choose how tokens are discovered with `--discovery-method` (or the
`DISCOVERY_METHOD` environment variable). Available methods are `onchain`,
`mempool`, `websocket`, `pools` and `file`.
The default `websocket` mode uses BirdEye when `BIRDEYE_API_KEY` is set and
falls back to on-chain scanning otherwise.
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

For testing or development without network access, pass the `--offline`
flag or provide `--token-list FILE`. The file should contain one token
address per line. In both cases no network requests are made.

## Web UI

Run a simple Flask-based UI with:

```bash
python -m solhunter_zero.ui
```

The UI provides **Start** and **Stop** buttons to control the trading loop. It
is served on `http://localhost:5000` by default.
When launched without a user configuration file or `SOLHUNTER_CONFIG`
environment variable, the UI automatically loads the `config.highrisk.toml`
preset.

## Additional Metrics

Recent updates introduce new real-time metrics used by the simulator and risk
model:

- **Order-book depth change** — short term change in available liquidity on the
  DEX order book.
- **Mempool transaction rate** — approximate number of transactions per second
  seen in the mempool for each token.
- **Whale wallet activity** — share of liquidity held by very large accounts.

These metrics are gathered automatically by the on-chain scanners and fed into
`run_simulations`.  Sudden spikes or drops adjust the `RiskManager` parameters so
the bot reduces exposure during potential dumps and scales in when activity
surges.
