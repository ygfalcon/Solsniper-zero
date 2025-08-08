# Usage

## Autopilot
Automatically selects the only keypair and active configuration, then launches all services and the trading loop:

```bash
python scripts/start_all.py autopilot
```
Use the `--testnet` flag to submit orders to a testnet DEX endpoint, `--dry-run` to skip order submission entirely, `--offline` to avoid network requests and use a static token list, or `--token-list <file>` to load token addresses from a file. Use `--strategy-rotation-interval N` with one or more `--weight-config` files to automatically test and switch weight presets every `N` iterations.

## Investor Demo

Run a small rolling backtest and generate lightweight reports:

```bash
python scripts/investor_demo.py --reports reports
```

This writes `summary.json`, `trade_history.csv` and `highlights.json` to the
specified reports directory and prints brief snippets to the console.

Enable a lightweight reinforcement‑learning stub:

```bash
python scripts/investor_demo.py --rl-demo --reports reports
```

Run a canned learning loop that rotates strategy weights:

```bash
python scripts/investor_demo.py --learn --reports reports
```

Exercise the full system with heavier dependencies:

```bash
python scripts/investor_demo.py --full-system --reports reports
```

All modes emit the same report files along with console summaries.

### One-Click Investor Demo

1. **Download the repo**

   ```bash
   git clone https://github.com/your-org/Solsniper-zero.git
   cd Solsniper-zero
   ```

2. **Launch the demo**

   On macOS, double-click `one_click_trading_demo.command`. On Linux or Windows run it from the terminal:

   ```bash
   ./one_click_trading_demo.command
   ```

   The script verifies Python 3.11+ and a handful of lightweight dependencies. Missing packages are installed automatically; if installation fails, a warning highlights what to install manually.

3. **Review the reports**

   After the demo finishes, inspect the generated files in `reports`:

   ```text
   reports/summary.json
   reports/trade_history.csv
   ```

   `summary.json` captures overall metrics, while `trade_history.csv` lists each trade. These can be imported into spreadsheets or dashboards for further analysis.

## MEV Bundles

When `use_mev_bundles` is enabled (the default), swaps are submitted
through the [Jito block-engine](https://jito.network/). The same
credentials can also be used to subscribe to Jito's searcher websocket
for real‑time pending transactions. The default `config.toml` leaves
`jito_auth` blank, so supply your token via the `JITO_AUTH` environment
variable or your own configuration file. Provide the block-engine and
websocket endpoints and authentication token:

```bash
export JITO_RPC_URL=https://mainnet.block-engine.jito.wtf/api/v1/bundles
export JITO_AUTH=your_token
export JITO_WS_URL=wss://mainnet.block-engine.jito.wtf/api/v1/ws
export JITO_WS_AUTH=your_token
```

The sniper and sandwich agents automatically pass these credentials to
`MEVExecutor` and will read pending transactions from the Jito stream
when both variables are set. A warning is logged if either variable is
missing while MEV bundles are enabled.

