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
orca_api_url: https://api.orca.so
raydium_api_url: https://api.raydium.io
orca_ws_url: ""
raydium_ws_url: ""
orca_dex_url: https://dex.orca.so
raydium_dex_url: https://dex.raydium.io
metrics_base_url: https://api.example.com
risk_tolerance: 0.1
max_allocation: 0.2
max_risk_per_token: 0.05
stop_loss: 0.1
take_profit: 0.2
trailing_stop: 0.1
max_drawdown: 0.5
volatility_factor: 1.0
risk_multiplier: 1.0
arbitrage_threshold: 0.05
arbitrage_amount: 1.0
learning_rate: 0.1
epsilon: 0.1
discount: 0.95
agents:
  - simulation
  - conviction
  - arbitrage
  - exit
agent_weights:
  simulation: 1.0
  conviction: 1.0
  arbitrage: 1.0
dynamic_weights: true
weight_step: 0.05
```

   An example configuration file named `config.example.toml` is included in
   the project root. Copy it to `config.toml` (or `config.yaml`) and edit the
   values as needed. A high risk preset called `config.highrisk.toml` is also
   provided. The example configuration loads several built‑in **agents** that
   replace the previous static strategy modules.

## Rust Depth Service

The `depth_service` crate provides low‑latency order book snapshots and
direct transaction submission to the Solana RPC. Start the service with

```bash
cargo run --manifest-path depth_service/Cargo.toml -- --serum wss://serum/ws --raydium wss://raydium/ws
```

It writes depth data to `/tmp/depth_service.mmap` and exposes an IPC socket at
`/tmp/depth_service.sock` used by the Python modules.

   The `AgentManager` loads the agents listed under `agents` and applies any
   weights defined in the `agent_weights` table.  When `dynamic_weights` is set
   to `true` a `SwarmCoordinator` derives weights dynamically from each
   agent's historical ROI. The coordinator normalizes ROI values into a
   confidence score and feeds these weights to the `AgentSwarm` at run time.
   To control how much each agent influences trades manually, add an
   `agent_weights` table mapping agent names to weights:

   ```toml
   [agent_weights]
   "simulation" = 1.0
   "conviction" = 1.0
   "arbitrage" = 1.0
   "exit" = 1.0
   ```

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
   endpoint. You can also override URLs for individual venues:
   ```bash
   export DEX_BASE_URL=https://dex.example/api
   export DEX_TESTNET_URL=https://dex.testnet/api
   export ORCA_DEX_URL=https://dex.orca.so
   export RAYDIUM_DEX_URL=https://dex.raydium.io
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

## Agents

The trading logic is implemented by a swarm of small agents:

- **DiscoveryAgent** — finds new token listings using the existing scanners.
- **SimulationAgent** — runs Monte Carlo simulations per token.
- **ConvictionAgent** — rates tokens based on expected ROI.
- **MetaConvictionAgent** — aggregates multiple conviction signals.
- **ArbitrageAgent** — detects DEX price discrepancies.
- **ExitAgent** — proposes sells when stop-loss, take-profit or trailing stop thresholds are hit.
- **ExecutionAgent** — rate‑limited order executor.
- **MemoryAgent** — records past trades for analysis. Trade context and emotion
  tags are saved to `memory.db` and a FAISS index (`trade.index`) for semantic
  search.
- **Adaptive memory** — each agent receives feedback on the outcome of its last
  proposal. The swarm logs success metrics to the advanced memory module and
  agents can query this history to refine future simulations.
- **EmotionAgent** — assigns emotion tags such as "confident" or "anxious" after each trade based on conviction delta, regret level and simulation misfires.
- **ReinforcementAgent** — learns from trade history using Q-learning.
- **DQNAgent** — deep Q-network that learns optimal trade actions.
- **PPOAgent** — actor-critic model trained on offline order book history.
- **RamanujanAgent** — proposes deterministic buys or sells from a hashed conviction score.
- **PortfolioAgent** — maintains per-token allocation using `max_allocation` and buys small amounts when idle with `buy_risk`.

Agents can be enabled or disabled in the configuration and their impact
controlled via the `agent_weights` table.  When dynamic weighting is enabled,
the `AgentManager` updates these weights automatically over time based on each
agent's trading performance.

The `AgentManager` periodically adjusts these weights using the
`update_weights()` method.  It reviews trade history recorded by the
`MemoryAgent` and slightly increases the weight of agents with a positive ROI
while decreasing the weight of those with losses.
Each trade outcome is also logged to the advanced memory. Agents look up
previous success rates when deciding whether to accept new simulation results.

Emotion tags produced by the `EmotionAgent` are stored alongside each trade.
Reinforcement agents can read these tags to temper their proposals. A streak of
negative emotions like `anxious` or `regret` reduces conviction in later
iterations, while positive emotions encourage larger allocations.


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
By default the `AgentManager` uses the agents listed in `config.toml`.  You can
edit that file to enable or disable agents or adjust their weights.  Then run
the bot with:
```bash
python -m solhunter_zero.main --config config.toml
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
also saves a FAISS index named `trade.index` next to the database.  The bot can
run against a DEX testnet or in dry‑run mode where orders are not submitted:
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

Recent updates embed **Chart.js** to visualise trading activity.  The UI now
plots ROI over time, recent trade counts and current agent weights.  You can
adjust risk parameters and agent weights directly in the browser and the values
are sent back to the server via the `/risk` and `/weights` endpoints.

The UI also exposes simple SQL helpers.  Use `/memory/insert` and
`/memory/update` to execute parameterised statements and `/memory/query` to
fetch rows from the `memory.db` database.

Example inserting and reading trades:

```bash
curl -X POST http://localhost:5000/memory/insert \
  -H 'Content-Type: application/json' \
  -d '{"sql":"INSERT INTO trades(token,direction,amount,price) VALUES(:t,:d,:a,:p)","params":{"t":"SOL","d":"buy","a":1.0,"p":20.0}}'

curl -X POST http://localhost:5000/memory/query \
  -H 'Content-Type: application/json' \
  -d '{"sql":"SELECT token, direction, amount, price FROM trades"}'
```

## Additional Metrics

Recent updates introduce new real-time metrics used by the simulator and risk
model:

- **Order-book depth change** — short term change in available liquidity on the
  DEX order book.
- **Mempool transaction rate** — approximate number of transactions per second
  seen in the mempool for each token.
- **Whale wallet activity** — share of liquidity held by very large accounts.

These metrics are gathered automatically by the on-chain scanners and fed into
`run_simulations`.  `RiskManager.adjusted()` now factors them directly into the
scaling of `risk_tolerance` and allocation limits.  High transaction rates or
volume spikes increase the scale, while large depth changes or heavy whale
concentration reduce it.  This helps the bot back off during potential dumps and
scale in aggressively when on-chain activity surges.

The helper function `predict_price_movement()` exposes this regression step
directly. `ConvictionAgent` mixes the predicted short‑term return with the
Monte Carlo average to adjust sniper timing. A strong positive prediction will
trigger buys sooner while a negative one delays them.

## Minimum Portfolio Value

`RiskManager` scales down risk when the value of the entire portfolio falls
below `min_portfolio_value` (default `$20`). The same threshold is passed to
`calculate_order_size` so order sizing never assumes less than this value. This
avoids placing trades that would not cover network fees once the balance is
very small.

## Architecture Details

- **Agent execution and roles** — `AgentManager` launches all agents
  concurrently with `asyncio.gather` via `AgentSwarm` and merges their trade
  proposals. Typical agents handle discovery, simulation, conviction scoring,
  arbitrage checks, exits, execution, logging and reinforcement learning.
- **Persistence** — trade history and order book snapshots are stored in
  `offline_data.db` for offline training. PPO models are periodically retrained
  on this dataset and saved to disk for inference. `MemoryAgent` also records
  trades in `memory.db` for ROI tracking.
- **Advanced forecasting** — transformer and LSTM models can be trained on the
  collected snapshots using `scripts/train_transformer_model.py`,
  `scripts/train_price_model.py` or the offline
  `scripts/train_transformer_agent.py`. Set `PRICE_MODEL_PATH` to the resulting
  model file and `predict_price_movement()` will load it automatically.
- **Scheduling loop** — trading iterations run in a time-driven loop using
  `asyncio` with a default delay of 60&nbsp;s. The optional Flask Web UI runs
  this loop in a dedicated thread while the web server handles requests.
- **Weight updates and ROI** — `SwarmCoordinator` computes ROI for each agent
  using the `MemoryAgent` logs and normalizes these values to produce
  per‑agent confidence scores supplied to `AgentSwarm` during evaluation.
- **Token discovery fallback** — the default `websocket` discovery mode uses
  BirdEye when `BIRDEYE_API_KEY` is set and automatically falls back to
  on-chain scanning if the key is missing.
- **Web UI polling** — the browser polls `/positions`, `/trades`, `/roi`,
  `/risk` and `/weights` every 5&nbsp;s. It assumes a single user and exposes
  JSON endpoints to inspect trades and ROI history.
- **Alerts and position sizing** — no Telegram or other alerting is built in.
  `RiskManager.adjusted()` factors whale liquidity share, mempool transaction
  rate and `min_portfolio_value` into position sizing.

## Backtesting and Datasets

The repository includes a simple backtesting framework. Tick level depth
data can be exported from `offline_data.db` using `scripts/build_tick_dataset.py`:

```bash
python scripts/build_tick_dataset.py --db offline_data.db --out datasets/tick_history.json
```

Offline snapshots can also be used to train a transformer-based price model:

```bash
python scripts/train_transformer_agent.py --db sqlite:///offline_data.db --out models/price.pt
```

Set the `PRICE_MODEL_PATH` environment variable to this file so agents and
`predict_price_movement()` can load it automatically.

`solhunter_zero.backtest_cli` now supports Bayesian optimisation of agent
weights. Optimisation runs the backtester repeatedly while a Gaussian process
searches the weight space:

```bash
python -m solhunter_zero.backtest_cli prices.json -c config.toml --optimize --iterations 30
```

The best weight configuration found is printed as JSON.
