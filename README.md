# SolHunter Zero

SolHunter Zero is an autonomous AI-driven trading bot for the Solana blockchain. 
See [QUICK_START.md](QUICK_START.md) for a condensed setup guide.

This project is targeted towards being the greatest Solana bot ever created and built by a mad genius known only as zero

## Primary User Preferences

The default workflow is intentionally simple:

1. Send SOL to the desired wallet.
2. Load the keypair in the SolHunter GUI.
3. Press **Start**.

All optional agents are enabled by default and wallet selection is always manual. Offline data (around two to three days of history, capped at 50&nbsp;GB by default) downloads automatically. Set `OFFLINE_DATA_LIMIT_GB` to adjust the size limit. The bot begins with an initial $20 balance linked to [`min_portfolio_value`](#minimum-portfolio-value).
Control how often snapshots and trades are flushed to disk with `OFFLINE_BATCH_SIZE` and `OFFLINE_FLUSH_INTERVAL`.
Trade logs use the same mechanism via `MEMORY_BATCH_SIZE` and `MEMORY_FLUSH_INTERVAL`.

**Suggested hardware upgrades for future expansion**

- Increase RAM to 32 GB or more.
- Use an external SSD for larger datasets.
- Consider a workstation-grade GPU for model training.

### High Risk Preset
For an aggressive starting point, load the `config.highrisk.toml` preset. It
turns on all built-in agents with dynamic weighting so risky trades are scaled
up automatically. Select this file from the UI or set
`SOLHUNTER_CONFIG=config.highrisk.toml` before running the bot.

## Quick Start

1. **Install Python 3.11+**
   Ensure that Python 3.11 or newer is installed. Tools like
   [pyenv](https://github.com/pyenv/pyenv) or your system package manager can help
   with installation.

2. **Install dependencies**
   ```bash
   pip install .[uvloop]
   ```
   This installs all required Python packages as defined in
   `pyproject.toml`, including
   [PyYAML](https://pyyaml.org/) and
   [solders](https://pypi.org/project/solders/) which are necessary when using
   YAML configuration files and Solana keypair functionality.
   The dependency [watchfiles](https://pypi.org/project/watchfiles/) is
   also installed and is used by the order book utilities to watch the
   depth mmap for changes.

   Heavy packages like `numpy`, `aiohttp`, `solana`, `torch` and `faiss`
   install automatically with `pip install .[uvloop]`. Running `./run.sh`
   performs the same installation when dependencies are missing. On Apple
   Silicon machines the script also installs the Metal PyTorch wheel if
   it isn't already present.

   The `uvloop` dependency is optional but recommended for reduced event
   loop latency on Unix-like systems. If available, it is enabled by calling
   `solhunter_zero.util.install_uvloop()` at startup.

   The optional `fastjson` group installs [orjson](https://pypi.org/project/orjson/)
   for faster JSON serialization and parsing. When installed, all HTTP helpers
   and the event bus return JSON as bytes via `orjson`, improving throughput by
   roughly 25%:

  ```bash
  pip install .[fastjson]
  ```

   The optional `fastcompress` group installs [lz4](https://pypi.org/project/lz4/)
   and [zstandard](https://pypi.org/project/zstandard/) for faster event
   compression:

   ```bash
   pip install .[fastcompress]
   ```

For a guided setup you can run `scripts/startup.py` which checks dependencies, prompts for configuration and wallet details, then launches the bot live. You can also simply run `make start`.


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
use_flash_loans: true
max_flash_amount: 0.02
mempool_threshold: 0.0
bundle_size: 1
use_mev_bundles: true
learning_rate: 0.1
dex_priorities: "orca,raydium,jupiter"
dex_fees: "{}"
dex_gas: "{}"
dex_latency: "{}"
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
evolve_interval: 1
mutation_threshold: 0.0
```

Key discovery options:

- `mempool_score_threshold` sets the minimum score for tokens observed in the
  mempool before they are considered by discovery agents.
- `trend_volume_threshold` filters out tokens with on-chain volume below this
  value when ranking new opportunities.
- `max_concurrency` limits how many discovery tasks run in parallel. The
  environment variable `MAX_CONCURRENCY` overrides this value.

   An example configuration file named `config.example.toml` is included in
   the project root. Copy it to `config.toml` (or `config.yaml`) and edit the
   values as needed. A high risk preset called `config.highrisk.toml` is also
   provided. The example configuration loads several built‑in **agents** that
   replace the previous static strategy modules.

## Rust Depth Service

The `depth_service` crate provides low‑latency order book snapshots and
direct transaction submission to the Solana RPC.

1. **Build and run the service**
   ```bash
   cargo run --manifest-path depth_service/Cargo.toml --release -- \
     --config config.toml --serum wss://serum/ws --raydium wss://raydium/ws
   ```
2. **Set environment variables**
   Ensure `DEPTH_SERVICE_SOCKET` and `DEPTH_MMAP_PATH` are exported before
   launching the Python modules. They default to
   `/tmp/depth_service.sock` and `/tmp/depth_service.mmap` respectively.
   The `order_book_ws` module watches this mmap file with
   [watchfiles](https://pypi.org/project/watchfiles/) and automatically
   invalidates its cache on changes.
   Additional variables allow customization:
   - `DEPTH_WS_ADDR` / `DEPTH_WS_PORT` – address and port for the built-in
     websocket server (defaults to `0.0.0.0:8765`).
   - `SOLANA_RPC_URL` and `SOLANA_KEYPAIR` – RPC endpoint and keypair for
     transaction submission.
  - `EVENT_BUS_URL` – optional websocket endpoint of an external event bus.
    When set, depth updates are forwarded using the topic `depth_update`.
    The same value can be provided via `event_bus_url` in your config.
  - `EVENT_BUS_COMPRESSION` – websocket compression algorithm (defaults to
    `deflate`). Set to `none` to disable compression.
  - `COMPRESS_EVENTS` – enable protobuf event compression when `1` (default
    if the `zstandard` package is available). Set to `0` to disable.
  - `EVENT_COMPRESSION` – compression algorithm for protobuf events. Choose
    `zstd`, `lz4`, `zlib` or `none`. When unset and `COMPRESS_EVENTS` is
    enabled, zstd is used if the `zstandard` package is installed, otherwise
    zlib is used. Set `EVENT_COMPRESSION=zlib` or `USE_ZLIB_EVENTS=1` to
    force zlib compression.
  - `EVENT_COMPRESSION_THRESHOLD` – skip compression for events smaller than
    this size in bytes (defaults to `512`).
  - `DEPTH_UPDATE_THRESHOLD` – minimum relative change before broadcasting a
    new snapshot (defaults to `0`).
  - `DEPTH_MIN_SEND_INTERVAL` – minimum interval in milliseconds between
    broadcasts (defaults to `100`).
  - `CPU_LOW_THRESHOLD` / `CPU_HIGH_THRESHOLD` – CPU usage percentages
    controlling delay adjustments (defaults to `20` and `80`).
  - `MAX_CONCURRENCY` – maximum number of concurrent discovery and ranking
    tasks. When set to `0` the scanner uses half the available CPUs.
  - `CPU_USAGE_THRESHOLD` – pause task creation when CPU usage exceeds this
    percentage.
  - `DYNAMIC_CONCURRENCY_INTERVAL` – how often CPU usage is sampled when
    adjusting task limits (defaults to `2`).
  - `CONCURRENCY_KP` – proportional gain for dynamic concurrency adjustments
    (defaults to `0.5`).
  - `system_metrics` events are aggregated from the local monitor and the
    depth service by `metrics_aggregator` which publishes them under
    `system_metrics_combined`.
  - When no metrics arrive for several seconds the scanners fall back to
    `psutil.cpu_percent(interval=None)` to keep scaling active.
  - `DEPTH_FREQ_LOW` / `DEPTH_FREQ_HIGH` – depth update rate thresholds in
    updates per second (defaults to `1` and `10`).
  - `--config <path>` – load these options from the given configuration file.
3. **Route transactions through the service**
   Python code signs transactions locally and forwards them via
   ``depth_client.submit_signed_tx`` or an ``EventExecutor`` from
   ``solhunter_zero.execution``. The service relays them using
   ``send_raw_tx`` for minimal latency.
4. **Faster route search**
  Path computation happens inside the Rust service when
  `use_service_route` is enabled (default). This is roughly ten times faster
  than the Python fallback. A lightweight `route_ffi` library is built
  automatically during installation and copied into the package so no
  manual steps are required. When the `cargo` command is available `run.sh`
  builds the library automatically. If you need to rebuild it manually run:

  ```bash
  cargo build --manifest-path route_ffi/Cargo.toml --release
  ```
  Python automatically loads the compiled library from
  ``solhunter_zero/libroute_ffi.so``. Set `USE_FFI_ROUTE=0` to force the
  Python implementation.

## Flash-Loan Arbitrage

Flash loans are enabled by default. Disable them by setting
`use_flash_loans` to `false` in your configuration. The bot will borrow up
to `max_flash_amount` of the trading
token using supported protocols (e.g. Solend), execute the swap chain and repay
the loan within the same transaction.  You must supply the required protocol
accounts and understand that failed repayment reverts the entire transaction.
The arbitrage path search now factors this flash-loan amount into the expected
profit calculation so routes are ranked based on the borrowed size.

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
7. **Export RL metrics**
    Provide a URL that receives ``rl_metrics`` events:
    ```bash
    export METRICS_URL=http://localhost:9000/metrics
    ```
8. **Configure news feeds for sentiment**
   Sentiment scores influence RL training. Provide comma-separated RSS URLs via `NEWS_FEEDS` and optional social feeds:
   ```bash
   export NEWS_FEEDS=https://news.example/rss
   export TWITTER_FEEDS=https://example.com/twitter.json
   export DISCORD_FEEDS=https://example.com/discord.json
   ```
9. **Provide a keypair for signing**
    Generate a keypair with `solana-keygen new` if you don't already have one and
    point the bot to it using `KEYPAIR_PATH`, `SOLANA_KEYPAIR` or the `--keypair`
    flag:
    ```bash
    export KEYPAIR_PATH=/path/to/your/keypair.json
    ```
    The path can also be supplied via `SOLANA_KEYPAIR` for the Rust depth
    service.  Placing the file in the `keypairs/` directory (for example
    `keypairs/main.json`) lets the bot discover it automatically:
    ```bash
    mkdir -p keypairs
    cp ~/my-keypair.json keypairs/main.json
    ```
    Set `AUTO_SELECT_KEYPAIR=1` so `run.sh` and the Web UI pick the only
    available keypair automatically. When there is just one keypair in the
    `keypairs/` directory it will be selected on start.

    You can also recover a keypair from a BIP‑39 mnemonic using the
    `solhunter-wallet` utility and activate it:
    ```bash
    solhunter-wallet derive mywallet "abandon abandon abandon abandon abandon abandon abandon abandon abandon abandon abandon about" --passphrase ""
    solhunter-wallet select mywallet
    ```
    Placing the resulting file in `keypairs/` and setting `AUTO_SELECT_KEYPAIR=1`
    lets `run.sh` load it automatically.

    To set up a default wallet non-interactively, export `MNEMONIC` (and
    optional `PASSPHRASE`) then run:
    ```bash
    scripts/setup_default_keypair.sh
    ```
10. **Priority RPC endpoints**
    Specify one or more RPC URLs used for high-priority submission:
    ```bash
    export PRIORITY_RPC=https://rpc1.example.com,https://rpc2.example.com
    ```
11. **Priority fee multipliers**
    Configure compute unit price multipliers used when the mempool is busy:
    ```bash
    export PRIORITY_FEES="0,1,2"
    ```
12. **Cache TTL overrides**
    Adjust in-memory cache lifetimes:
    ```bash
    export PRICE_CACHE_TTL=10
    export TOKEN_METRICS_CACHE_TTL=60
    export SIM_MODEL_CACHE_TTL=600
    export TREND_CACHE_TTL=120
    export LISTING_CACHE_TTL=120
    export DEX_METRICS_CACHE_TTL=45
    export TOKEN_VOLUME_CACHE_TTL=45
    export TOP_VOLUME_TOKENS_CACHE_TTL=90
    export DEPTH_CACHE_TTL=1
    export EDGE_CACHE_TTL=60
    ```
13. **HTTP connector limits**
   Tune the aiohttp connector:
   ```bash
   export HTTP_CONNECTOR_LIMIT=0
   export HTTP_CONNECTOR_LIMIT_PER_HOST=0
   ```
14. **Auto-execution**
    Register tokens and pre-signed transactions so the depth service
    dispatches them when thresholds are crossed:
    ```bash
    export AUTO_EXEC='{"TOKEN":{"threshold":1.0,"txs":["BASE64"]}}'
    ```
15. **Run the bot**
   ```bash
   ./run.sh --auto
   ```
16. **External event bus**
   Set `EVENT_BUS_URL` to automatically connect to a remote websocket bus:
   ```bash
   export EVENT_BUS_URL=wss://bus.example.com
   ```
   Alternatively specify `event_bus_url` in the configuration file.
17. **Event compression**
   Choose a compression algorithm for protobuf messages with
   `EVENT_COMPRESSION`:
   ```bash
   export EVENT_COMPRESSION=zstd  # or lz4, zlib, none
   ```
   A typical `depth_update` event (~2.2&nbsp;KB) becomes ~1.8&nbsp;KB with zlib
   (~0.27&nbsp;ms), ~1.9&nbsp;KB with lz4 (~0.004&nbsp;ms) and ~1.7&nbsp;KB with
   zstd (~0.012&nbsp;ms). When the `zstandard` package is installed,
   `COMPRESS_EVENTS=1` selects zstd by default. Set `COMPRESS_EVENTS=0` to
   disable compression or `USE_ZLIB_EVENTS=1` if older nodes expect zlib
   compressed messages.
18. **Full system startup**
   Launch the Rust service, RL daemon and trading loop together:
   ```bash
   python scripts/start_all.py
   ```
   The script waits for the depth websocket and forwards `--config`, `EVENT_BUS_URL` and `SOLANA_RPC_URL` to all subprocesses.

Running `scripts/startup.py` handles these steps interactively and forwards any options to `./run.sh --auto`. The `make start` target is a convenient shortcut.

   This loads the selected configuration (or the `config.highrisk.toml` preset
   when none is chosen). If there is exactly one keypair in `keypairs/`, `run.sh`
   uses it automatically. Set `AUTO_SELECT_KEYPAIR=1` so the Web UI does the
   same.

   You can still run the bot manually with explicit options:
   ```bash
   ./run.sh --dry-run
   # or
   python -m solhunter_zero.main

   ```
19. **Autopilot**
   Automatically selects the only keypair and active configuration,
   then launches all services and the trading loop:
   ```bash
   python scripts/start_all.py autopilot
   ```
Use the `--testnet` flag to submit orders to a testnet DEX endpoint,
`--dry-run` to skip order submission entirely, `--offline` to avoid
network requests and use a static token list, or `--token-list <file>`
to load token addresses from a file.

## MEV Bundles

When `use_mev_bundles` is enabled (the default), swaps are submitted
through the [Jito block-engine](https://jito.network/). The same
credentials can also be used to subscribe to Jito's searcher websocket
for real‑time pending transactions. Provide the block-engine and
websocket endpoints and authentication token:

```bash
export JITO_RPC_URL=https://block-engine.example.com
export JITO_AUTH=your_token
export JITO_WS_URL=wss://searcher.example.com
export JITO_WS_AUTH=your_token
```

The sniper and sandwich agents automatically pass these credentials to
`MEVExecutor` and will read pending transactions from the Jito stream
when both variables are set. A warning is logged if either variable is
missing while MEV bundles are enabled.

## Agents

The trading logic is implemented by a swarm of small agents:

- **DiscoveryAgent** — finds new token listings using the existing scanners.
- **SimulationAgent** — runs Monte Carlo simulations per token.
- **ConvictionAgent** — rates tokens based on expected ROI.
- **MetaConvictionAgent** — aggregates multiple conviction signals.
- **ArbitrageAgent** — detects DEX price discrepancies.
-   The agent polls multiple venues simultaneously and chooses
    the most profitable route accounting for per‑DEX fees, gas and
    latency.  Custom costs can be configured with `dex_fees`,
    `dex_gas` and `dex_latency`.  These latency values are
    measured concurrently at startup by pinging each API and
    websocket endpoint.  Set `MEASURE_DEX_LATENCY=0` to skip
    this automatic measurement.
- **ExitAgent** — proposes sells when stop-loss, take-profit or trailing stop thresholds are hit.
- **ExecutionAgent** — rate‑limited order executor.
  When `PRIORITY_FEES` is set the agent scales the compute-unit price
  based on mempool transaction rate so high-priority submits use a
  larger fee.
- **MempoolSniperAgent** — bundles buys when liquidity appears or a mempool
  event exceeds `mempool_threshold` and submits them with `MEVExecutor`.
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
- **SACAgent** — soft actor-critic with continuous trade sizing.
- **RamanujanAgent** — proposes deterministic buys or sells from a hashed conviction score.
- **StrangeAttractorAgent** — chaotic Lorenz model seeded with order-book depth,
  mempool entropy and conviction velocity. Trades when divergence aligns with a
  known profitable manifold.
- **FractalAgent** — matches ROI fractal patterns using wavelet fingerprints.
- **PortfolioAgent** — maintains per-token allocation using `max_allocation` and buys small amounts when idle with `buy_risk`.
- **PortfolioOptimizer** — adjusts positions using mean-variance analysis and risk metrics.
 - **CrossDEXRebalancer** — distributes trades across venues according to order-book depth, measured latency and per‑venue fees. It asks `PortfolioOptimizer` for base actions,
   splits them between venues with the best liquidity and fastest response, then forwards the resulting
   orders to `ExecutionAgent` (or `MEVExecutor` bundles when enabled) so other
   strategy agents can coordinate around the final execution.

Agents can be enabled or disabled in the configuration and their impact
controlled via the `agent_weights` table.  When dynamic weighting is enabled,
the `AgentManager` updates these weights automatically over time based on each
agent's trading performance.

The `AgentManager` periodically adjusts these weights using the
`update_weights()` method.  It reviews trade history recorded by the
`MemoryAgent` and slightly increases the weight of agents with a positive ROI
while decreasing the weight of those with losses.
Every ``evolve_interval`` iterations the manager also calls ``evolve()`` to
spawn new agent mutations and prune those with an ROI below
``mutation_threshold``.
Each trade outcome is also logged to the advanced memory. Agents look up
previous success rates when deciding whether to accept new simulation results.

Emotion tags produced by the `EmotionAgent` are stored alongside each trade.
Reinforcement agents can read these tags to temper their proposals. A streak of
negative emotions like `anxious` or `regret` reduces conviction in later
iterations, while positive emotions encourage larger allocations.
### Custom Agents via Entry Points

Third-party packages can register their own agent classes under the
`solhunter_zero.agents` entry point group. Any entry points found in this
group are merged into the built-in registry and can be loaded just like the
bundled agents.

```toml
[project.entry-points."solhunter_zero.agents"]
myagent = "mypkg.agents:MyAgent"
```

Example custom agent:

```python
# mypkg/agents.py
from solhunter_zero.agents import BaseAgent


class MyAgent(BaseAgent):
    name = "myagent"

    async def propose_trade(self, token, portfolio, *, depth=None, imbalance=None):
        # implement strategy
        return []
```

## Platform Coordination

Agents and services communicate via a lightweight event bus. Three topics are
published by default:

- `action_executed` whenever the `AgentManager` finishes an order
- `weights_updated` after agent weights change
- `rl_weights` when the RL daemon publishes new weights
- `system_metrics_combined` aggregated CPU and memory usage from
  `metrics_aggregator`
- `risk_updated` when the risk multiplier is modified
- `config_updated` when a configuration file is saved
- `risk_metrics` whenever portfolio risk metrics are recalculated
- `trade_logged` after any trade is written to a memory database

Handlers can subscribe using :func:`subscribe` or the :func:`subscription`
context manager:

```python
from solhunter_zero.event_bus import subscription

async def on_action(event):
    print("executed", event)

with subscription("action_executed", on_action):
    ...  # run trading loop
```

Other processes such as the RL daemon listen to these events to train models and
adjust configuration in real time.

When `replicate_trades` is enabled, `AdvancedMemory` also listens for
`trade_logged` events to mirror trades from other nodes.

When the Web UI is running, these events are also forwarded over a simple
WebSocket endpoint at `ws://localhost:8766/ws`. Clients can subscribe and react
to updates directly in the browser:

```javascript
const ws = new WebSocket('ws://localhost:8766/ws');
ws.onmessage = (ev) => {
  const msg = JSON.parse(ev.data);
  console.log(msg.topic, msg.payload);
};
```



## Requirements
- Python 3.11+
- [Poetry](https://python-poetry.org/) (optional but recommended)
- [PyYAML](https://pyyaml.org/) for YAML configuration files
- [solders](https://pypi.org/project/solders/) for Solana keypair operations
- Optional: `faiss`, `sentence-transformers` and `torch` for advanced features
  like semantic memory search, transformer-based models and deep reinforcement
  learning. These packages are not required for the lightweight tests or
  baseline trading functionality. Install them manually with:
  ```bash
  pip install faiss-cpu sentence-transformers torch pytorch-lightning
  ```

Install Python dependencies:
```bash
pip install .
```

For development you can also install the package in editable mode so changes are
picked up automatically and the test suite has all required tools:
```bash
pip install -e .[dev]
```

### Apple Silicon (ARM64)

Install the ARM64 wheels for PyTorch on Apple M1/M2 machines:

```bash
pip install torch==2.1.0 torchvision==0.16.0 \
  --extra-index-url https://download.pytorch.org/whl/metal
```

Enable the Metal backend for GPU acceleration:

```bash
export PYTORCH_ENABLE_MPS_FALLBACK=1
python - <<'EOF'
import torch
torch.set_default_device('mps')
EOF
```

Heavy features such as reinforcement learning automatically detect the MPS
backend. The RL daemon and :class:`RLTraining` default to `device='mps'` when
available unless you specify another device.

### CUDA GPUs

Install PyTorch with CUDA support and switch the default device:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
python - <<'EOF'
import torch
torch.set_default_device('cuda')
EOF
```

Any training script, including `scripts/online_train_transformer.py`, will then
use the GPU automatically.

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
./run.sh
```
This performs the automatic configuration and keypair selection described above.
Pass any additional CLI options to control the run manually, for example:
```bash
./run.sh --dry-run
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
Add `--profile` to gather performance stats with `cProfile`. The results are
written to `profile.out`:

```bash
python -m solhunter_zero.main --profile --iterations 1 --dry-run --offline
python -m pstats profile.out
```
Set the keypair path with the `--keypair` flag or the `KEYPAIR_PATH` (or
`SOLANA_KEYPAIR`) environment variable if you want to sign orders.

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

The UI provides **Start** and **Stop** buttons to control the trading loop and
includes a **Full Auto Mode** switch. When enabled the UI posts to the
`/autostart` endpoint which launches the bot with the active configuration and
keypair. The dashboard shows the running status, selected keypair and config.
It is served on `http://localhost:5000` by default.
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
- **Mempool ranking** — tokens are scored by volume spikes, liquidity and known
  wallet activity to surface high-potential listings.

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
- **Discovery ranking** — tokens from trending APIs, mempool events and on-chain
  scans are combined, deduplicated and sorted by volume and liquidity.
- **Web UI polling** — the browser polls `/positions`, `/trades`, `/roi`,
  `/risk` and `/weights` every 5&nbsp;s. It assumes a single user and exposes
  JSON endpoints to inspect trades and ROI history.
- **Status endpoint** — `/status` reports if the trading loop, RL daemon,
  depth service and external event bus are alive.
- **Alerts and position sizing** — no Telegram or other alerting is built in.
  `RiskManager.adjusted()` factors whale liquidity share, mempool transaction
  rate and `min_portfolio_value` into position sizing.

## Backtesting and Datasets

The repository includes a simple backtesting framework. Tick level depth
data can be exported from `offline_data.db` using `scripts/build_tick_dataset.py`:

```bash
python scripts/build_tick_dataset.py --db offline_data.db --out datasets/tick_history.json
```

For faster reinforcement learning training you can export the offline tables to
a compressed NumPy archive and memory‑map it:

```bash
python scripts/build_mmap_dataset.py --db offline_data.db --out datasets/offline_data.npz
```

When present, `TradeDataModule` loads this archive instead of querying SQLite.
If it is missing it will be created automatically when `RLTraining` or
`RLDaemon` starts.  On a small dataset this lowered preparation time from around
3&nbsp;s to roughly 0.2&nbsp;s thanks to ``numpy.fromiter`` and memory mapping.

Offline snapshots can also be used to train a transformer-based price model:

```bash
python scripts/train_transformer_agent.py --db sqlite:///offline_data.db --out models/price.pt
```

Set the `PRICE_MODEL_PATH` environment variable to this file so agents and
`predict_price_movement()` can load it automatically.

You can train a soft actor-critic policy from the same dataset:

```bash
python scripts/train_sac_agent.py --db sqlite:///offline_data.db --out sac_model.pt
```

### Continuous Training

Running `online_train_transformer.py` keeps the transformer model up to date by
periodically fitting new snapshots on GPU and saving checkpoints. On a CUDA
machine launch it with daemon mode enabled:

```bash
python scripts/online_train_transformer.py \
  --db sqlite:///offline_data.db \
  --model models/price.pt --device cuda \
  --daemon --log-progress
```

For quick adjustments during live trading use `scripts/live_finetune_transformer.py`.
It reloads the latest checkpoints, performs a few gradient steps and writes the
updated model back to disk so `ConvictionAgent` picks up the new weights.

Set the `PRICE_MODEL_PATH` environment variable to `models/price.pt` so trading
agents reload each checkpoint automatically.

To continuously retrain the RL models on GPU run `scripts/train_rl_gpu.py`:

```bash
python scripts/train_rl_gpu.py --db sqlite:///offline_data.db --model ppo_model.pt --interval 3600
```

Training now also writes a TorchScript version of the checkpoint next to the
regular file (`ppo_model.ptc`). Agents and the RL daemon load this compiled
module automatically when present for faster inference.

You can also launch the built-in RL daemon directly with GPU acceleration:

```bash
./run.sh --daemon --device cuda
```

To forward events to a remote bus use the `--event-bus` option when running
`scripts/run_rl_daemon.py`:

```bash
python scripts/run_rl_daemon.py --event-bus wss://bus.example.com
```

Alternatively start the trainer manually using the dedicated CLI:

```bash
solhunter-train --daemon --device cuda
```

The dataloader now chooses the worker count automatically based on the dataset
size. Set `rl_dynamic_workers = true` so the trainer scales workers with CPU
load. You can still override this using the `RL_NUM_WORKERS` environment
variable or the `--num-workers` flag on `solhunter-train` and
`python -m solhunter_zero.multi_rl`.
On a 4‑core test machine throughput increased from about 250 samples/s with a
single worker to around 700 samples/s with four workers.

When `multi_rl = true` the RL daemon maintains a small population of models
and trains each one on the most recent trades. After every update the model
with the highest score publishes its policy via the `rl_weights` event. The
size of this population is controlled by `rl_population_size`.

Torch 2 adds the `torch.compile` API which can speed up both training and
inference. Models are compiled automatically when this feature is available.
Set `USE_TORCH_COMPILE=0` to disable this optimization.

Set `rl_auto_train = true` in `config.toml` to enable automatic hyperparameter
tuning. When enabled the RL daemon starts automatically with the trading loop.
It spawns `scripts/auto_train_rl.py` which periodically retrains the PPO model
from `offline_data.db`. Control how often tuning runs via `rl_tune_interval`
(seconds).
The RL agents also take the current market regime as an additional input.
Adjust the influence of this indicator with the new `regime_weight` setting
(defaults to `1.0`).

`solhunter_zero.backtest_cli` now supports Bayesian optimisation of agent
weights. Optimisation runs the backtester repeatedly while a Gaussian process
searches the weight space:

```bash
python -m solhunter_zero.backtest_cli prices.json -c config.toml --optimize --iterations 30
```

The best weight configuration found is printed as JSON.

## Troubleshooting

- **Permission denied when connecting to the socket** — check that
  `DEPTH_SERVICE_SOCKET` points to a writable location and that the Rust
  service owns the file. Delete any stale socket file and restart the
  service.
- **Missing keypair** — ensure a valid Solana keypair file is available.
  Set `KEYPAIR_PATH` or `SOLANA_KEYPAIR` to its path or place it in the
  `keypairs/` directory. Use `AUTO_SELECT_KEYPAIR=1` so `run.sh` or the Web UI
  select the sole available key automatically.
- **Service not running** — verify `depth_service` is running and that
  `USE_SERVICE_EXEC`, `USE_RUST_EXEC` and `USE_DEPTH_STREAM` are all set to
  `True`.
- **Slow routing** — the Rust service computes paths much faster. Leave
  `USE_SERVICE_ROUTE` enabled unless debugging the Python fallback.

## Testing

See [TESTING.md](TESTING.md) for instructions on installing dependencies and running the test suite.
