# Agents

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
    this automatic measurement. Ongoing refreshes are controlled by
    `dex_latency_refresh_interval`.
    When the optional `route_ffi`
    library is available the agent uses it automatically for
    path computations. Set `USE_FFI_ROUTE=0` to disable this
    behavior.
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
- **ArtifactMathAgent** — evaluates simple math expressions using `load_artifact_math` to read `solhunter_zero/data/artifact_math.json`.
- **AlienCipherAgent** — logistic-map strategy that reads coefficients via `load_alien_cipher` from `solhunter_zero/data/alien_cipher.json`.
- **PortfolioAgent** — maintains per-token allocation using `max_allocation` and buys small amounts when idle with `buy_risk`.
- **PortfolioOptimizer** — adjusts positions using mean-variance analysis and risk metrics.
- **CrossDEXRebalancer** — distributes trades across venues according to order-book depth, measured latency and per‑venue fees. It asks `PortfolioOptimizer` for base actions,
  splits them between venues with the best liquidity and fastest response, then forwards the resulting
  orders to `ExecutionAgent` (or `MEVExecutor` bundles when enabled) so other
  strategy agents can coordinate around the final execution.
- **CrossDEXArbitrage** — searches for multi-hop swap chains using the `route_ffi`
  path finder. Latency for each venue is measured with `measure_dex_latency_async`
  and combined with `dex_fees`, `dex_gas` and `dex_latency` to select the most
  profitable route. Limit the search depth with `max_hops`.

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
If multiple weight presets are provided via ``weight_config_paths`` the manager
evaluates them every ``strategy_rotation_interval`` iterations and activates the
best performing set.
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
