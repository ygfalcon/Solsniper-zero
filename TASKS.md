# Performance and Connectivity Tasks

The following tasks aim to improve platform connectedness and algorithmic efficiency, with a focus on CPU usage and latency.

:::task{title="Unify Event Bus Communication"}
Extend `solhunter_zero.event_bus` to manage a single WebSocket connection per process and share it across modules. This reduces connection overhead and improves cross-module message latency.
:::

:::task{title="Adopt orjson Serialization"}
Replace standard `json` usage in high-frequency modules (e.g. `depth_client`, `dex_ws`, `jito_stream`) with `orjson` when available. This minimizes CPU time spent in serialization and deserialization.
:::

:::task{title="Cache DEX Metrics"}
Implement an in-memory cache for DEX metrics in `dex_scanner` and `depth_client` to avoid redundant network requests and decrease latency during trading decisions.
:::

:::task{title="Vectorize RL Training"}
Profile `rl_training` loops and convert CPU-bound computations to vectorized NumPy/PyTorch operations or GPU execution to lower CPU usage and accelerate training cycles.
:::

:::task{title="Adaptive Loop Scheduling"}
Review all polling loops (e.g. in `main.py`, `websocket_scanner.py`) and adjust to event-driven patterns or adaptive sleep intervals based on activity. This prevents busy waiting and reduces unnecessary CPU load.
:::

These tasks will collectively enhance system connectivity, increase processing speed, and minimize latency.
