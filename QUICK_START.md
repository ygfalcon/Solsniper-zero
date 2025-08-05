# Quick Start

- make start runs scripts/startup.py for guided setup.
- Run `cp config.example.toml config.toml` and edit the values.
- Set `AUTO_SELECT_KEYPAIR=1` to always use a single keypair.
- Execute `./run.sh --auto` to start trading automatically.
- Launch the Web UI with `python -m solhunter_zero.ui`.
- Toggle **Full Auto Mode** in the UI to start trading with the active config.
- Or start everything at once with `python scripts/start_all.py`.
- Run `make demo` to execute the investor demo with the bundled short preset. You can also use the `solhunter-demo` CLI to backtest bundled prices. Both finish in a few seconds and write summaries and trade history to the folder given by `--reports`:

  ```bash
  make demo
  # or
  solhunter-demo --preset short --reports reports
  ```

  The `reports/` directory is ignored by Git so these generated files stay
  local.

  The demo now exercises the real arbitrage, flash loan, sniper and
  DEX-scanning modules using bundled deterministic data so it can run fully
  offline.

  To try multiple tokens, use the bundled `multi` preset or pass the dataset
  path explicitly:

  ```bash
  solhunter-demo --preset multi --reports reports
  # or
  solhunter-demo --data solhunter_zero/data/investor_demo_prices_multi.json --reports reports
  ```

  Append `--full-system` to exercise the reinforcement-learning and full
  arbitrage pipelines:

  ```bash
  solhunter-demo --preset multi --reports reports --full-system
  ```

  Expect extra outputs like `correlations.json`, `hedged_weights.json` and an
  RL reward entry in `highlights.json`. This mode depends on heavier packages
  including `torch`, `pytorch-lightning`, `sqlalchemy` and `psutil`.

  For a lightweight reinforcement-learning demonstration without these
  dependencies, use the RL stub:

  ```bash
  solhunter-demo --preset multi --reports reports --rl-demo
  ```

  The `--rl-demo` flag enables a minimal RL stub that runs in environments
  without the extra packages required by `--full-system`.
