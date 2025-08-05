# Quick Start

- make start runs scripts/startup.py for guided setup.
- Run `cp config.example.toml config.toml` and edit the values.
- Set `AUTO_SELECT_KEYPAIR=1` to always use a single keypair.
- Execute `./run.sh --auto` to start trading automatically.
- Launch the Web UI with `python -m solhunter_zero.ui`.
- Toggle **Full Auto Mode** in the UI to start trading with the active config.
- Or start everything at once with `python scripts/start_all.py`.
- Run `make demo` to execute the investor demo with the bundled short preset.
- Run `make demo-full` for the multi-token, full-system demo.

You can also use the `solhunter-demo` CLI to backtest bundled prices. All of
these commands write summaries and trade history to the folder given by
`--reports`:

  ```bash
  make demo
  # or
  solhunter-demo --preset short --reports reports
  ```

  ```bash
  make demo-full
  # or
  solhunter-demo --preset multi --full-system --reports reports
  ```

  To try multiple tokens without the full system, use the bundled `multi`
  preset or pass the dataset path explicitly:

  ```bash
  solhunter-demo --preset multi --reports reports
  # or
  solhunter-demo --data solhunter_zero/data/investor_demo_prices_multi.json --reports reports
  ```

  The full-system demo produces extra outputs like `correlations.json`,
  `hedged_weights.json` and an RL reward entry in `highlights.json`. This mode
  depends on heavier packages including `torch`, `pytorch-lightning`,
  `sqlalchemy` and `psutil`.
