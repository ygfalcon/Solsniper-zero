# Quick Start

- make start runs scripts/startup.py for guided setup.
- Run `cp config.example.toml config.toml` and edit the values.
- Set `AUTO_SELECT_KEYPAIR=1` to always use a single keypair.
- Execute `./run.sh --auto` to start trading automatically.
- Launch the Web UI with `python -m solhunter_zero.ui`.
- Toggle **Full Auto Mode** in the UI to start trading with the active config.
- Or start everything at once with `python scripts/start_all.py`.
- Run the investor demo to backtest bundled prices. It finishes in a few seconds
  and writes summaries and trade history to the folder given by `--reports`:

  ```bash
  python scripts/investor_demo.py --reports reports
  ```

  To try multiple tokens, supply the sample dataset in
  `tests/data/prices_multitoken.json`:

  ```bash
  python scripts/investor_demo.py --data tests/data/prices_multitoken.json --reports reports
  ```
