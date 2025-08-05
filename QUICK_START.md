# Quick Start

- make start runs scripts/startup.py for guided setup.
- Run `cp config.example.toml config.toml` and edit the values.
- Set `AUTO_SELECT_KEYPAIR=1` to always use a single keypair.
- Execute `./run.sh --auto` to start trading automatically.
- Launch the Web UI with `python -m solhunter_zero.ui`.
- Toggle **Full Auto Mode** in the UI to start trading with the active config.
- Or start everything at once with `python scripts/start_all.py`.

## Investor Demo

Run a small rolling backtest and produce reports:

```bash
python scripts/investor_demo.py --reports reports
```

This writes `summary.json`, `trade_history.csv` and `highlights.json` to the
specified directory and prints brief snippets to the console.

Enable a lightweight reinforcement‑learning stub:

```bash
python scripts/investor_demo.py --rl-demo --reports reports
```

Exercise the full system with heavier dependencies:

```bash
python scripts/investor_demo.py --full-system --reports reports
```

All modes emit the same report files and console snippets. The `reports/`
directory is ignored by Git so these generated files remain local.
