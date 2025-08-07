# Quick Start

 - A default keypair (`keypairs/default.json`) and configuration (`config.toml`) are bundled for immediate runs.
 - To customize, copy `config/default.toml` to `config.toml` and edit the values.
 - `make start` runs `scripts/startup.py` for guided setup and launches `depth_service` automatically.
 - Use `solhunter-start --auto` for a fully automated launch. The `start.py`, `run.sh`, and `start.command` wrappers forward to this CLI, which ensures the `solhunter-wallet` CLI is present, auto-selects the sole keypair and active config, verifies RPC endpoints, and warns if the wallet balance is below `min_portfolio_value`.
- On macOS, run `scripts/mac_setup.py` to install the Xcode command line tools if needed. The script exits after starting the installation; rerun it once the tools are installed before continuing.
- Set `AUTO_SELECT_KEYPAIR=1` to have the Web UI pick the single keypair automatically.
- Launch the Web UI with `python -m solhunter_zero.ui`.
- Toggle **Full Auto Mode** in the UI to start trading with the active config.
- Or start everything at once with `python scripts/start_all.py` (includes `depth_service`).
- Programmatic consumers can call `solhunter_zero.bootstrap.bootstrap()` to
  perform the same setup steps before interacting with the library.

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

## Troubleshooting Preflight Checks

- **RPC unreachable** — ensure `SOLANA_RPC_URL` points to a healthy endpoint and that your network allows outbound requests.
- **Wallet balance too low** — fund the default keypair or lower `min_portfolio_value` in `config.toml`.
