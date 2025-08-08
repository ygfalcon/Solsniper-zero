# Quick Start

 - A default keypair (`keypairs/default.json`) and configuration (`config.toml`) are bundled for immediate runs.
 - To customize, copy `config/default.toml` to `config.toml` and edit the values.
 - `make start` runs `scripts/startup.py` for guided setup and launches `depth_service` automatically.
 - Use `solhunter-start` to launch the same startup routine with `--one-click` by default while still accepting additional flags.
- Run `./run.sh --auto` (or `python start.py --auto`) for a fully automated launch. The trading engine and Web UI start together. It ensures the `solhunter-wallet` CLI is present, auto-selects the sole keypair and active config, verifies RPC endpoints, and warns if the wallet balance is below `min_portfolio_value`. Use `--no-ui` to disable the UI for headless deployments.
- `scripts/quick_setup.py --auto` populates `config.toml` with defaults. Use `--non-interactive` to rely on environment variables only. Set `AUTO_SELECT_KEYPAIR=1` to have the sole keypair chosen without prompts.
- On macOS, run `scripts/mac_setup.py` to install the Xcode command line tools if needed. The script exits after starting the installation; rerun it once the tools are installed before continuing.
- The Web UI starts automatically; launch it manually with `python -m solhunter_zero.ui` when using `--no-ui`.
- Toggle **Full Auto Mode** in the UI to start trading with the active config.
- Or start everything at once with `python scripts/start_all.py` (includes `depth_service`).
- Programmatic consumers can call `solhunter_zero.bootstrap.bootstrap()` to
  perform the same setup steps before interacting with the library.

## Investor Demo

For an instant teaser covering all strategies, double-click `demo.command` (or
run `python demo.py`). This executes the demo with the full preset and writes
summary files to the default `reports/` directory.

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

## Strategy Showcase Test

Run the default strategies against a fixed price feed:

```bash
pytest tests/staging/test_investor_showcase.py
```

## One-Click Trading Demo

Simulate the investor staging "double click" by running all default strategies
with deterministic inputs:

```bash
pytest tests/test_one_click_trading_all_strategies.py::test_one_click_all_strategies
```

## Troubleshooting Preflight Checks

- **RPC unreachable** — ensure `SOLANA_RPC_URL` points to a healthy endpoint and that your network allows outbound requests.
- **Wallet balance too low** — fund the default keypair or lower `min_portfolio_value` in `config.toml`.
