# Full system startup

Launch the Rust service, RL daemon and trading loop together:

```bash
python scripts/start_all.py
```

The script waits for the depth websocket and forwards `--config`, `EVENT_BUS_URL` and `SOLANA_RPC_URL` to all subprocesses.

Running `scripts/startup.py` handles these steps interactively and forwards any options to the cross-platform entry point
`python start.py` (or `python -m solhunter_zero.main --auto`), which performs a fully automated launch using the bundled defaults.

The script ensures the `solhunter-wallet` command-line tool is available, loads the active configuration (falling back to `config/default.toml` when none is chosen) and automatically selects the sole keypair in `keypairs/`. It checks RPC endpoints and prints a warning if the wallet balance is below `min_portfolio_value`. Set `AUTO_SELECT_KEYPAIR=1` so the Web UI matches this behaviour.

You can still run the bot manually with explicit options:
```bash
python start.py --dry-run
# or
python -m solhunter_zero.main

```
