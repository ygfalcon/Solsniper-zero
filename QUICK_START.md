# Quick Start

- make start runs scripts/startup.py for guided setup.
- Set `AUTO_SELECT_KEYPAIR=1` to always use a single keypair.
- Execute `./run.sh --auto` to start trading automatically.
- Launch the Web UI with `python -m solhunter_zero.ui`.
- Or start everything at once with `python scripts/start_all.py`.
- By default ranking tasks use half your CPU cores. Override with
  `MAX_CONCURRENCY` and reduce load automatically by setting
  `CPU_USAGE_THRESHOLD`.
