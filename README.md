# SolHunter Zero

SolHunter Zero is an autonomous AI-driven trading bot for the Solana blockchain.

See [QUICK_START.md](QUICK_START.md) for a condensed setup guide.

## Documentation

- [Full system startup](docs/full-system-startup.md)
- [MEV Bundles](docs/mev-bundles.md)
- [Agents](docs/agents.md)

## Quick Start

1. Install Python 3.11+.
2. Install dependencies:
   ```bash
   pip install .[uvloop]
   ```
3. Create a configuration file (`config.toml` or `config.yaml`) with your API keys, RPC URL and DEX endpoints.
4. Run:
   ```bash
   python start.py
   ```
   Load the keypair in the SolHunter GUI and press **Start**.
