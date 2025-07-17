# SolHunter Zero

SolHunter Zero is an autonomous AI-driven trading bot for the Solana blockchain. This project is a starting point based on the provided high-level specifications. It includes a scanner, Monte Carlo simulation engine and a simple decision loop backed by SQLite.

## Requirements
- Python 3.11+
- [Poetry](https://python-poetry.org/) (optional but recommended)

Install Python dependencies:
```bash
pip install -r requirements.txt
```

### Optional: local LLM via Ollama
Install [Ollama](https://ollama.ai/download) (works on macOS/arm64) and pull the `mistral` model:
```bash
ollama pull mistral
```
The bot will send analysis prompts to `http://localhost:11434` by default.

## Usage
Run the bot with:
```bash
python -m solhunter_zero.main [--loop-delay SECS] [--memory-path FILE]
```
`--loop-delay` sets the initial delay between scans (default 60 seconds) and `--memory-path` controls where the SQLite database is written.

You can also pass a full database URL and delay value:
```bash
python -m solhunter_zero.main --memory-path sqlite:///my.db --loop-delay 30
```

The scanner queries the Solana blockchain directly. You can point it to a different RPC endpoint by setting the environment variable:
```bash
export SOLANA_RPC_URL=https://your.rpc.endpoint
```

The project is still experimental and does **not** place real trades. It scans for newly created tokens that end with `bonk`, runs a Monte Carlo simulation and a small sentiment check via the local LLM, then logs all decisions to a SQLite database. Further development is required to implement real trading logic.
