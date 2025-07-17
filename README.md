# SolHunter Zero

SolHunter Zero is an autonomous AI-driven trading bot for the Solana blockchain. The codebase includes a lightweight scanner, a Monte Carlo simulation engine and a simple decision loop backed by SQLite.

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
python -m solhunter_zero.main
```

The scanner queries the Solana blockchain directly. You can point it to a different RPC endpoint by setting the environment variable:
```bash
export SOLANA_RPC_URL=https://your.rpc.endpoint
```

The project is still experimental and does **not** place real trades. The bot scans for newly created tokens that end with `bonk`, runs a Monte Carlo simulation and a small sentiment check via the local LLM, then logs all decisions to a SQLite database.

The current implementation is a minimal scaffold that demonstrates the system architecture. Further development is required to implement real trading logic.
