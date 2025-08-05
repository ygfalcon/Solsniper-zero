# Testing

Before running the test suite make sure all dependencies are installed.
Install the package in editable mode with the development extras:

```bash
pip install -e .[dev]
```
The dependency list in `pyproject.toml` pins each package to a
minimum/maximum version range so tests run against consistent builds.

Heavy packages such as `torch`, `transformers` and `faiss-cpu` are
required for the full test suite along with core dependencies like
`numpy`, `aiohttp` and `solana`. If you need to reinstall them
manually, run:

```bash
pip install numpy aiohttp solana torch transformers faiss-cpu
```

Then run the tests from the project root:

```bash
pytest
```

## Investor demo

The investor demo performs a small rolling backtest and writes a JSON and CSV
summary for each strategy. Run its test directly to generate these reports:

```bash
pytest tests/test_investor_demo.py
```

The test stores `summary.json` and `summary.csv` in a temporary reports
directory. Each entry lists the configuration name along with metrics such as
ROI, Sharpe ratio, maximum drawdown and final capital for strategies like
`buy_hold`, `momentum` and `mixed`. Inspect either file to compare strategy
performance.

To run the demo from the command line specify an output folder:

```bash
python scripts/investor_demo.py --reports reports
```

To run the static analysis checks used in CI, execute:

```bash
python -m compileall .
flake8
```
