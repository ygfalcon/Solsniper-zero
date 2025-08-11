# Testing

Before running the test suite make sure all dependencies are installed.
Install the package in editable mode with the development extras:

```bash
pip install -e .[dev]
```
The dependency list in `pyproject.toml` pins each package to a
minimum/maximum version range so tests run against consistent builds.

Heavy packages such as `torch`, `transformers` and `faiss-cpu` are
required for the full test suite. Install them along with their
dependencies using the ``full`` extra:

```bash
pip install .[full]
```

Then run the tests from the project root:

```bash
pytest
```

## Investor demo

The investor demo performs a small rolling backtest and writes lightweight
reports for each strategy. Run its test directly to generate these files:

```bash
pytest tests/test_investor_demo.py
```

The test stores `summary.json`, `summary.csv`, `trade_history.csv` and
`highlights.json` in a temporary reports directory. Each entry lists the
configuration name along with metrics such as ROI, Sharpe ratio, maximum
drawdown and final capital for strategies like `buy_hold`, `momentum` and
`mixed`. Inspect any of the files to compare strategy performance.

To run the demo from the command line specify an output folder. It is designed
to run quickly even in restricted environments and accepts additional options to
control the input data and starting capital:

```bash
python demo.py --reports reports --data data.csv --capital 1000
```

After it finishes, inspect the generated reports:

```bash
head reports/trade_history.csv
python -m json.tool reports/highlights.json
```

## Paper Trading

Execute the lightweight paper trading workflow and verify ROI calculation:

```bash
pytest tests/test_paper.py
```

Run the CLI directly to generate a simple ROI summary:

```bash
python paper.py --reports reports
```

## Startup integration flow

Verify the launcher and startup script integration without invoking the full
stack by running:

```bash
pytest tests/test_startup_sequence.py
```

To run the static analysis checks used in CI, execute:

```bash
python -m compileall .
flake8
```
