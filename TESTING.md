# Testing

Before running the test suite make sure all dependencies are installed.
Install the package in editable mode with the development extras:

```bash
pip install -e .[dev]
```
The dependency list in `pyproject.toml` pins each package to a
minimum/maximum version range so tests run against consistent builds.

Heavy packages such as `numpy`, `aiohttp`, `solana`, `torch` and
`faiss-cpu` are included in the default requirements. If you need to
reinstall them manually, run:

```bash
pip install numpy aiohttp solana torch faiss-cpu
```

Then run the tests from the project root:

```bash
pytest
```
