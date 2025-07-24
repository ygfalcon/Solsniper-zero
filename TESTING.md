# Testing

Before running the test suite make sure all dependencies are installed.
Install the package in editable mode:

```bash
pip install -e .
```

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
