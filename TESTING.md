# Testing

Before running the test suite make sure all dependencies are installed. The project
uses optional heavy packages for advanced features and many tests expect them to
be available. Install the package in editable mode with the heavy extras:

```bash
pip install -e .[heavy]
```

If extras are unavailable, manually install the common heavy packages first:

```bash
pip install numpy aiohttp solana torch faiss-cpu
```

Then run the tests from the project root:

```bash
pytest
```
