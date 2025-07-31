# Datasets

This directory hosts small JSON examples for tests and experimentation.

- `sample_ticks.json` &mdash; synthetic tick data for offline training.
- `alien_cipher.json` &mdash; minimal table of **glyph-derived coefficients** used by some sandbox agents. Each entry maps a token symbol to an `r` value and number of `iterations`.

To regenerate `alien_cipher.json` programmatically, see [`scripts/generate_alien_cipher_dataset.py`](../scripts/generate_alien_cipher_dataset.py).
