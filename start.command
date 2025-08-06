#!/bin/bash
# This launcher expects Python 3.11 or newer to be available as `python`.
cd "$(dirname "$0")"
python scripts/startup.py --one-click
