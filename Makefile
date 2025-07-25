# Simple Makefile to run SolHunter Zero

PYTHON ?= python

.PHONY: start test

start:
	$(PYTHON) scripts/startup.py $(ARGS)

test:
	$(PYTHON) -m pytest $(ARGS)

