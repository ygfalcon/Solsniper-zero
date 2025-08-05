# Simple Makefile to run SolHunter Zero

PYTHON ?= python

.PHONY: start test demo demo-full

start:
	$(PYTHON) scripts/startup.py $(ARGS)

test:
	$(PYTHON) -m pytest $(ARGS)

demo:
	$(PYTHON) scripts/investor_demo.py --preset short --reports reports $(ARGS)

demo-full:
	$(PYTHON) scripts/investor_demo.py --preset multi --full-system --reports reports $(ARGS)

