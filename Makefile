# Simple Makefile to run SolHunter Zero

PYTHON ?= python

.PHONY: start test demo demo-rl

start:
	$(PYTHON) scripts/startup.py $(ARGS)

test:
	$(PYTHON) -m pytest $(ARGS)

demo:
        $(PYTHON) scripts/investor_demo.py --preset short --reports reports $(ARGS)

demo-rl:
        $(PYTHON) scripts/investor_demo.py --preset short --reports reports --rl-demo $(ARGS)

