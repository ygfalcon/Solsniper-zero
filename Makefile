# Simple Makefile to run SolHunter Zero

PYTHON ?= python


.PHONY: start test demo demo-rl demo-multi

start:
	$(PYTHON) scripts/startup.py $(ARGS)

test:
        $(PYTHON) -m pytest $(ARGS)

# Run the investor demo. Pass ARGS to override or extend defaults.
# Examples:
#   make demo ARGS="--preset multi"
#   make demo ARGS="--rl-demo --reports reports"
demo:
        $(PYTHON) scripts/investor_demo.py --preset short --reports reports $(ARGS)

demo-multi:
        $(MAKE) demo ARGS="--preset multi"

demo-rl:
        $(MAKE) demo ARGS="--rl-demo --reports reports"

