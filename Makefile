# Simple Makefile to run SolHunter Zero
# The `run` target uses the cross-platform Python entry point.

PYTHON ?= python3

.RECIPEPREFIX := >

.PHONY: start run test demo demo-rl demo-multi

start:
>./start.py $(ARGS)

# Launch directly without the shell script (works on all platforms)
run:
>$(PYTHON) -m solhunter_zero.main --auto $(ARGS)

test:
>$(PYTHON) -m pytest $(ARGS)

# Run the investor demo. Pass ARGS to override or extend defaults.
# Examples:
#   make demo ARGS="--preset multi"
#   make demo ARGS="--rl-demo --reports reports"
demo:
>$(PYTHON) scripts/investor_demo.py --preset short --reports reports $(ARGS)

demo-multi:
>$(MAKE) demo ARGS="--preset multi"

demo-rl:
>$(MAKE) demo ARGS="--rl-demo --reports reports"

