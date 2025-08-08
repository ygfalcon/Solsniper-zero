# Simple Makefile to run SolHunter Zero
# The `run` target uses the cross-platform Python entry point.

PYTHON ?= python3

.RECIPEPREFIX := >

.PHONY: start run test demo demo-rl demo-multi setup proto

start: solhunter_zero/event_pb2.py
>$(PYTHON) start.py $(ARGS)

setup: solhunter_zero/event_pb2.py
>$(PYTHON) start.py --one-click $(ARGS)

# Launch directly without the shell script (works on all platforms)
run: solhunter_zero/event_pb2.py
>$(PYTHON) -m solhunter_zero.main --auto $(ARGS)

test:
>$(PYTHON) -m pytest $(ARGS)

typecheck:
>$(PYTHON) -m mypy solhunter_zero tests

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

solhunter_zero/event_pb2.py: proto/event.proto
>$(PYTHON) -m grpc_tools.protoc -I proto --python_out=solhunter_zero proto/event.proto

proto: solhunter_zero/event_pb2.py

