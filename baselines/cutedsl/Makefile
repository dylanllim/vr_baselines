FAMILY ?= fp8
SIZE ?= 8192
PYTHON ?= python3

run:
	$(PYTHON) runner.py --family $(FAMILY) --size $(SIZE)

tune:
	$(PYTHON) runner.py --family $(FAMILY) --size $(SIZE) --tune

.PHONY: run tune
