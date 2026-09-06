FAMILY ?= fp8
SIZE ?= 8192
PYTHON ?= python3

run:
	$(PYTHON) runner.py $(FAMILY) $(SIZE)

tune:
	$(PYTHON) runner.py $(FAMILY) $(SIZE) tune

.PHONY: run tune
