PYTHON ?= python3
SIZE ?= 8192
FAMILY ?= all
JOBS ?= 64

build:
	JOBS=$(JOBS) ./scripts/build.sh

run:
	$(PYTHON) runner.py $(FAMILY) $(SIZE)

tune:
	$(PYTHON) runner.py $(FAMILY) $(SIZE) tune

.PHONY: build run tune
