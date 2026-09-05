PYTHON ?= python3
SIZE ?= 8192
FAMILY ?= all
JOBS ?= 64

build:
	JOBS=$(JOBS) ./build.sh

run:
	$(PYTHON) benchmark.py --family $(FAMILY) --size $(SIZE)

tune:
	$(PYTHON) benchmark.py --family $(FAMILY) --size $(SIZE) --tune

.PHONY: build run tune
