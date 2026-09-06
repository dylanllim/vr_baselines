FAMILY ?= fp8
SIZE ?= 8192
PYTHON ?= python3

# BF16 output: FAMILY=fp8 or FAMILY=nvfp4
# Low-precision output: FAMILY=fp8_fp8 or FAMILY=nvfp4_fp4

run:
	$(PYTHON) runner.py $(FAMILY) $(SIZE)

tune:
	$(PYTHON) runner.py $(FAMILY) $(SIZE) tune

.PHONY: run tune
