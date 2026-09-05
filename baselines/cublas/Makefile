FAMILY ?= fp8
SIZE ?= 8192
NVCC ?= /usr/local/cuda/bin/nvcc
PYTHON ?= python3
OUT := $(FAMILY).out
PROTOCOL := .protocol.cuh

$(PROTOCOL): ../protocol.py
	$(PYTHON) $< > $@

$(OUT): $(FAMILY).cu baseline_utils.cuh autotune.cuh $(PROTOCOL)
	$(NVCC) -std=c++20 -O3 --use_fast_math -gencode arch=compute_107a,code=sm_107a $< -o $@ -lcublasLt -lcublas

run: $(OUT)
	$(PYTHON) benchmark.py --binary ./$(OUT) --size $(SIZE)

tune: $(OUT)
	$(PYTHON) benchmark.py --binary ./$(OUT) --size $(SIZE) --tune

single: $(OUT)
	./$(OUT) $(SIZE)

clean:
	$(RM) *.out $(PROTOCOL)

.PHONY: run tune single clean
