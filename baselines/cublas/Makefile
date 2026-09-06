# FP8:   make FAMILY=fp8 build    # src/fp8.cu   -> build/fp8
# NVFP4: make FAMILY=nvfp4 build  # src/nvfp4.cu -> build/nvfp4
# Replace `build` with `run` or `tune` to execute the selected family.
FAMILY ?= fp8
SIZE ?= 8192
NVCC ?= /usr/local/cuda/bin/nvcc
PYTHON ?= python3
NVCCFLAGS ?= -std=c++20 -O3 --use_fast_math -gencode arch=compute_107a,code=sm_107a
BUILD := build
OUT := $(BUILD)/$(FAMILY)
SOURCE := src/$(FAMILY).cu
PROTOCOL := $(BUILD)/protocol.cuh

build: $(OUT)

$(PROTOCOL): ../protocol.py
	mkdir -p $(BUILD)
	$(PYTHON) $< > $@

$(OUT): $(SOURCE) src/common.cuh src/algorithms.cuh $(PROTOCOL)
	$(NVCC) $(NVCCFLAGS) -I$(BUILD) $< -o $@ -lcublasLt -lcublas

run: build
	$(PYTHON) runner.py $(FAMILY) $(SIZE)

tune: build
	$(PYTHON) runner.py $(FAMILY) $(SIZE) tune

clean:
	$(RM) -r $(BUILD)

.PHONY: build run tune clean
