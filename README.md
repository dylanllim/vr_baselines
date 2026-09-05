# Vera Rubin GEMM baselines

Minimal reproduction of the cuBLASLt, CuTeDSL, and CUTLASS FP8/NVFP4 baselines. CUTLASS is pinned as a submodule; no ThunderKittens code is included.

## Setup

Requires CUDA 13.4, CMake, Ninja, and (for CuTeDSL) PyTorch plus `nvidia-cutlass-dsl`. Initialize CUTLASS:

```bash
git submodule update --init
```

## Fixed protocol

- Single source of truth: [`baselines/protocol.py`](baselines/protocol.py).
- Square `M=N=K`: `1024 2048 4096 8192 16384 32768` only.
- Seed: `2024`.
- FP8 inputs: uniform `[-448,448]`, quantized to E4M3.
- NVFP4 inputs: uniform packed E2M1 codes; UE4M3 block and FP32 global scales uniform `[0.1,10]` where the API exposes them.
- Every reported sample: 5 warmups + 10 timed launches.
- Inputs rotate over at least 3× L2 when the problem fits in cache.
- Reference values: [`baselines/reference_results.csv`](baselines/reference_results.csv).

## Results

Preliminary node 9 mean TFLOP/s from five samples. These were collected before the final unified protocol and must be replayed before publication; `pending` means the structural search is still running.

### NVFP4

| M=N=K | CuTeDSL | cuBLASLt | CUTLASS |
|---:|---:|---:|---:|
| 1,024 | pending | 513 | 408 |
| 2,048 | pending | 2,932 | 2,181 |
| 4,096 | pending | 9,099 | 7,749 |
| 8,192 | pending | 16,623 | 13,322 |
| 16,384 | pending | 15,393 | 12,379 |
| 32,768 | pending | 11,926 | 9,859 |

### FP8

| M=N=K | CuTeDSL | cuBLASLt | CUTLASS |
|---:|---:|---:|---:|
| 1,024 | 238 | 508 | 422 |
| 2,048 | 1,812 | 2,502 | 1,882 |
| 4,096 | 6,105 | 6,225 | 5,262 |
| 8,192 | 10,146 | 9,347 | 7,272 |
| 16,384 | 11,011 | 8,885 | 7,231 |
| 32,768 | 8,343 | 6,520 | 5,406 |

## Run

Normal runs replay the pinned winner and collect five fresh 5/10 samples:

```bash
make -C baselines/cublas FAMILY=fp8 SIZE=8192 run
make -C baselines/cublas FAMILY=nvfp4 SIZE=8192 run
```

CuTeDSL imports Rubin kernels directly from `submodules/cutlass/examples/python/CuTeDSL/cute/rubin/kernel`:

```bash
PYTHON=/path/to/python make -C baselines/cutedsl FAMILY=fp8 SIZE=8192 run
PYTHON=/path/to/python make -C baselines/cutedsl FAMILY=nvfp4 SIZE=8192 run
```

CUTLASS builds 40 SM107a kernels per datatype. Its build temporarily applies [`protocol.patch`](baselines/cutlass/protocol.patch) to initialize FP4 operands and scale tensors from the shared protocol, then restores the pinned submodule:

```bash
make -C baselines/cutlass build
make -C baselines/cutlass FAMILY=fp8 SIZE=8192 run
make -C baselines/cutlass FAMILY=nvfp4 SIZE=8192 run
```

Use `tune` instead of `run` to search again. CuTeDSL searches every checked-in configuration; CUTLASS screens every compiled kernel for two seconds; cuBLASLt times every returned top-32 heuristic candidate.

Use `CUDA_VISIBLE_DEVICES` to select the GPU. Results are TFLOP/s; all three backends emit BF16.
