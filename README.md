# Vera Rubin GEMM baselines

Minimal reproduction of the cuBLASLt, CuTeDSL, and CUTLASS FP8/NVFP4 baselines.

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

## Results

Preliminary node 9 mean TFLOP/s from five samples. CuTeDSL NVFP4 values are
legacy structural-search measurements and must be replayed with the fixed
protocol above before publication. `pending` means the structural search is
still running.

### NVFP4

| M=N=K | CuTeDSL | cuBLASLt | CUTLASS |
|---:|---:|---:|---:|
| 1,024 | 157 | 513 | 408 |
| 2,048 | 1,097 | 2,932 | 2,181 |
| 4,096 | 9,571 | 9,099 | 7,749 |
| 8,192 | pending | 16,623 | 13,322 |
| 16,384 | 20,937 | 15,393 | 12,379 |
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

All backends use the same Make interface:

```bash
CUDA_VISIBLE_DEVICES=<gpu> make -C baselines/<backend> \
  FAMILY=<fp8|nvfp4> SIZE=<size> <run|tune>
```

- Backends: `cublas`, `cutedsl`, or `cutlass`.
- Sizes: `1024`, `2048`, `4096`, `8192`, `16384`, or `32768`.
- `run` replays the pinned winner and collects five fresh samples.
- `tune` searches again before collecting the samples.

For example, the FP8 8K replay is identical across backends except for the backend name:

```bash
CUDA_VISIBLE_DEVICES=1 make -C baselines/cublas  FAMILY=fp8 SIZE=8192 run
CUDA_VISIBLE_DEVICES=1 make -C baselines/cutedsl FAMILY=fp8 SIZE=8192 run
CUDA_VISIBLE_DEVICES=1 make -C baselines/cutlass FAMILY=fp8 SIZE=8192 run
```

Use `FAMILY=nvfp4` for NVFP4. Set `PYTHON=/path/to/python` on any command if the required Python packages are not installed in the default interpreter.

### Backend setup

- cuBLASLt compiles its selected family automatically on the first `run` or `tune`.
- CuTeDSL imports the Rubin kernels directly from `submodules/cutlass/examples/python/CuTeDSL/cute/rubin/kernel` and requires PyTorch plus `nvidia-cutlass-dsl`.
- CUTLASS requires its profiler to be built once before `run` or `tune`:

```bash
make -C baselines/cutlass build
```

CUTLASS builds 40 SM107a kernels per datatype. The stock profiler cannot independently initialize packed E2M1 operands and UE4M3 scales with the ranges required by our shared protocol. The build therefore applies [`protocol.patch`](baselines/cutlass/patches/protocol.patch) temporarily and restores the pinned submodule afterward. The patch changes profiler data initialization only, not the GEMM kernels.

CuTeDSL searches every checked-in configuration; CUTLASS screens every compiled kernel for two seconds; cuBLASLt times every returned top-32 heuristic candidate. Results are TFLOP/s, and all three backends emit BF16.
