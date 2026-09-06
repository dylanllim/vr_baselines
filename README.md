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
- Cold-L2 mode flushes cache and/or rotates disjoint buffers, as supported by each backend.

## Results

TFLOP/s, measured sequentially on node 9 GPU 1 of a Vera Rubin QS rack. Each
entry is the mean of five fresh samples under the fixed protocol above (CUDA
13.4, driver 615.62).

### NVFP4, bf16 out

| M=N=K | CuTeDSL | cuBLASLt | CUTLASS |
|---:|---:|---:|---:|
| 1,024 | 142 | 518 | 415 |
| 2,048 | 1,098 | 3,006 | 2,234 |
| 4,096 | 7,873 | 9,129 | 7,851 |
| 8,192 | 18,466 | 18,381 | 13,287 |
| 16,384 | 20,892 | 21,214 | 12,378 |
| 32,768 | 19,216 | 21,102 | 9,948 |

### FP8, bf16 out

| M=N=K | CuTeDSL | cuBLASLt | CUTLASS |
|---:|---:|---:|---:|
| 1,024 | 248 | 544 | 437 |
| 2,048 | 1,930 | 2,492 | 1,976 |
| 4,096 | 6,688 | 6,918 | 5,350 |
| 8,192 | 10,159 | 11,082 | 7,317 |
| 16,384 | 11,036 | 11,771 | 7,329 |
| 32,768 | 8,840 | 8,736 | 5,678 |

### NVFP4, fp4 out

| M=N=K | CuTeDSL | cuBLASLt |
|---:|---:|---:|
| 1,024 | 163 | 506 |
| 2,048 | 1,325 | 2,911 |
| 4,096 | 7,246 | 9,894 |
| 8,192 | 16,607 | 18,612 |
| 16,384 | 19,274 | 21,954 |
| 32,768 | 19,724 | 21,499 |

### FP8, fp8 out

| M=N=K | CuTeDSL | cuBLASLt |
|---:|---:|---:|
| 1,024 | 351 | 550 |
| 2,048 | 2,326 | 2,619 |
| 4,096 | 7,137 | 7,493 |
| 8,192 | 10,337 | 11,588 |
| 16,384 | 11,196 | 12,065 |
| 32,768 | 8,917 | 8,871 |

## Run

All backends use the same Make interface:

```bash
CUDA_VISIBLE_DEVICES=<gpu> make -C baselines/<backend> \
  FAMILY=<family> SIZE=<size> <run|tune>
```

- Backends: `cublas`, `cutedsl`, or `cutlass`.
- Families: `fp8` and `nvfp4` emit BF16 on all backends; `fp8_fp8` and
  `nvfp4_fp4` select low-precision output in cuBLASLt and CuTeDSL.
- Sizes: `1024`, `2048`, `4096`, `8192`, `16384`, or `32768`.
- `run` replays the pinned winner and collects five fresh samples.
- `tune` searches again and reports the selected configuration. Pin the winner,
  then use `run` for the five published samples.

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

CuTeDSL searches 104 checked-in dense FP8 configurations or 1,374 dense NVFP4
configurations per shape. Its FP4-output search attempts the complete stock
Rubin grouped and mixed-cluster structural space: 516 configurations at 1K and
552 at larger sizes. CUTLASS screens all 40 compiled kernels per datatype for two seconds.
cuBLASLt enumerates every compatible algorithm ID and its reported structural
choices. It tests split-K 1 plus `2,3,4,5,6,8,12,16,32` with every supported
reduction scheme, screens with 1+3, retimes the best 100 with 5+10, and ranks
the best 10 plus the incumbent over three 5+10 measurements.

The FP8 low-output paths emit E4M3. For FP4 output, cuBLASLt emits packed E2M1
plus its tiled E4M3 output scales; the stock Rubin CuTeDSL grouped kernel emits
a direct packed E2M1 cast without output-scale generation. The FP4-output table
therefore exposes an epilogue-contract difference rather than implying the two
paths perform identical output quantization.
