#!/usr/bin/env bash
set -euo pipefail

here=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cutlass=$here/../../submodules/cutlass
build=$cutlass/build
cmake=${CMAKE:-cmake}
cuda=${CUDA_HOME:-/usr/local/cuda}
patch=$here/patches/protocol.patch

# Match the shared NVFP4 operand/scale distributions without modifying kernels.
if git -C "$cutlass" apply -R --check "$patch" 2>/dev/null; then
  git -C "$cutlass" apply -R "$patch"
fi
git -C "$cutlass" apply --check "$patch"
git -C "$cutlass" apply "$patch"
trap 'git -C "$cutlass" apply -R "$patch"' EXIT

"$cmake" -S "$cutlass" -B "$build" -GNinja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_COMPILER="$cuda/bin/nvcc" \
  -DCUDA_TOOLKIT_ROOT_DIR="$cuda" \
  -DCUTLASS_NVCC_ARCHS=107a \
  -DCUTLASS_LIBRARY_OPERATIONS=gemm \
  -DCUTLASS_LIBRARY_KERNELS='cutlass3x_sm107_bstensorop_gemm_ue4m3xe2m1_ue4m3xe2m1_f32_bf16_bf16,cutlass3x_sm107_tensorop_gemm_e4m3_e4m3_f32_bf16_bf16' \
  -DCUTLASS_LIBRARY_EXCLUDE_KERNELS='stream_k,gemm_grouped,4x1x1,4x4x1,e3m2,e2m3,f4,ttt,nnn,nnt,tnn,ttt,ntn,ttn,ntt' \
  -DCUTLASS_ENABLE_TESTS=OFF \
  -DCUTLASS_ENABLE_EXAMPLES=OFF \
  -DCUTLASS_UNITY_BUILD_ENABLED=OFF \
  -DCUTLASS_ENABLE_CUBLAS=OFF

list="$build/tools/library/generated_kernels.txt"
count=$(awk 'NF { n++ } END { print n + 0 }' "$list")
printf 'GENERATED_KERNEL_COUNT=%s\n' "$count"
(( count <= 1000 )) || { printf 'Refusing to compile more than 1000 kernels\n' >&2; exit 1; }
"$cmake" --build "$build" --target cutlass_profiler --parallel "${JOBS:-64}"
