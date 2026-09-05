/***************************************************************************************************
 * cuBLASLt FP8 GEMM Benchmark
 *
 * D = A * B (no alpha/beta scaling, no C input)
 * A: RowMajor (M x K), B: ColMajor (N x K), D: RowMajor (M x N)
 * Input: FP8 E4M3, Accumulator: FP32, Output: BF16
 **************************************************************************************************/

#include <iostream>
#include <vector>
#include <cuda_runtime.h>
#include <cublasLt.h>
#include <cuda_fp8.h>
#include <cuda_bf16.h>

#include "baseline_utils.cuh"
#include "autotune.cuh"

static constexpr AlgoConfig winners[] = {
  {1024, 66, 318, 36, 1, 0, 0, 0, 3, 1},
  {2048, 66, 29, 36, 1, 0, 0, 0, 3, 1},
  {4096, 66, 23, 36, 3, 0, 0, 0, 9, 1},
  {8192, 66, 23, 36, 1, 0, 0, 0, 3, 1},
  {16384, 66, 513, 36, 1, 0, 0, 0, 3, 1},
  {32768, 66, 513, 36, 1, 0, 0, 0, 3, 1},
};

///////////////////////////////////////////////////////////////////////////////////////////////////
// cuBLASLt GEMM: D = A * B
// A: RowMajor (M x K), B: ColMajor (N x K), D: RowMajor (M x N)
// Input: FP8 E4M3, Accumulator: FP32, Output: BF16
///////////////////////////////////////////////////////////////////////////////////////////////////

struct CublasLtGemm {
  cublasLtHandle_t handle;
  cublasLtMatmulDesc_t matmulDesc;
  cublasLtMatrixLayout_t layoutA, layoutB, layoutD;
  cublasLtMatmulPreference_t preference;
  cublasLtMatmulHeuristicResult_t heuristic;
  void* workspace;
  size_t workspaceSize;

  void init(int M, int N, int K) {
    CHECK_CUBLAS(cublasLtCreate(&handle));

    // Create matmul descriptor
    CHECK_CUBLAS(cublasLtMatmulDescCreate(&matmulDesc, CUBLAS_COMPUTE_32F, CUDA_R_32F));

    // D[m,n] = sum_k A[m,k] * B[n,k]
    // A: RowMajor MxK = ColMajor KxM, B: ColMajor NxK = RowMajor NxK = ColMajor KxN
    // D: RowMajor MxN = ColMajor NxM
    // In col-major: D' = B'^T * A' where B' is KxN, A' is KxM, D' is NxM
    cublasOperation_t transA = CUBLAS_OP_T;  // B' (KxN) transposed gives NxK
    cublasOperation_t transB = CUBLAS_OP_N;  // A' (KxM) as-is gives KxM
    CHECK_CUBLAS(cublasLtMatmulDescSetAttribute(matmulDesc, CUBLASLT_MATMUL_DESC_TRANSA, &transA, sizeof(transA)));
    CHECK_CUBLAS(cublasLtMatmulDescSetAttribute(matmulDesc, CUBLASLT_MATMUL_DESC_TRANSB, &transB, sizeof(transB)));

    // Layout for B (cuBLAS "A"): RowMajor NxK = ColMajor KxN, ld=K
    CHECK_CUBLAS(cublasLtMatrixLayoutCreate(&layoutA, CUDA_R_8F_E4M3, K, N, K));
    // Layout for A (cuBLAS "B"): RowMajor MxK = ColMajor KxM, ld=K
    CHECK_CUBLAS(cublasLtMatrixLayoutCreate(&layoutB, CUDA_R_8F_E4M3, K, M, K));
    // Layout for D: RowMajor MxN = ColMajor NxM, ld=N
    CHECK_CUBLAS(cublasLtMatrixLayoutCreate(&layoutD, CUDA_R_16BF, N, M, N));

    // Workspace
    workspaceSize = size_t(512) << 20;
    CHECK_CUDA(cudaMalloc(&workspace, workspaceSize));

    // Preference
    CHECK_CUBLAS(cublasLtMatmulPreferenceCreate(&preference));
    CHECK_CUBLAS(cublasLtMatmulPreferenceSetAttribute(preference, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
                                                       &workspaceSize, sizeof(workspaceSize)));

  }

  void run(__nv_fp8_e4m3 const* A, __nv_fp8_e4m3 const* B, __nv_bfloat16* D, cudaStream_t stream = nullptr) {
    const float alpha = 1.0f;
    const float beta = 0.0f;
    // Note: B is first arg, A is second arg (for the transpose trick)
    CHECK_CUBLAS(cublasLtMatmul(handle, matmulDesc, &alpha,
                                 B, layoutA,   // "A" in cublasLt = our B
                                 A, layoutB,   // "B" in cublasLt = our A
                                 &beta,
                                 D, layoutD,
                                 D, layoutD,
                                 &heuristic.algo, workspace, workspaceSize, stream));
  }

  void autotune(__nv_fp8_e4m3 const* A, __nv_fp8_e4m3 const* B, __nv_bfloat16* D) {
    const float alpha = 1.0f, beta = 0.0f;
    heuristic = ::autotune(
        handle, matmulDesc, layoutA, layoutB, layoutD, layoutD, preference,
        &alpha, B, A, &beta, D, D, workspace, workspaceSize);
  }

  void select(int size) {
    heuristic = pinned_algo(
        winners, size, handle, matmulDesc, layoutA, layoutB, layoutD, layoutD,
        CUDA_R_8F_E4M3);
  }

  void destroy() {
    CHECK_CUDA(cudaFree(workspace));
    CHECK_CUBLAS(cublasLtMatmulPreferenceDestroy(preference));
    CHECK_CUBLAS(cublasLtMatrixLayoutDestroy(layoutA));
    CHECK_CUBLAS(cublasLtMatrixLayoutDestroy(layoutB));
    CHECK_CUBLAS(cublasLtMatrixLayoutDestroy(layoutD));
    CHECK_CUBLAS(cublasLtMatmulDescDestroy(matmulDesc));
    CHECK_CUBLAS(cublasLtDestroy(handle));
  }
};

///////////////////////////////////////////////////////////////////////////////////////////////////
// Benchmark function
///////////////////////////////////////////////////////////////////////////////////////////////////

void benchmark(int M, int N, int K, bool tune) {
  // Cooldown between configurations
  sleep_ms(500);

  std::cout << "\n----------------------------------------" << std::endl;
  std::cout << "Problem size: M=" << M << ", N=" << N << ", K=" << K << std::endl;

  // L2 cache eviction - multiple buffer groups
  int l2_cache_size;
  cudaDeviceGetAttribute(&l2_cache_size, cudaDevAttrL2CacheSize, 0);
  // FP8 inputs are 1 byte each, BF16 output is 2 bytes
  const size_t arg_size = size_t(M) * K + size_t(N) * K + 2 * size_t(M) * N;
  const size_t ideal_arg_size = size_t(l2_cache_size) * 3;
  const int arg_group_count = BENCHMARK_COLD_L2 && arg_size <= ideal_arg_size
      ? int(ideal_arg_size / arg_size) + 1 : 1;

  // Allocate buffer groups
  std::vector<__nv_fp8_e4m3*> blocks_A(arg_group_count);
  std::vector<__nv_fp8_e4m3*> blocks_B(arg_group_count);
  std::vector<__nv_bfloat16*> blocks_D(arg_group_count);
  size_t size_A = size_t(M) * K;
  size_t size_B = size_t(K) * N;
  size_t size_D = size_t(M) * N;

  uint64_t seed = BENCHMARK_SEED;
  for (int i = 0; i < arg_group_count; ++i) {
    CHECK_CUDA(cudaMalloc(&blocks_A[i], size_A * sizeof(__nv_fp8_e4m3)));
    CHECK_CUDA(cudaMalloc(&blocks_B[i], size_B * sizeof(__nv_fp8_e4m3)));
    CHECK_CUDA(cudaMalloc(&blocks_D[i], size_D * sizeof(__nv_bfloat16)));

    fill_uniform(blocks_A[i], size_A, seed + i * 100, BENCHMARK_FP8_LOW, BENCHMARK_FP8_HIGH);
    fill_uniform(blocks_B[i], size_B, seed + i * 100 + 1, BENCHMARK_FP8_LOW, BENCHMARK_FP8_HIGH);
    fill_zero(blocks_D[i], size_D);
  }
  CHECK_CUDA(cudaDeviceSynchronize());

  // Initialize cuBLASLt
  CublasLtGemm gemm;
  gemm.init(M, N, K);
  if (tune) gemm.autotune(blocks_A[0], blocks_B[0], blocks_D[0]);
  else gemm.select(M);

  cudaStream_t stream;
  CHECK_CUDA(cudaStreamCreate(&stream));

  cudaEvent_t start, stop;
  CHECK_CUDA(cudaEventCreate(&start));
  CHECK_CUDA(cudaEventCreate(&stop));
  int64_t flops = int64_t(2) * M * N * K;
  for (int i = 0; i < BENCHMARK_WARMUPS; ++i)
    gemm.run(blocks_A[i % arg_group_count], blocks_B[i % arg_group_count],
             blocks_D[i % arg_group_count], stream);
  CHECK_CUDA(cudaStreamSynchronize(stream));
  CHECK_CUDA(cudaEventRecord(start, stream));
  for (int i = 0; i < BENCHMARK_ITERATIONS; ++i)
    gemm.run(blocks_A[i % arg_group_count], blocks_B[i % arg_group_count],
             blocks_D[i % arg_group_count], stream);
  CHECK_CUDA(cudaEventRecord(stop, stream));
  CHECK_CUDA(cudaStreamSynchronize(stream));
  float elapsed = 0;
  CHECK_CUDA(cudaEventElapsedTime(&elapsed, start, stop));
  double runtime_ms = elapsed / BENCHMARK_ITERATIONS;
  double tflops = (double(flops) / 1e12) / (runtime_ms / 1000.0);
  std::cout << "Average runtime: " << runtime_ms << " ms\n"
            << "Performance: " << tflops << " TFLOP/s\n";

  // Cleanup
  CHECK_CUDA(cudaEventDestroy(start));
  CHECK_CUDA(cudaEventDestroy(stop));
  CHECK_CUDA(cudaStreamDestroy(stream));
  gemm.destroy();

  for (int i = 0; i < arg_group_count; ++i) {
    CHECK_CUDA(cudaFree(blocks_A[i]));
    CHECK_CUDA(cudaFree(blocks_B[i]));
    CHECK_CUDA(cudaFree(blocks_D[i]));
  }
}

///////////////////////////////////////////////////////////////////////////////////////////////////

int main(int argc, char **argv) {
  std::cout << "cuBLASLt FP8 GEMM Profiler" << std::endl;
  std::cout << "D = A * B, A: RowMajor (MxK), B: ColMajor (NxK), D: RowMajor (MxN)" << std::endl;
  std::cout << "Input: FP8 E4M3, Accumulator: FP32, Output: BF16" << std::endl;
  std::cout << "Warmup: " << BENCHMARK_WARMUPS << ", Profiling: " << BENCHMARK_ITERATIONS << std::endl;

  bool tune = argc > 2 && std::string(argv[2]) == "--tune";
  if (argc > 1) {
    size_t n = atoll(argv[1]);
    if (!benchmark_size(n)) {
      std::cerr << "Unsupported SIZE=" << n << "; expected one of";
      print_benchmark_sizes(std::cerr);
      std::cerr << std::endl;
      return 2;
    }
    benchmark(n, n, n, tune);
    return 0;
  }

  for (int n : benchmark_sizes) benchmark(n, n, n, tune);

  return 0;
}
