/***************************************************************************************************
 * cuBLASLt NVFP4 GEMM Benchmark
 *
 * D = A * B (no alpha/beta scaling, no C input)
 * A: RowMajor (M x K), B: ColMajor (N x K), D: RowMajor (M x N)
 * D[m,n] = sum_k A[m,k] * B[n,k]
 * Input: FP4 E2M1 with block-wise UE4M3 scaling (16-element blocks along K) + per-tensor FP32 scale
 * Accumulator: FP32, Output: BF16
 **************************************************************************************************/

#include <iostream>
#include <vector>
#include <cuda_runtime.h>
#include <cublasLt.h>
#include <cuda_fp8.h>
#include <cuda_bf16.h>
#include <cuda_fp4.h>

#include "baseline_utils.cuh"
#include "autotune.cuh"

static constexpr AlgoConfig winners[] = {
  {1024, 66, 20, 37, 2, 0, 0, 0, 5, 1},
  {2048, 66, 23, 37, 3, 0, 0, 0, 9, 1},
  {4096, 66, 23, 37, 3, 0, 0, 0, 6, 1},
  {8192, 66, 24, 37, 0, 0, 0, 0, 8, 1},
  {16384, 66, 24, 37, 0, 0, 0, 0, 8, 1},
  {32768, 66, 23, 38, 1, 0, 0, 0, 6, 1},
};

///////////////////////////////////////////////////////////////////////////////////////////////////
// cuBLASLt NVFP4 GEMM: D = A * B
// A: RowMajor (M x K), B: ColMajor (N x K), D: RowMajor (M x N)
// Block-wise UE4M3 scaling with 16-element blocks + per-tensor FP32 scale
///////////////////////////////////////////////////////////////////////////////////////////////////

struct CublasLtNvfp4Gemm {
  cublasLtHandle_t handle;
  cublasLtMatmulDesc_t matmulDesc;
  cublasLtMatrixLayout_t layoutA, layoutB, layoutC, layoutD;
  cublasLtMatmulPreference_t preference;
  cublasLtMatmulHeuristicResult_t heuristic;
  void* workspace;
  size_t workspaceSize;

  void init(int M, int N, int K) {
    CHECK_CUBLAS(cublasLtCreate(&handle));

    // Create matmul descriptor with FP32 compute
    CHECK_CUBLAS(cublasLtMatmulDescCreate(&matmulDesc, CUBLAS_COMPUTE_32F, CUDA_R_32F));

    // NVFP4 GEMM requires TN layout
    cublasOperation_t transA = CUBLAS_OP_T;
    cublasOperation_t transB = CUBLAS_OP_N;
    CHECK_CUBLAS(cublasLtMatmulDescSetAttribute(matmulDesc, CUBLASLT_MATMUL_DESC_TRANSA, &transA, sizeof(transA)));
    CHECK_CUBLAS(cublasLtMatmulDescSetAttribute(matmulDesc, CUBLASLT_MATMUL_DESC_TRANSB, &transB, sizeof(transB)));

    // Set NVFP4 block scaling mode (VEC16_UE4M3 = 16-element blocks with UE4M3 scales)
    cublasLtMatmulMatrixScale_t scaleMode = CUBLASLT_MATMUL_MATRIX_SCALE_VEC16_UE4M3;
    CHECK_CUBLAS(cublasLtMatmulDescSetAttribute(matmulDesc, CUBLASLT_MATMUL_DESC_A_SCALE_MODE, &scaleMode, sizeof(scaleMode)));
    CHECK_CUBLAS(cublasLtMatmulDescSetAttribute(matmulDesc, CUBLASLT_MATMUL_DESC_B_SCALE_MODE, &scaleMode, sizeof(scaleMode)));

    // Layout for B (cuBLAS "A"): FP4 packed
    CHECK_CUBLAS(cublasLtMatrixLayoutCreate(&layoutA, CUDA_R_4F_E2M1, K, N, K));
    // Layout for A (cuBLAS "B"): FP4 packed
    CHECK_CUBLAS(cublasLtMatrixLayoutCreate(&layoutB, CUDA_R_4F_E2M1, K, M, K));
    // Layout for D: RowMajor MxN = ColMajor NxM
    CHECK_CUBLAS(cublasLtMatrixLayoutCreate(&layoutC, CUDA_R_16BF, N, M, N));
    CHECK_CUBLAS(cublasLtMatrixLayoutCreate(&layoutD, CUDA_R_16BF, N, M, N));

    // Workspace
    workspaceSize = size_t(512) << 20;
    CHECK_CUDA(cudaMalloc(&workspace, workspaceSize));

    // Preference
    CHECK_CUBLAS(cublasLtMatmulPreferenceCreate(&preference));
    CHECK_CUBLAS(cublasLtMatmulPreferenceSetAttribute(preference, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
                                                       &workspaceSize, sizeof(workspaceSize)));

    // Set dummy scale pointers for heuristic selection (actual values don't affect algorithm choice)
    void* dummyScalePtr = workspace;
    CHECK_CUBLAS(cublasLtMatmulDescSetAttribute(matmulDesc, CUBLASLT_MATMUL_DESC_A_SCALE_POINTER, &dummyScalePtr, sizeof(dummyScalePtr)));
    CHECK_CUBLAS(cublasLtMatmulDescSetAttribute(matmulDesc, CUBLASLT_MATMUL_DESC_B_SCALE_POINTER, &dummyScalePtr, sizeof(dummyScalePtr)));

  }

  void run(__nv_fp4x2_e2m1 const* A, __nv_fp4x2_e2m1 const* B,
           __nv_fp8_e4m3 const* A_scale, __nv_fp8_e4m3 const* B_scale, float alpha,
           __nv_bfloat16* D, cudaStream_t stream = nullptr) {

    // Set block scale pointers
    CHECK_CUBLAS(cublasLtMatmulDescSetAttribute(matmulDesc, CUBLASLT_MATMUL_DESC_A_SCALE_POINTER, &B_scale, sizeof(B_scale)));
    CHECK_CUBLAS(cublasLtMatmulDescSetAttribute(matmulDesc, CUBLASLT_MATMUL_DESC_B_SCALE_POINTER, &A_scale, sizeof(A_scale)));

    float beta = 0.0f;

    CHECK_CUBLAS(cublasLtMatmul(handle, matmulDesc, &alpha,
                                 B, layoutA,
                                 A, layoutB,
                                 &beta,
                                 D, layoutC,
                                 D, layoutD,
                                 &heuristic.algo, workspace, workspaceSize, stream));
  }

  void autotune(__nv_fp4x2_e2m1 const* A, __nv_fp4x2_e2m1 const* B,
                __nv_fp8_e4m3 const* A_scale, __nv_fp8_e4m3 const* B_scale, float alpha,
                __nv_bfloat16* D) {
    CHECK_CUBLAS(cublasLtMatmulDescSetAttribute(matmulDesc, CUBLASLT_MATMUL_DESC_A_SCALE_POINTER, &B_scale, sizeof(B_scale)));
    CHECK_CUBLAS(cublasLtMatmulDescSetAttribute(matmulDesc, CUBLASLT_MATMUL_DESC_B_SCALE_POINTER, &A_scale, sizeof(A_scale)));
    float beta = 0.0f;
    heuristic = ::autotune(
        handle, matmulDesc, layoutA, layoutB, layoutC, layoutD, preference,
        &alpha, B, A, &beta, D, D, workspace, workspaceSize);
  }

  void select(int size) {
    heuristic = pinned_algo(
        winners, size, handle, matmulDesc, layoutA, layoutB, layoutC, layoutD,
        CUDA_R_4F_E2M1);
  }

  void destroy() {
    CHECK_CUDA(cudaFree(workspace));
    CHECK_CUBLAS(cublasLtMatmulPreferenceDestroy(preference));
    CHECK_CUBLAS(cublasLtMatrixLayoutDestroy(layoutA));
    CHECK_CUBLAS(cublasLtMatrixLayoutDestroy(layoutB));
    CHECK_CUBLAS(cublasLtMatrixLayoutDestroy(layoutC));
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

  int K_blocks = K / 16;  // NVFP4 block size
  size_t size_A_packed = size_t(M) * K / 2;  // FP4 packed 2 per byte
  size_t size_B_packed = size_t(N) * K / 2;
  size_t size_A_scale = size_t(M) * K_blocks;
  size_t size_B_scale = size_t(N) * K_blocks;
  size_t size_D = size_t(M) * N;

  const size_t arg_size = size_A_packed + size_B_packed +
                          size_A_scale + size_B_scale +
                          2 * size_D * sizeof(__nv_bfloat16);
  const size_t ideal_arg_size = size_t(l2_cache_size) * 3;
  const int arg_group_count = BENCHMARK_COLD_L2 && arg_size <= ideal_arg_size
      ? int(ideal_arg_size / arg_size) + 1 : 1;

  // Allocate buffer groups
  std::vector<__nv_fp4x2_e2m1*> blocks_A(arg_group_count);
  std::vector<__nv_fp4x2_e2m1*> blocks_B(arg_group_count);
  std::vector<__nv_fp8_e4m3*> blocks_A_scale(arg_group_count);
  std::vector<__nv_fp8_e4m3*> blocks_B_scale(arg_group_count);
  std::vector<__nv_bfloat16*> blocks_D(arg_group_count);
  std::vector<float> alphas(arg_group_count);
  uint64_t seed = BENCHMARK_SEED;

  for (int i = 0; i < arg_group_count; ++i) {
    CHECK_CUDA(cudaMalloc(&blocks_A[i], size_A_packed * sizeof(__nv_fp4x2_e2m1)));
    CHECK_CUDA(cudaMalloc(&blocks_B[i], size_B_packed * sizeof(__nv_fp4x2_e2m1)));
    CHECK_CUDA(cudaMalloc(&blocks_A_scale[i], size_A_scale * sizeof(__nv_fp8_e4m3)));
    CHECK_CUDA(cudaMalloc(&blocks_B_scale[i], size_B_scale * sizeof(__nv_fp8_e4m3)));
    CHECK_CUDA(cudaMalloc(&blocks_D[i], size_D * sizeof(__nv_bfloat16)));

    fill_uniform(reinterpret_cast<uint8_t*>(blocks_A[i]), size_A_packed,
                 seed + i * 100, BENCHMARK_NVFP4_PACKED_LOW,
                 BENCHMARK_NVFP4_PACKED_HIGH);
    fill_uniform(reinterpret_cast<uint8_t*>(blocks_B[i]), size_B_packed,
                 seed + i * 100 + 1, BENCHMARK_NVFP4_PACKED_LOW,
                 BENCHMARK_NVFP4_PACKED_HIGH);
    fill_uniform(blocks_A_scale[i], size_A_scale, seed + i * 100 + 2,
                 BENCHMARK_NVFP4_SCALE_LOW, BENCHMARK_NVFP4_SCALE_HIGH);
    fill_uniform(blocks_B_scale[i], size_B_scale, seed + i * 100 + 3,
                 BENCHMARK_NVFP4_SCALE_LOW, BENCHMARK_NVFP4_SCALE_HIGH);
    float a_global = uniform_value(seed + i * 100 + 4,
                                   BENCHMARK_NVFP4_SCALE_LOW,
                                   BENCHMARK_NVFP4_SCALE_HIGH);
    float b_global = uniform_value(seed + i * 100 + 5,
                                   BENCHMARK_NVFP4_SCALE_LOW,
                                   BENCHMARK_NVFP4_SCALE_HIGH);
    alphas[i] = a_global * b_global;
    fill_zero(blocks_D[i], size_D);
  }
  CHECK_CUDA(cudaDeviceSynchronize());

  // Initialize cuBLASLt
  CublasLtNvfp4Gemm gemm;
  gemm.init(M, N, K);
  if (tune) {
    gemm.autotune(blocks_A[0], blocks_B[0], blocks_A_scale[0], blocks_B_scale[0],
                  alphas[0], blocks_D[0]);
  } else {
    gemm.select(M);
  }

  cudaStream_t stream;
  CHECK_CUDA(cudaStreamCreate(&stream));

  cudaEvent_t start, stop;
  CHECK_CUDA(cudaEventCreate(&start));
  CHECK_CUDA(cudaEventCreate(&stop));
  int64_t flops = int64_t(2) * M * N * K;
  for (int i = 0; i < BENCHMARK_WARMUPS; ++i) {
    int j = i % arg_group_count;
    gemm.run(blocks_A[j], blocks_B[j], blocks_A_scale[j], blocks_B_scale[j],
             alphas[j], blocks_D[j], stream);
  }
  CHECK_CUDA(cudaStreamSynchronize(stream));
  CHECK_CUDA(cudaEventRecord(start, stream));
  for (int i = 0; i < BENCHMARK_ITERATIONS; ++i) {
    int j = i % arg_group_count;
    gemm.run(blocks_A[j], blocks_B[j], blocks_A_scale[j], blocks_B_scale[j],
             alphas[j], blocks_D[j], stream);
  }
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
    CHECK_CUDA(cudaFree(blocks_A_scale[i]));
    CHECK_CUDA(cudaFree(blocks_B_scale[i]));
    CHECK_CUDA(cudaFree(blocks_D[i]));
  }
}

///////////////////////////////////////////////////////////////////////////////////////////////////

int main(int argc, char** argv) {
  const int size = argc > 1 ? atoi(argv[1]) : 0;
  bool tune = argc > 2 && std::string(argv[2]) == "--tune";

  if (size != 0 && !benchmark_size(size)) {
    std::cerr << "Unsupported SIZE=" << size << "; expected one of";
    print_benchmark_sizes(std::cerr);
    std::cerr << std::endl;
    return 2;
  }

  std::cout << "cuBLASLt NVFP4 GEMM Profiler" << std::endl;
  std::cout << "D = A * B, A: RowMajor (MxK), B: ColMajor (NxK), D: RowMajor (MxN)" << std::endl;
  std::cout << "D[m,n] = sum_k A[m,k] * B[n,k]" << std::endl;
  std::cout << "Input: FP4 E2M1 + UE4M3 block scales (16-element blocks along K) + FP32 per-tensor scale" << std::endl;
  std::cout << "Accumulator: FP32, Output: BF16" << std::endl;
  std::cout << "Warmup: " << BENCHMARK_WARMUPS << ", Profiling: " << BENCHMARK_ITERATIONS << std::endl;

  for (int n : benchmark_sizes) if (size == 0 || size == n) benchmark(n, n, n, tune);

  return 0;
}
