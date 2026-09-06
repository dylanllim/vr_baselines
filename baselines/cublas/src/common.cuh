#pragma once

#include <cublasLt.h>
#include <cuda_runtime.h>

#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <string_view>
#include <thread>
#include <vector>

#include "protocol.cuh"

#define CHECK_CUDA(call) do {                                                   \
  cudaError_t status = (call);                                                  \
  if (status != cudaSuccess) {                                                  \
    std::cerr << cudaGetErrorString(status) << '\n';                            \
    std::exit(EXIT_FAILURE);                                                    \
  }                                                                             \
} while (0)

#define CHECK_CUBLAS(call) do {                                                 \
  cublasStatus_t status = (call);                                               \
  if (status != CUBLAS_STATUS_SUCCESS) {                                        \
    std::cerr << "cuBLASLt status " << status << '\n';                         \
    std::exit(EXIT_FAILURE);                                                    \
  }                                                                             \
} while (0)

inline constexpr int kBenchmarkSizes[] = BENCHMARK_SIZES;
inline constexpr size_t kWorkspaceBytes = size_t{512} << 20;

inline bool valid_size(int value) {
  for (int size : kBenchmarkSizes) if (size == value) return true;
  return false;
}

inline int cold_buffer_groups(size_t bytes_per_group) {
  if (!BENCHMARK_COLD_L2) return 1;
  int l2_bytes = 0;
  CHECK_CUDA(cudaDeviceGetAttribute(&l2_bytes, cudaDevAttrL2CacheSize, 0));
  const size_t target = 3 * size_t(l2_bytes);
  return bytes_per_group > target ? 1 : int(target / bytes_per_group) + 1;
}

template <class T>
class DeviceGroups {
 public:
  DeviceGroups(int groups, size_t elements) : pointers_(groups), elements_(elements) {
    for (T*& pointer : pointers_)
      CHECK_CUDA(cudaMalloc(&pointer, elements_ * sizeof(T)));
  }

  ~DeviceGroups() {
    for (T* pointer : pointers_) CHECK_CUDA(cudaFree(pointer));
  }

  DeviceGroups(DeviceGroups const&) = delete;
  DeviceGroups& operator=(DeviceGroups const&) = delete;

  T* operator[](int group) { return pointers_[group]; }
  T const* operator[](int group) const { return pointers_[group]; }
  int size() const { return int(pointers_.size()); }
  size_t elements() const { return elements_; }

 private:
  std::vector<T*> pointers_;
  size_t elements_;
};

__host__ __device__ inline float uniform_value(
    uint64_t seed, float low, float high) {
  uint64_t x = seed;
  x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
  x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
  x ^= x >> 31;
  return low + float(x >> 40) * ((high - low) / 16777216.0f);
}

template <class T>
__global__ void fill_uniform_kernel(
    T* data, size_t count, uint64_t seed, float low, float high) {
  size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= count) return;
  data[i] = static_cast<T>(uniform_value(seed + i, low, high));
}

template <class T>
inline void fill_uniform(
    T* data, size_t count, uint64_t seed, float low, float high) {
  fill_uniform_kernel<<<(count + 255) / 256, 256>>>(data, count, seed, low, high);
}

template <class T>
inline void fill_zero(T* data, size_t count) {
  CHECK_CUDA(cudaMemset(data, 0, count * sizeof(T)));
}

class LtPlan {
 public:
  LtPlan(int size, cudaDataType_t input_type) : input_type(input_type) {
    CHECK_CUBLAS(cublasLtCreate(&handle));
    CHECK_CUBLAS(cublasLtMatmulDescCreate(
        &operation, CUBLAS_COMPUTE_32F, CUDA_R_32F));

    cublasOperation_t trans_a = CUBLAS_OP_T;
    cublasOperation_t trans_b = CUBLAS_OP_N;
    CHECK_CUBLAS(cublasLtMatmulDescSetAttribute(
        operation, CUBLASLT_MATMUL_DESC_TRANSA, &trans_a, sizeof(trans_a)));
    CHECK_CUBLAS(cublasLtMatmulDescSetAttribute(
        operation, CUBLASLT_MATMUL_DESC_TRANSB, &trans_b, sizeof(trans_b)));

    // The transpose trick presents logical B as cuBLASLt A and logical A as B.
    CHECK_CUBLAS(cublasLtMatrixLayoutCreate(
        &a_layout, input_type, size, size, size));
    CHECK_CUBLAS(cublasLtMatrixLayoutCreate(
        &b_layout, input_type, size, size, size));
    CHECK_CUBLAS(cublasLtMatrixLayoutCreate(
        &output_layout, CUDA_R_16BF, size, size, size));

    CHECK_CUDA(cudaMalloc(&workspace, kWorkspaceBytes));
    CHECK_CUBLAS(cublasLtMatmulPreferenceCreate(&preference));
    CHECK_CUBLAS(cublasLtMatmulPreferenceSetAttribute(
        preference, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
        &kWorkspaceBytes, sizeof(kWorkspaceBytes)));
  }

  ~LtPlan() {
    CHECK_CUBLAS(cublasLtMatmulPreferenceDestroy(preference));
    CHECK_CUBLAS(cublasLtMatrixLayoutDestroy(a_layout));
    CHECK_CUBLAS(cublasLtMatrixLayoutDestroy(b_layout));
    CHECK_CUBLAS(cublasLtMatrixLayoutDestroy(output_layout));
    CHECK_CUBLAS(cublasLtMatmulDescDestroy(operation));
    CHECK_CUBLAS(cublasLtDestroy(handle));
    CHECK_CUDA(cudaFree(workspace));
  }

  void set_scale_pointers(void const* a_scale, void const* b_scale) {
    CHECK_CUBLAS(cublasLtMatmulDescSetAttribute(
        operation, CUBLASLT_MATMUL_DESC_A_SCALE_POINTER,
        &a_scale, sizeof(a_scale)));
    CHECK_CUBLAS(cublasLtMatmulDescSetAttribute(
        operation, CUBLASLT_MATMUL_DESC_B_SCALE_POINTER,
        &b_scale, sizeof(b_scale)));
  }

  cublasLtHandle_t handle{};
  cublasLtMatmulDesc_t operation{};
  cublasLtMatrixLayout_t a_layout{}, b_layout{}, output_layout{};
  cublasLtMatmulPreference_t preference{};
  cublasLtMatmulHeuristicResult_t selected{};
  void* workspace{};
  cudaDataType_t input_type;
};

struct Measurement {
  double milliseconds;
  double tflops;
};

template <class Workload>
Measurement measure(int size, Workload& workload) {
  cudaStream_t stream;
  cudaEvent_t start, stop;
  CHECK_CUDA(cudaStreamCreate(&stream));
  CHECK_CUDA(cudaEventCreate(&start));
  CHECK_CUDA(cudaEventCreate(&stop));

  for (int i = 0; i < BENCHMARK_WARMUPS; ++i)
    workload.run(i % workload.groups(), stream);
  CHECK_CUDA(cudaStreamSynchronize(stream));

  CHECK_CUDA(cudaEventRecord(start, stream));
  for (int i = 0; i < BENCHMARK_ITERATIONS; ++i)
    workload.run(i % workload.groups(), stream);
  CHECK_CUDA(cudaEventRecord(stop, stream));
  CHECK_CUDA(cudaEventSynchronize(stop));

  float elapsed = 0;
  CHECK_CUDA(cudaEventElapsedTime(&elapsed, start, stop));
  CHECK_CUDA(cudaEventDestroy(start));
  CHECK_CUDA(cudaEventDestroy(stop));
  CHECK_CUDA(cudaStreamDestroy(stream));

  const double milliseconds = elapsed / BENCHMARK_ITERATIONS;
  const double flops = 2.0 * size * size * size;
  return {milliseconds, flops / (milliseconds * 1e9)};
}

template <class Workload>
int benchmark_main(int argc, char** argv) {
  if (argc < 2 || argc > 3 || (argc == 3 && std::string_view(argv[2]) != "--tune")) {
    std::cerr << "usage: " << argv[0] << " SIZE [--tune]\n";
    return 2;
  }

  const int size = std::atoi(argv[1]);
  if (!valid_size(size)) {
    std::cerr << "unsupported size " << argv[1] << '\n';
    return 2;
  }

  std::this_thread::sleep_for(std::chrono::milliseconds(500));
  Workload workload(size);
  workload.select(argc == 3);
  Measurement result = measure(size, workload);
  std::cout << std::fixed << std::setprecision(6)
            << "RESULT size=" << size
            << " runtime_ms=" << result.milliseconds
            << " tflops=" << result.tflops << '\n';
  return 0;
}
