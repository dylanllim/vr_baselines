#pragma once

#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <thread>
#include <cublasLt.h>
#include <cuda_fp4.h>
#include <cuda_runtime.h>
#include ".protocol.cuh"

static constexpr int benchmark_sizes[] = BENCHMARK_SIZES;

static inline bool benchmark_size(int value) {
  for (int size : benchmark_sizes) if (size == value) return true;
  return false;
}

static inline void print_benchmark_sizes(std::ostream& out) {
  for (int size : benchmark_sizes) out << ' ' << size;
}

#define CHECK_CUDA(call) do { \
  cudaError_t status = (call); \
  if (status != cudaSuccess) { \
    std::cerr << cudaGetErrorString(status) << '\n'; \
    std::exit(EXIT_FAILURE); \
  } \
} while (0)

#define CHECK_CUBLAS(call) do { \
  cublasStatus_t status = (call); \
  if (status != CUBLAS_STATUS_SUCCESS) { \
    std::cerr << "cuBLASLt status " << status << '\n'; \
    std::exit(EXIT_FAILURE); \
  } \
} while (0)

static inline void sleep_ms(int milliseconds) {
  std::this_thread::sleep_for(std::chrono::milliseconds(milliseconds));
}

static inline float uniform_value(uint64_t seed, float low, float high) {
  uint64_t x = seed;
  x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
  x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
  x ^= x >> 31;
  return low + float(x >> 40) * ((high - low) / 16777216.0f);
}

template <typename T>
__global__ void fill_kernel(
    T* data, size_t count, uint64_t seed, float low, float high) {
  size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= count) return;
  uint64_t x = seed + i;
  x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
  x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
  x ^= x >> 31;
  float u = float(x >> 40) * (1.0f / 16777216.0f);
  data[i] = static_cast<T>(low + u * (high - low));
}

template <typename T>
static inline void fill_uniform(
    T* data, size_t count, uint64_t seed, float low, float high) {
  fill_kernel<<<(count + 255) / 256, 256>>>(data, count, seed, low, high);
}

template <typename T>
static inline void fill_zero(T* data, size_t count) {
  cudaMemset(data, 0, count * sizeof(T));
}
