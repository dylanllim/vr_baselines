/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2025 NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: Apache-2.0
 * Adapted from NVIDIA CUDALibrarySamples/cuBLASLt/Common/LtMatmulCustomFind.h.
 */
#pragma once

#include "common.cuh"

#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

struct AlgorithmConfig {
  int size;
  int id = 66;
  uint32_t tile = 0;
  uint32_t stages = 0;
  uint32_t custom = 0;
  uint32_t swizzle = 0;
  uint32_t reduction = 0;
  uint16_t inner = 0;
  uint16_t cluster = 0;
  int32_t split_k = 1;
};

template <class T>
inline T algorithm_attribute(
    cublasLtMatmulAlgo_t const& algorithm,
    cublasLtMatmulAlgoConfigAttributes_t attribute) {
  T value{};
  size_t written = 0;
  CHECK_CUBLAS(cublasLtMatmulAlgoConfigGetAttribute(
      &algorithm, attribute, &value, sizeof(value), &written));
  return value;
}

inline void print_selected(
    cublasLtMatmulHeuristicResult_t const& result, double milliseconds) {
  auto const& algorithm = result.algo;
  std::cout << std::fixed << std::setprecision(6)
            << "ALGORITHM"
            << " id=" << algorithm_attribute<int>(algorithm, CUBLASLT_ALGO_CONFIG_ID)
            << " tile=" << algorithm_attribute<uint32_t>(algorithm, CUBLASLT_ALGO_CONFIG_TILE_ID)
            << " stages=" << algorithm_attribute<uint32_t>(algorithm, CUBLASLT_ALGO_CONFIG_STAGES_ID)
            << " inner=" << algorithm_attribute<uint16_t>(algorithm, CUBLASLT_ALGO_CONFIG_INNER_SHAPE_ID)
            << " cluster=" << algorithm_attribute<uint16_t>(algorithm, CUBLASLT_ALGO_CONFIG_CLUSTER_SHAPE_ID)
            << " custom=" << algorithm_attribute<uint32_t>(algorithm, CUBLASLT_ALGO_CONFIG_CUSTOM_OPTION)
            << " swizzle=" << algorithm_attribute<uint32_t>(algorithm, CUBLASLT_ALGO_CONFIG_CTA_SWIZZLING)
            << " split_k=" << algorithm_attribute<int32_t>(algorithm, CUBLASLT_ALGO_CONFIG_SPLITK_NUM)
            << " reduction=" << algorithm_attribute<uint32_t>(algorithm, CUBLASLT_ALGO_CONFIG_REDUCTION_SCHEME)
            << " workspace=" << result.workspaceSize
            << " runtime_ms=" << milliseconds << '\n';
}

template <size_t N>
cublasLtMatmulHeuristicResult_t pinned_algorithm(
    AlgorithmConfig const (&configs)[N], int size, LtPlan const& plan) {
  AlgorithmConfig const* config = nullptr;
  for (auto const& candidate : configs)
    if (candidate.size == size) config = &candidate;
  if (!config) {
    std::cerr << "no pinned algorithm for size " << size << '\n';
    std::exit(EXIT_FAILURE);
  }

  cublasLtMatmulAlgo_t algorithm{};
  CHECK_CUBLAS(cublasLtMatmulAlgoInit(
      plan.handle, CUBLAS_COMPUTE_32F, CUDA_R_32F,
      plan.input_type, plan.input_type, plan.c_type, plan.d_type,
      config->id, &algorithm));

#define SET_ALGORITHM(attribute, member)                                        \
  CHECK_CUBLAS(cublasLtMatmulAlgoConfigSetAttribute(                            \
      &algorithm, attribute, &config->member, sizeof(config->member)))
  SET_ALGORITHM(CUBLASLT_ALGO_CONFIG_TILE_ID, tile);
  SET_ALGORITHM(CUBLASLT_ALGO_CONFIG_STAGES_ID, stages);
  SET_ALGORITHM(CUBLASLT_ALGO_CONFIG_INNER_SHAPE_ID, inner);
  SET_ALGORITHM(CUBLASLT_ALGO_CONFIG_CLUSTER_SHAPE_ID, cluster);
  SET_ALGORITHM(CUBLASLT_ALGO_CONFIG_CUSTOM_OPTION, custom);
  SET_ALGORITHM(CUBLASLT_ALGO_CONFIG_CTA_SWIZZLING, swizzle);
  SET_ALGORITHM(CUBLASLT_ALGO_CONFIG_SPLITK_NUM, split_k);
  SET_ALGORITHM(CUBLASLT_ALGO_CONFIG_REDUCTION_SCHEME, reduction);
#undef SET_ALGORITHM

  cublasLtMatmulHeuristicResult_t result{};
  CHECK_CUBLAS(cublasLtMatmulAlgoCheck(
      plan.handle, plan.operation, plan.a_layout, plan.b_layout,
      plan.c_layout, plan.d_layout, &algorithm, &result));
  if (result.state != CUBLAS_STATUS_SUCCESS ||
      result.workspaceSize > kWorkspaceBytes) {
    std::cerr << "pinned algorithm is unsupported or requires too much workspace\n";
    std::exit(EXIT_FAILURE);
  }
  result.algo = algorithm;
  return result;
}

template <class Launch>
float time_algorithm(
    cublasLtMatmulAlgo_t const& algorithm, int groups, Launch launch,
    cudaStream_t stream, cudaEvent_t start, cudaEvent_t stop,
    int warmups = BENCHMARK_WARMUPS, int iterations = BENCHMARK_ITERATIONS) {
  cublasStatus_t status = CUBLAS_STATUS_SUCCESS;
  for (int i = 0; i < warmups && status == CUBLAS_STATUS_SUCCESS; ++i)
    status = launch(algorithm, i % groups, stream);
  if (status != CUBLAS_STATUS_SUCCESS || cudaStreamSynchronize(stream) != cudaSuccess) {
    cudaGetLastError();
    return std::numeric_limits<float>::infinity();
  }
  CHECK_CUDA(cudaEventRecord(start, stream));
  for (int i = 0; i < iterations && status == CUBLAS_STATUS_SUCCESS; ++i)
    status = launch(algorithm, (warmups + i) % groups, stream);
  CHECK_CUDA(cudaEventRecord(stop, stream));
  if (status != CUBLAS_STATUS_SUCCESS || cudaEventSynchronize(stop) != cudaSuccess) {
    cudaGetLastError();
    return std::numeric_limits<float>::infinity();
  }
  float elapsed = 0;
  CHECK_CUDA(cudaEventElapsedTime(&elapsed, start, stop));
  return elapsed / iterations;
}

template <class T>
bool algorithm_capability(
    cublasLtMatmulAlgo_t const& algorithm,
    cublasLtMatmulAlgoCapAttributes_t attribute, T& value) {
  size_t written = 0;
  return cublasLtMatmulAlgoCapGetAttribute(
             &algorithm, attribute, &value, sizeof(value), &written) ==
             CUBLAS_STATUS_SUCCESS && written == sizeof(value);
}

inline std::vector<int> algorithm_capability_list(
    cublasLtMatmulAlgo_t const& algorithm,
    cublasLtMatmulAlgoCapAttributes_t attribute, int fallback) {
  size_t bytes = 0;
  if (cublasLtMatmulAlgoCapGetAttribute(
          &algorithm, attribute, nullptr, 0, &bytes) != CUBLAS_STATUS_SUCCESS ||
      bytes == 0)
    return {fallback};
  std::vector<int> values(bytes / sizeof(int));
  if (cublasLtMatmulAlgoCapGetAttribute(
          &algorithm, attribute, values.data(), bytes, &bytes) !=
      CUBLAS_STATUS_SUCCESS)
    return {fallback};
  values.resize(bytes / sizeof(int));
  return values.empty() ? std::vector<int>{fallback} : values;
}

template <class T>
bool set_algorithm(
    cublasLtMatmulAlgo_t& algorithm,
    cublasLtMatmulAlgoConfigAttributes_t attribute, T value) {
  return cublasLtMatmulAlgoConfigSetAttribute(
             &algorithm, attribute, &value, sizeof(value)) == CUBLAS_STATUS_SUCCESS;
}

template <class Launch>
cublasLtMatmulHeuristicResult_t find_fastest_algorithm(
    LtPlan const& plan, int groups,
    cublasLtMatmulHeuristicResult_t const& incumbent, Launch launch) {
  constexpr int kIdCapacity = 8192;
  std::vector<int> ids(kIdCapacity);
  int id_count = 0;
  CHECK_CUBLAS(cublasLtMatmulAlgoGetIds(
      plan.handle, CUBLAS_COMPUTE_32F, CUDA_R_32F, plan.input_type,
      plan.input_type, plan.c_type, plan.d_type,
      kIdCapacity, ids.data(), &id_count));
  if (id_count <= 0 || id_count == kIdCapacity) {
    std::cerr << "cuBLASLt returned no algorithm IDs or truncated the result\n";
    std::exit(EXIT_FAILURE);
  }
  ids.resize(id_count);

  std::vector<uint16_t> clusters{CUBLASLT_CLUSTER_SHAPE_AUTO};
  int device = 0, cluster_launch = 0;
  CHECK_CUDA(cudaGetDevice(&device));
  CHECK_CUDA(cudaDeviceGetAttribute(&cluster_launch, cudaDevAttrClusterLaunch, device));
  if (cluster_launch)
    for (uint16_t value = CUBLASLT_CLUSTER_SHAPE_1x1x1;
         value < CUBLASLT_CLUSTER_SHAPE_END; ++value)
      clusters.push_back(value);
  std::vector<uint16_t> inner_shapes;
  for (uint16_t value = CUBLASLT_MATMUL_INNER_SHAPE_UNDEFINED;
       value < CUBLASLT_MATMUL_INNER_SHAPE_END; ++value)
    inner_shapes.push_back(value);

  cudaStream_t stream;
  cudaEvent_t start, stop;
  CHECK_CUDA(cudaStreamCreate(&stream));
  CHECK_CUDA(cudaEventCreate(&start));
  CHECK_CUDA(cudaEventCreate(&stop));
  struct Candidate {
    cublasLtMatmulHeuristicResult_t result;
    float milliseconds;
  };
  std::vector<Candidate> candidates;
  size_t checked_count = 0;
  constexpr int32_t split_counts[] = {2, 3, 4, 5, 6, 8, 12, 16, 32};

  for (int id : ids) {
    cublasLtMatmulAlgo_t base{};
    if (cublasLtMatmulAlgoInit(
            plan.handle, CUBLAS_COMPUTE_32F, CUDA_R_32F,
            plan.input_type, plan.input_type, plan.c_type, plan.d_type,
            id, &base) != CUBLAS_STATUS_SUCCESS)
      continue;
    auto tiles = algorithm_capability_list(
        base, CUBLASLT_ALGO_CAP_TILE_IDS, CUBLASLT_MATMUL_TILE_UNDEFINED);
    auto stages = algorithm_capability_list(
        base, CUBLASLT_ALGO_CAP_STAGES_IDS, CUBLASLT_MATMUL_STAGES_UNDEFINED);
    int32_t split_support = 0;
    uint32_t reduction_mask = 0, max_swizzle = 0, max_custom = 0;
    algorithm_capability(base, CUBLASLT_ALGO_CAP_SPLITK_SUPPORT, split_support);
    algorithm_capability(base, CUBLASLT_ALGO_CAP_REDUCTION_SCHEME_MASK, reduction_mask);
    algorithm_capability(base, CUBLASLT_ALGO_CAP_CTA_SWIZZLING_SUPPORT, max_swizzle);
    algorithm_capability(base, CUBLASLT_ALGO_CAP_CUSTOM_OPTION_MAX, max_custom);
    struct SplitReduction { int32_t split; uint32_t reduction; };
    std::vector<SplitReduction> splits{{1, CUBLASLT_REDUCTION_SCHEME_NONE}};
    if (split_support)
      for (int32_t split : split_counts)
        for (uint32_t reduction = 1;
             reduction < CUBLASLT_REDUCTION_SCHEME_MASK; reduction <<= 1)
          if (reduction & reduction_mask) splits.push_back({split, reduction});

    for (int tile : tiles) for (int stage : stages)
    for (uint16_t inner : inner_shapes) for (uint16_t cluster : clusters)
    for (uint32_t custom = 0; custom <= max_custom; ++custom)
    for (uint32_t swizzle = 0; swizzle <= max_swizzle; ++swizzle)
    for (auto split : splits) {
      ++checked_count;
      cublasLtMatmulAlgo_t algorithm = base;
      if (!set_algorithm(algorithm, CUBLASLT_ALGO_CONFIG_TILE_ID,
                         static_cast<uint32_t>(tile)) ||
          !set_algorithm(algorithm, CUBLASLT_ALGO_CONFIG_STAGES_ID,
                         static_cast<uint32_t>(stage)) ||
          !set_algorithm(algorithm, CUBLASLT_ALGO_CONFIG_INNER_SHAPE_ID, inner) ||
          !set_algorithm(algorithm, CUBLASLT_ALGO_CONFIG_CLUSTER_SHAPE_ID, cluster) ||
          !set_algorithm(algorithm, CUBLASLT_ALGO_CONFIG_CUSTOM_OPTION, custom) ||
          !set_algorithm(algorithm, CUBLASLT_ALGO_CONFIG_CTA_SWIZZLING, swizzle) ||
          !set_algorithm(algorithm, CUBLASLT_ALGO_CONFIG_SPLITK_NUM, split.split) ||
          !set_algorithm(algorithm, CUBLASLT_ALGO_CONFIG_REDUCTION_SCHEME,
                         split.reduction))
        continue;
      cublasLtMatmulHeuristicResult_t result{};
      if (cublasLtMatmulAlgoCheck(
              plan.handle, plan.operation, plan.a_layout, plan.b_layout,
              plan.c_layout, plan.d_layout, &algorithm, &result) !=
              CUBLAS_STATUS_SUCCESS || result.state != CUBLAS_STATUS_SUCCESS ||
          result.workspaceSize > kWorkspaceBytes)
        continue;
      float milliseconds = time_algorithm(
          algorithm, groups, launch, stream, start, stop, 1, 3);
      if (std::isfinite(milliseconds)) {
        result.algo = algorithm;
        candidates.push_back({result, milliseconds});
      }
    }
  }

  std::sort(candidates.begin(), candidates.end(), [](auto const& a, auto const& b) {
    return a.milliseconds < b.milliseconds;
  });
  std::cout << "SEARCH algorithm_ids=" << ids.size()
            << " checked=" << checked_count
            << " supported=" << candidates.size()
            << " screen=1+3 finalists=100x5+10"
            << " top=10x3x5+10 incumbent=3x5+10\n";
  for (size_t i = 0; i < std::min<size_t>(100, candidates.size()); ++i)
    candidates[i].milliseconds = time_algorithm(
        candidates[i].result.algo, groups, launch, stream, start, stop);
  std::sort(candidates.begin(), candidates.begin() + std::min<size_t>(100, candidates.size()),
            [](auto const& a, auto const& b) {
              return a.milliseconds < b.milliseconds;
            });
  float best_ms = std::numeric_limits<float>::infinity();
  cublasLtMatmulHeuristicResult_t best{};
  auto rank = [&](cublasLtMatmulHeuristicResult_t const& result) {
    std::vector<float> samples;
    for (int repetition = 0; repetition < 3; ++repetition)
      samples.push_back(time_algorithm(
          result.algo, groups, launch, stream, start, stop));
    std::sort(samples.begin(), samples.end());
    if (samples[1] < best_ms) {
      best_ms = samples[1];
      best = result;
    }
  };
  for (size_t i = 0; i < std::min<size_t>(10, candidates.size()); ++i)
    rank(candidates[i].result);
  rank(incumbent);
  CHECK_CUDA(cudaEventDestroy(start));
  CHECK_CUDA(cudaEventDestroy(stop));
  CHECK_CUDA(cudaStreamDestroy(stream));
  if (!std::isfinite(best_ms)) {
    std::cerr << "no cuBLASLt configuration ran successfully\n";
    std::exit(EXIT_FAILURE);
  }
  print_selected(best, best_ms);
  return best;
}
