/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2025 NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: Apache-2.0
 * Adapted from NVIDIA CUDALibrarySamples/cuBLASLt/Common/LtMatmulCustomFind.h.
 */
#pragma once

#include "common.cuh"

#include <array>
#include <limits>

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
      plan.input_type, plan.input_type, CUDA_R_16BF, CUDA_R_16BF,
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
      plan.output_layout, plan.output_layout, &algorithm, &result));
  result.algo = algorithm;
  return result;
}

template <class Launch>
cublasLtMatmulHeuristicResult_t find_fastest_algorithm(
    LtPlan const& plan, int groups, Launch launch) {
  constexpr int kCandidateCount = 32;
  std::array<cublasLtMatmulHeuristicResult_t, kCandidateCount> candidates{};
  int count = 0;
  CHECK_CUBLAS(cublasLtMatmulAlgoGetHeuristic(
      plan.handle, plan.operation, plan.a_layout, plan.b_layout,
      plan.output_layout, plan.output_layout, plan.preference,
      kCandidateCount, candidates.data(), &count));
  if (count == 0) {
    std::cerr << "cuBLASLt returned no algorithms\n";
    std::exit(EXIT_FAILURE);
  }

  cudaStream_t stream;
  cudaEvent_t start, stop;
  CHECK_CUDA(cudaStreamCreate(&stream));
  CHECK_CUDA(cudaEventCreate(&start));
  CHECK_CUDA(cudaEventCreate(&stop));

  int best = -1;
  float best_ms = std::numeric_limits<float>::infinity();
  for (int rank = 0; rank < count; ++rank) {
    auto& candidate = candidates[rank];
    if (candidate.state != CUBLAS_STATUS_SUCCESS ||
        candidate.workspaceSize > kWorkspaceBytes) continue;

    cublasStatus_t status = CUBLAS_STATUS_SUCCESS;
    for (int i = 0; i < BENCHMARK_WARMUPS && status == CUBLAS_STATUS_SUCCESS; ++i)
      status = launch(candidate.algo, i % groups, stream);
    if (status != CUBLAS_STATUS_SUCCESS || cudaStreamSynchronize(stream) != cudaSuccess)
      continue;

    CHECK_CUDA(cudaEventRecord(start, stream));
    for (int i = 0; i < BENCHMARK_ITERATIONS && status == CUBLAS_STATUS_SUCCESS; ++i)
      status = launch(candidate.algo, i % groups, stream);
    CHECK_CUDA(cudaEventRecord(stop, stream));
    if (status != CUBLAS_STATUS_SUCCESS || cudaEventSynchronize(stop) != cudaSuccess)
      continue;

    float elapsed = 0;
    CHECK_CUDA(cudaEventElapsedTime(&elapsed, start, stop));
    const float milliseconds = elapsed / BENCHMARK_ITERATIONS;
    if (milliseconds < best_ms) {
      best = rank;
      best_ms = milliseconds;
    }
  }

  CHECK_CUDA(cudaEventDestroy(start));
  CHECK_CUDA(cudaEventDestroy(stop));
  CHECK_CUDA(cudaStreamDestroy(stream));
  if (best < 0) {
    std::cerr << "no cuBLASLt candidate ran successfully\n";
    std::exit(EXIT_FAILURE);
  }

  print_selected(candidates[best], best_ms);
  return candidates[best];
}
