/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2025 NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: Apache-2.0
 * Adapted from NVIDIA CUDALibrarySamples/cuBLASLt/Common/LtMatmulCustomFind.h.
 */
#pragma once

#include <cublasLt.h>
#include <cuda_runtime.h>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <limits>

struct AlgoConfig {
  int size, id;
  uint32_t tile, stages, custom, swizzle, reduction;
  uint16_t inner, cluster;
  int32_t split_k;
};

static inline void print_algo(
    char const* label, int rank, cublasLtMatmulHeuristicResult_t const& result,
    float milliseconds);

template <size_t N>
static inline cublasLtMatmulHeuristicResult_t pinned_algo(
    AlgoConfig const (&configs)[N], int size, cublasLtHandle_t handle,
    cublasLtMatmulDesc_t operation, cublasLtMatrixLayout_t a_layout,
    cublasLtMatrixLayout_t b_layout, cublasLtMatrixLayout_t c_layout,
    cublasLtMatrixLayout_t d_layout, cudaDataType_t input_type) {
  AlgoConfig const* cfg = nullptr;
  for (auto const& candidate : configs) if (candidate.size == size) cfg = &candidate;
  if (!cfg) {
    std::cerr << "No pinned cuBLASLt algorithm for size " << size << '\n';
    std::exit(EXIT_FAILURE);
  }
  cublasLtMatmulAlgo_t algo{};
  CHECK_CUBLAS(cublasLtMatmulAlgoInit(
      handle, CUBLAS_COMPUTE_32F, CUDA_R_32F, input_type, input_type,
      CUDA_R_16BF, CUDA_R_16BF, cfg->id, &algo));
#define SET_ALGO(attribute, member) \
  CHECK_CUBLAS(cublasLtMatmulAlgoConfigSetAttribute( \
      &algo, attribute, &cfg->member, sizeof(cfg->member)))
  SET_ALGO(CUBLASLT_ALGO_CONFIG_TILE_ID, tile);
  SET_ALGO(CUBLASLT_ALGO_CONFIG_STAGES_ID, stages);
  SET_ALGO(CUBLASLT_ALGO_CONFIG_INNER_SHAPE_ID, inner);
  SET_ALGO(CUBLASLT_ALGO_CONFIG_CLUSTER_SHAPE_ID, cluster);
  SET_ALGO(CUBLASLT_ALGO_CONFIG_CUSTOM_OPTION, custom);
  SET_ALGO(CUBLASLT_ALGO_CONFIG_CTA_SWIZZLING, swizzle);
  SET_ALGO(CUBLASLT_ALGO_CONFIG_SPLITK_NUM, split_k);
  SET_ALGO(CUBLASLT_ALGO_CONFIG_REDUCTION_SCHEME, reduction);
#undef SET_ALGO
  cublasLtMatmulHeuristicResult_t result{};
  CHECK_CUBLAS(cublasLtMatmulAlgoCheck(
      handle, operation, a_layout, b_layout, c_layout, d_layout, &algo, &result));
  result.algo = algo;
  print_algo("PINNED", 0, result, 0);
  return result;
}

static inline int config(
    cublasLtMatmulAlgo_t const& algo,
    cublasLtMatmulAlgoConfigAttributes_t attribute) {
  int value = -1;
  size_t written = 0;
  cublasLtMatmulAlgoConfigGetAttribute(
      &algo, attribute, &value, sizeof(value), &written);
  return value;
}

static inline int cluster(cublasLtMatmulAlgo_t const& algo) {
  uint16_t value = 0;
  size_t written = 0;
  cublasLtMatmulAlgoConfigGetAttribute(
      &algo, CUBLASLT_ALGO_CONFIG_CLUSTER_SHAPE_ID,
      &value, sizeof(value), &written);
  return value;
}

static inline void print_algo(
    char const* label, int rank, cublasLtMatmulHeuristicResult_t const& result,
    float milliseconds) {
  auto const& algo = result.algo;
  std::cout << label << " rank=" << rank
            << " algo=" << config(algo, CUBLASLT_ALGO_CONFIG_ID)
            << " tile=" << config(algo, CUBLASLT_ALGO_CONFIG_TILE_ID)
            << " stages=" << config(algo, CUBLASLT_ALGO_CONFIG_STAGES_ID)
            << " cluster=" << cluster(algo)
            << " custom=" << config(algo, CUBLASLT_ALGO_CONFIG_CUSTOM_OPTION)
            << " workspace=" << result.workspaceSize
            << " ms=" << std::fixed << std::setprecision(6) << milliseconds
            << std::defaultfloat << '\n';
}

static inline cublasLtMatmulHeuristicResult_t autotune(
    cublasLtHandle_t handle, cublasLtMatmulDesc_t operation,
    cublasLtMatrixLayout_t a_layout, cublasLtMatrixLayout_t b_layout,
    cublasLtMatrixLayout_t c_layout, cublasLtMatrixLayout_t d_layout,
    cublasLtMatmulPreference_t preference, void const* alpha, void const* a,
    void const* b, void const* beta, void const* c, void* d, void* workspace,
    size_t workspace_size) {
  constexpr int count = 32;
  cublasLtMatmulHeuristicResult_t candidates[count]{};
  int returned = 0;
  if (cublasLtMatmulAlgoGetHeuristic(
          handle, operation, a_layout, b_layout, c_layout, d_layout,
          preference, count, candidates, &returned) != CUBLAS_STATUS_SUCCESS ||
      returned == 0) {
    std::cerr << "cuBLASLt returned no algorithms\n";
    std::exit(EXIT_FAILURE);
  }

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  int best = -1;
  float best_ms = std::numeric_limits<float>::infinity();
  std::cout << "AUTOTUNE requested=32 returned=" << returned << '\n';
  for (int rank = 0; rank < returned; ++rank) {
    auto& candidate = candidates[rank];
    if (candidate.state != CUBLAS_STATUS_SUCCESS ||
        candidate.workspaceSize > workspace_size) continue;
    cublasStatus_t status = CUBLAS_STATUS_SUCCESS;
    for (int i = 0; i < BENCHMARK_WARMUPS && status == CUBLAS_STATUS_SUCCESS; ++i)
      status = cublasLtMatmul(
          handle, operation, alpha, a, a_layout, b, b_layout, beta, c,
          c_layout, d, d_layout, &candidate.algo, workspace, workspace_size, 0);
    if (status != CUBLAS_STATUS_SUCCESS || cudaDeviceSynchronize() != cudaSuccess)
      continue;
    cudaEventRecord(start);
    for (int i = 0; i < BENCHMARK_ITERATIONS && status == CUBLAS_STATUS_SUCCESS; ++i)
      status = cublasLtMatmul(
          handle, operation, alpha, a, a_layout, b, b_layout, beta, c,
          c_layout, d, d_layout, &candidate.algo, workspace, workspace_size, 0);
    cudaEventRecord(stop);
    if (status != CUBLAS_STATUS_SUCCESS || cudaEventSynchronize(stop) != cudaSuccess)
      continue;
    float elapsed = 0;
    cudaEventElapsedTime(&elapsed, start, stop);
    float milliseconds = elapsed / BENCHMARK_ITERATIONS;
    print_algo("CANDIDATE", rank, candidate, milliseconds);
    if (milliseconds < best_ms) {
      best = rank;
      best_ms = milliseconds;
    }
  }
  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  if (best < 0) {
    std::cerr << "No cuBLASLt candidate ran successfully\n";
    std::exit(EXIT_FAILURE);
  }
  print_algo("SELECTED", best, candidates[best], best_ms);
  return candidates[best];
}
