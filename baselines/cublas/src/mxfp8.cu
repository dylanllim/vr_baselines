#include <cuda_bf16.h>
#include <cuda_fp8.h>

#include "algorithms.cuh"

namespace {

constexpr AlgorithmConfig kWinners[] = {
    {.size = 1024,  .tile = 20,  .stages = 36, .custom = 1, .cluster = 3},
    {.size = 2048,  .tile = 23,  .stages = 36, .custom = 1, .cluster = 14},
    {.size = 4096,  .tile = 23,  .stages = 36, .custom = 1, .cluster = 7},
    {.size = 8192,  .tile = 513, .stages = 36, .custom = 3, .cluster = 14},
    {.size = 16384, .tile = 513, .stages = 36, .custom = 3, .cluster = 9},
    {.size = 32768, .tile = 513, .stages = 36, .custom = 2, .cluster = 9},
};

class Mxfp8Workload {
 public:
  explicit Mxfp8Workload(int size)
      : size_(size),
        groups_(cold_buffer_groups(bytes_per_group(size))),
        a_(groups_, elements(size)),
        b_(groups_, elements(size)),
        a_scales_(groups_, scale_elements(size)),
        b_scales_(groups_, scale_elements(size)),
        d_(groups_, elements(size)),
        plan_(size, CUDA_R_8F_E4M3) {
    cublasLtMatmulMatrixScale_t mode = CUBLASLT_MATMUL_MATRIX_SCALE_VEC32_UE8M0;
    CHECK_CUBLAS(cublasLtMatmulDescSetAttribute(
        plan_.operation, CUBLASLT_MATMUL_DESC_A_SCALE_MODE, &mode, sizeof(mode)));
    CHECK_CUBLAS(cublasLtMatmulDescSetAttribute(
        plan_.operation, CUBLASLT_MATMUL_DESC_B_SCALE_MODE, &mode, sizeof(mode)));

    for (int group = 0; group < groups_; ++group) {
      const uint64_t seed = BENCHMARK_SEED + 100 * group;
      fill_uniform(a_[group], a_.elements(), seed,
                   BENCHMARK_FP8_LOW, BENCHMARK_FP8_HIGH);
      fill_uniform(b_[group], b_.elements(), seed + 1,
                   BENCHMARK_FP8_LOW, BENCHMARK_FP8_HIGH);
      fill_uniform(a_scales_[group], a_scales_.elements(), seed + 2,
                   BENCHMARK_MXFP8_SCALE_LOW, BENCHMARK_MXFP8_SCALE_HIGH);
      fill_uniform(b_scales_[group], b_scales_.elements(), seed + 3,
                   BENCHMARK_MXFP8_SCALE_LOW, BENCHMARK_MXFP8_SCALE_HIGH);
      fill_zero(d_[group], d_.elements());
    }
    CHECK_CUDA(cudaDeviceSynchronize());
    plan_.set_scale_pointers(plan_.workspace, plan_.workspace);
  }

  int groups() const { return groups_; }

  void select(bool tune) {
    auto incumbent = pinned_algorithm(kWinners, size_, plan_);
    plan_.selected = tune
        ? find_fastest_algorithm(plan_, groups_, incumbent,
                                 [this](auto const& algorithm, int group,
                                        cudaStream_t stream) {
            return launch(algorithm, group, stream);
          })
        : incumbent;
  }

  void run(int group, cudaStream_t stream) {
    CHECK_CUBLAS(launch(plan_.selected.algo, group, stream));
  }

 private:
  static size_t elements(int size) { return size_t(size) * size; }
  static size_t scale_elements(int size) { return elements(size) / 32; }

  static size_t bytes_per_group(int size) {
    return 2 * elements(size) * sizeof(__nv_fp8_e4m3) +
           2 * scale_elements(size) * sizeof(__nv_fp8_e8m0) +
           elements(size) * sizeof(__nv_bfloat16);
  }

  cublasStatus_t launch(
      cublasLtMatmulAlgo_t const& algorithm, int group, cudaStream_t stream) {
    // cuBLASLt A is logical B because of the row-major transpose trick.
    plan_.set_scale_pointers(b_scales_[group], a_scales_[group]);
    const float alpha = 1.0f;
    const float beta = 0.0f;
    return cublasLtMatmul(
        plan_.handle, plan_.operation, &alpha,
        b_[group], plan_.a_layout, a_[group], plan_.b_layout, &beta,
        d_[group], plan_.c_layout, d_[group], plan_.d_layout,
        &algorithm, plan_.workspace, kWorkspaceBytes, stream);
  }

  int size_;
  int groups_;
  DeviceGroups<__nv_fp8_e4m3> a_, b_;
  DeviceGroups<__nv_fp8_e8m0> a_scales_, b_scales_;
  DeviceGroups<__nv_bfloat16> d_;
  LtPlan plan_;
};

}  // namespace

int main(int argc, char** argv) {
  return benchmark_main<Mxfp8Workload>(argc, argv);
}
