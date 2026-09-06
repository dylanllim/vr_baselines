#include <cuda_bf16.h>
#include <cuda_fp8.h>

#include "algorithms.cuh"

namespace {

constexpr AlgorithmConfig kWinners[] = {
    {.size = 1024,  .tile = 17,  .stages = 36, .custom = 1, .cluster = 7},
    {.size = 2048,  .tile = 30,  .stages = 36, .custom = 2, .cluster = 5},
    {.size = 4096,  .tile = 176, .stages = 36, .custom = 3, .cluster = 3},
    {.size = 8192,  .tile = 513, .stages = 36, .custom = 2, .cluster = 7},
    {.size = 16384, .tile = 513, .stages = 36, .cluster = 7},
    {.size = 32768, .tile = 513, .stages = 36, .custom = 3, .cluster = 9},
};

class Fp8OutputWorkload {
 public:
  explicit Fp8OutputWorkload(int size)
      : size_(size),
        groups_(cold_buffer_groups(bytes_per_group(size))),
        a_(groups_, elements(size)),
        b_(groups_, elements(size)),
        c_(groups_, elements(size)),
        d_(groups_, elements(size)),
        scales_(groups_, 4),
        plan_(size, CUDA_R_8F_E4M3, CUDA_R_16BF, CUDA_R_8F_E4M3) {
    for (int group = 0; group < groups_; ++group) {
      const uint64_t seed = BENCHMARK_SEED + 100 * group;
      fill_uniform(a_[group], a_.elements(), seed,
                   BENCHMARK_FP8_LOW, BENCHMARK_FP8_HIGH);
      fill_uniform(b_[group], b_.elements(), seed + 1,
                   BENCHMARK_FP8_LOW, BENCHMARK_FP8_HIGH);
      fill_zero(c_[group], c_.elements());
      fill_zero(d_[group], d_.elements());
      fill_uniform(scales_[group], 4, seed + 2, 1.0f, 1.0f);
    }
    CHECK_CUDA(cudaDeviceSynchronize());
    set_scale_pointers(plan_.workspace);
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

  static size_t bytes_per_group(int size) {
    return 3 * elements(size) * sizeof(__nv_fp8_e4m3);
  }

  void set_scale_pointers(void const* pointer) {
    plan_.set_scale_pointers(pointer, pointer);
    CHECK_CUBLAS(cublasLtMatmulDescSetAttribute(
        plan_.operation, CUBLASLT_MATMUL_DESC_C_SCALE_POINTER,
        &pointer, sizeof(pointer)));
    CHECK_CUBLAS(cublasLtMatmulDescSetAttribute(
        plan_.operation, CUBLASLT_MATMUL_DESC_D_SCALE_POINTER,
        &pointer, sizeof(pointer)));
  }

  cublasStatus_t launch(
      cublasLtMatmulAlgo_t const& algorithm, int group, cudaStream_t stream) {
    float* scales = scales_[group];
    plan_.set_scale_pointers(scales, scales + 1);
    void const* c_scale = scales + 2;
    void const* d_scale = scales + 3;
    CHECK_CUBLAS(cublasLtMatmulDescSetAttribute(
        plan_.operation, CUBLASLT_MATMUL_DESC_C_SCALE_POINTER,
        &c_scale, sizeof(c_scale)));
    CHECK_CUBLAS(cublasLtMatmulDescSetAttribute(
        plan_.operation, CUBLASLT_MATMUL_DESC_D_SCALE_POINTER,
        &d_scale, sizeof(d_scale)));
    const float alpha = 1.0f;
    const float beta = 0.0f;
    return cublasLtMatmul(
        plan_.handle, plan_.operation, &alpha,
        b_[group], plan_.a_layout, a_[group], plan_.b_layout, &beta,
        c_[group], plan_.c_layout, d_[group], plan_.d_layout,
        &algorithm, plan_.workspace, kWorkspaceBytes, stream);
  }

  int size_;
  int groups_;
  DeviceGroups<__nv_fp8_e4m3> a_, b_, d_;
  DeviceGroups<__nv_bfloat16> c_;
  DeviceGroups<float> scales_;
  LtPlan plan_;
};

}  // namespace

int main(int argc, char** argv) {
  return benchmark_main<Fp8OutputWorkload>(argc, argv);
}
