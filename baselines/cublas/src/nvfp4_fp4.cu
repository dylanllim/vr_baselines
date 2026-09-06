#include <cuda_bf16.h>
#include <cuda_fp4.h>
#include <cuda_fp8.h>

#include "algorithms.cuh"

namespace {

constexpr AlgorithmConfig kWinners[] = {
    {.size = 1024,  .tile = 20,  .stages = 37, .cluster = 3},
    {.size = 2048,  .tile = 24,  .stages = 37, .custom = 2, .cluster = 8},
    {.size = 4096,  .tile = 24,  .stages = 37, .custom = 2, .cluster = 9},
    {.size = 8192,  .tile = 24,  .stages = 37, .cluster = 7},
    {.size = 16384, .tile = 513, .stages = 37, .cluster = 10},
    {.size = 32768, .tile = 513, .stages = 37, .cluster = 7},
};

class Nvfp4OutputWorkload {
 public:
  explicit Nvfp4OutputWorkload(int size)
      : size_(size),
        groups_(cold_buffer_groups(bytes_per_group(size))),
        a_(groups_, packed_elements(size)),
        b_(groups_, packed_elements(size)),
        a_scales_(groups_, scale_elements(size)),
        b_scales_(groups_, scale_elements(size)),
        c_(groups_, output_elements(size)),
        d_(groups_, packed_elements(size)),
        d_scale_(groups_, 1),
        d_out_scales_(groups_, output_scale_elements(size)),
        alpha_(groups_),
        plan_(size, CUDA_R_4F_E2M1, CUDA_R_16BF, CUDA_R_4F_E2M1) {
    cublasLtMatmulMatrixScale_t block = CUBLASLT_MATMUL_MATRIX_SCALE_VEC16_UE4M3;
    cublasLtMatmulMatrixScale_t scalar = CUBLASLT_MATMUL_MATRIX_SCALE_SCALAR_32F;
    CHECK_CUBLAS(cublasLtMatmulDescSetAttribute(
        plan_.operation, CUBLASLT_MATMUL_DESC_A_SCALE_MODE, &block, sizeof(block)));
    CHECK_CUBLAS(cublasLtMatmulDescSetAttribute(
        plan_.operation, CUBLASLT_MATMUL_DESC_B_SCALE_MODE, &block, sizeof(block)));
    CHECK_CUBLAS(cublasLtMatmulDescSetAttribute(
        plan_.operation, CUBLASLT_MATMUL_DESC_D_SCALE_MODE, &scalar, sizeof(scalar)));
    CHECK_CUBLAS(cublasLtMatmulDescSetAttribute(
        plan_.operation, CUBLASLT_MATMUL_DESC_D_OUT_SCALE_MODE,
        &block, sizeof(block)));

    for (int group = 0; group < groups_; ++group) {
      const uint64_t seed = BENCHMARK_SEED + 100 * group;
      fill_uniform(a_[group], a_.elements(), seed,
                   BENCHMARK_NVFP4_PACKED_LOW,
                   BENCHMARK_NVFP4_PACKED_HIGH + 1.0f);
      fill_uniform(b_[group], b_.elements(), seed + 1,
                   BENCHMARK_NVFP4_PACKED_LOW,
                   BENCHMARK_NVFP4_PACKED_HIGH + 1.0f);
      fill_uniform(a_scales_[group], a_scales_.elements(), seed + 2,
                   BENCHMARK_NVFP4_SCALE_LOW, BENCHMARK_NVFP4_SCALE_HIGH);
      fill_uniform(b_scales_[group], b_scales_.elements(), seed + 3,
                   BENCHMARK_NVFP4_SCALE_LOW, BENCHMARK_NVFP4_SCALE_HIGH);
      fill_zero(c_[group], c_.elements());
      fill_zero(d_[group], d_.elements());
      fill_uniform(d_scale_[group], 1, seed + 4, 1.0f, 1.0f);
      fill_zero(d_out_scales_[group], d_out_scales_.elements());
      alpha_[group] = uniform_value(seed + 5, BENCHMARK_NVFP4_SCALE_LOW,
                                    BENCHMARK_NVFP4_SCALE_HIGH) *
                      uniform_value(seed + 6, BENCHMARK_NVFP4_SCALE_LOW,
                                    BENCHMARK_NVFP4_SCALE_HIGH);
    }
    CHECK_CUDA(cudaDeviceSynchronize());
    set_scale_pointers(
        plan_.workspace, plan_.workspace, plan_.workspace, plan_.workspace);
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
  static size_t output_elements(int size) { return size_t(size) * size; }
  static size_t packed_elements(int size) { return output_elements(size) / 2; }
  static size_t scale_elements(int size) { return output_elements(size) / 16; }
  static size_t output_scale_elements(int size) {
    return size_t((size + 63) / 64 * 4) * ((size + 127) / 128 * 128);
  }

  static size_t bytes_per_group(int size) {
    return 3 * packed_elements(size) * sizeof(uint8_t) +
           2 * scale_elements(size) * sizeof(__nv_fp8_e4m3) +
           output_scale_elements(size) * sizeof(__nv_fp8_e4m3);
  }

  void set_scale_pointers(
      void const* a_scale, void const* b_scale, void const* d_scale,
      void* d_out_scale) {
    plan_.set_scale_pointers(a_scale, b_scale);
    CHECK_CUBLAS(cublasLtMatmulDescSetAttribute(
        plan_.operation, CUBLASLT_MATMUL_DESC_D_SCALE_POINTER,
        &d_scale, sizeof(d_scale)));
    CHECK_CUBLAS(cublasLtMatmulDescSetAttribute(
        plan_.operation, CUBLASLT_MATMUL_DESC_D_OUT_SCALE_POINTER,
        &d_out_scale, sizeof(d_out_scale)));
  }

  cublasStatus_t launch(
      cublasLtMatmulAlgo_t const& algorithm, int group, cudaStream_t stream) {
    // cuBLASLt A is logical B because of the row-major transpose trick.
    set_scale_pointers(b_scales_[group], a_scales_[group], d_scale_[group],
                       d_out_scales_[group]);
    const float beta = 0.0f;
    return cublasLtMatmul(
        plan_.handle, plan_.operation, &alpha_[group],
        b_[group], plan_.a_layout, a_[group], plan_.b_layout, &beta,
        c_[group], plan_.c_layout, d_[group], plan_.d_layout,
        &algorithm, plan_.workspace, kWorkspaceBytes, stream);
  }

  int size_;
  int groups_;
  DeviceGroups<uint8_t> a_, b_, d_;
  DeviceGroups<__nv_fp8_e4m3> a_scales_, b_scales_, d_out_scales_;
  DeviceGroups<__nv_bfloat16> c_;
  DeviceGroups<float> d_scale_;
  std::vector<float> alpha_;
  LtPlan plan_;
};

}  // namespace

int main(int argc, char** argv) {
  return benchmark_main<Nvfp4OutputWorkload>(argc, argv);
}
