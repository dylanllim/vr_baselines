import json
import os
import sys

import cutlass
import cutlass.torch as cutlass_torch
import torch

from blackwell.kernel.blockscaled_grouped_gemm import grouped_blockscaled_gemm as grouped_base
from protocol import (COLD_L2, ITERATIONS, NVFP4_CODES, NVFP4_PACKED_RANGE,
                      NVFP4_SCALE_RANGE, SEED, WARMUPS)
from rubin.kernel.blockscaled_gemm import dense_blockscaled_gemm_persistent as persistent
from rubin.kernel.blockscaled_gemm import dense_blockscaled_gemm_persistent_mixed_clusters as mixed
from rubin.kernel.blockscaled_grouped_gemm import grouped_blockscaled_gemm as grouped
from rubin.kernel.blockscaled_grouped_gemm import grouped_blockscaled_gemm_mixed_clusters as grouped_mixed


def shape(value):
    return tuple(map(int, value.split("x")))


def patch_grouped_data():
    base = grouped_base.Sm100GroupedBlockScaledGemmKernel
    original_init = base.__init__
    original_tensor = grouped_base.create_tensor_and_stride
    original_scale = grouped_base.create_scale_factor_tensor
    original_abc = grouped.create_tensors_abc_for_all_groups
    tensor_count = 0
    scale_count = 0

    def init(self, sf_vec_size, tiler, cluster, cached=True):
        original_init(self, sf_vec_size, tiler, cluster)
        self.use_cached_problem_shapes = cached

    base.__init__ = init

    def uniform_tensor(*args, **kwargs):
        nonlocal tensor_count
        result = list(original_tensor(*args, **kwargs))
        torch.cuda.manual_seed(SEED + tensor_count)
        storage = result[1].view(torch.uint8)
        if tensor_count % 3 < 2:
            low, high = map(int, NVFP4_PACKED_RANGE)
            storage.random_(low, high + 1)
        else:
            storage.zero_()
        tensor_count += 1
        return tuple(result)

    def uniform_scale(*args, **kwargs):
        nonlocal scale_count
        result = list(original_scale(*args, **kwargs))
        torch.cuda.manual_seed(SEED + 1000 + scale_count)
        source = torch.empty_like(result[3], dtype=torch.float32)
        source.uniform_(*NVFP4_SCALE_RANGE)
        result[2] = cutlass_torch.convert_cute_tensor(
            source, result[2], args[4], is_dynamic_layout=True)
        scale_count += 1
        return tuple(result)

    grouped_base.create_tensor_and_stride = uniform_tensor
    grouped_base.create_scale_factor_tensor = uniform_scale

    def tensors_abc(problem_sizes, a_dtype, b_dtype, c_dtype,
                    a_major, b_major, c_major, **_):
        if a_dtype != b_dtype:
            raise ValueError("grouped helper requires matching A/B dtypes")
        return original_abc(
            problem_sizes, a_dtype, c_dtype, a_major, b_major, c_major)

    grouped.create_tensors_abc_for_all_groups = tensors_abc
    grouped_mixed.create_tensors_abc_for_all_groups = tensors_abc


def run_grouped(size, config):
    patch_grouped_data()
    screen = os.getenv("BASELINE_TUNE_SCREEN") == "1"
    warmups, iterations = (1, 3) if screen else (WARMUPS, ITERATIONS)
    common = (
        1, [(size, size, size, 1)], True,
        cutlass.Float4E2M1FN, cutlass.Float4E2M1FN,
        cutlass.Float8E4M3FN, 16, cutlass.Float4E2M1FN,
        "k", "k", "n", shape(config["mma_tiler"]),
        shape(config["mma_inst"]),
    )
    if config["variant"] == "grouped_mixed":
        return grouped_mixed.run(
            *common, shape(config["cluster"]), shape(config["fallback"]),
            0.1, warmups, iterations, True, COLD_L2)
    return grouped.run(
        *common, shape(config["cluster"]), 0.1, warmups, iterations,
        True, COLD_L2, config.get("cached", "1") == "1")


def run(size, config, low_output=False):
    if low_output:
        return run_grouped(size, config)
    variant = config.get("variant") or "persistent"
    kernel = mixed if variant == "mixed" else persistent
    original = kernel.create_and_init_tensors_emulated

    def uniform(*pos, **kw):
        torch.manual_seed(SEED)
        torch.cuda.manual_seed_all(SEED)
        a, b, c, sfa, sfb = original(*pos, **kw)
        values = torch.tensor(NVFP4_CODES, dtype=torch.float32, device="cuda")
        for tensor in (a, b):
            tensor.copy_(values[torch.randint(len(values), tensor.shape, device="cuda")])
        for tensor in (sfa, sfb):
            tensor.uniform_(*NVFP4_SCALE_RANGE)
        return a, b, c, sfa, sfb

    kernel.create_and_init_tensors_emulated = uniform
    common = (
        (size, size, size, 1),
        cutlass.Float4E2M1FN, cutlass.Float4E2M1FN, cutlass.Float8E4M3FN,
        16, cutlass.BFloat16, "k", "k", "n",
        shape(config["mma_tiler"]), shape(config["mma_inst"]),
    )
    prefetch = config.get("prefetch", "auto")
    prefetch = None if prefetch == "auto" else int(prefetch)
    screen = os.getenv("BASELINE_TUNE_SCREEN") == "1"
    warmups, iterations = (1, 3) if screen else (WARMUPS, ITERATIONS)
    if variant == "mixed":
        return kernel.run(
            *common, shape(config["cluster"]), shape(config["fallback"]), 0.1,
            warmups, iterations, True, COLD_L2, prefetch_dist=prefetch)
    return kernel.run(
        *common, shape(config["cluster"]), int(config.get("swizzle", 1)),
        config.get("raster", "m"), config.get("scheduler", "static_persistent"),
        0.1, warmups, iterations, True, COLD_L2, prefetch_dist=prefetch)


if __name__ == "__main__":
    result = run(
        int(sys.argv[1]), json.loads(sys.argv[2]),
        len(sys.argv) == 4 and sys.argv[3] == "nvfp4_fp4")
    print(f"CUTEDSL_RESULT_US={float(result)}")
