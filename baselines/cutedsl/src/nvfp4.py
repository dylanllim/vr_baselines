import json
import sys

import cutlass
import torch

from protocol import (COLD_L2, ITERATIONS, NVFP4_CODES, NVFP4_SCALE_RANGE,
                      SEED, WARMUPS)
from rubin.kernel.blockscaled_gemm import dense_blockscaled_gemm_persistent as persistent
from rubin.kernel.blockscaled_gemm import dense_blockscaled_gemm_persistent_mixed_clusters as mixed


def shape(value):
    return tuple(map(int, value.split("x")))


def run(size, config):
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
    if variant == "mixed":
        return kernel.run(
            *common, shape(config["cluster"]), shape(config["fallback"]), 0.1,
            WARMUPS, ITERATIONS, True, COLD_L2, prefetch_dist=prefetch)
    return kernel.run(
        *common, shape(config["cluster"]), int(config.get("swizzle", 1)),
        config.get("raster", "m"), config.get("scheduler", "static_persistent"),
        0.1, WARMUPS, ITERATIONS, True, COLD_L2, prefetch_dist=prefetch)


if __name__ == "__main__":
    print(f"CUTEDSL_RESULT_US={float(run(int(sys.argv[1]), json.loads(sys.argv[2])))}")
