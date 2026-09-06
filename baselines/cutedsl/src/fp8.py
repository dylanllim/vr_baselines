import json
import sys

import cutlass
import torch

from blackwell.kernel.dense_gemm import dense_gemm_persistent as blackwell
from protocol import COLD_L2, FP8_RANGE, ITERATIONS, SEED, WARMUPS
from rubin.kernel.dense_gemm import dense_gemm_persistent as kernel


def shape(value):
    return tuple(map(int, value.split("x")))


def patch_base():
    base = blackwell.PersistentDenseGemmKernel
    original_init = base.__init__

    def init(self, acc, two_cta, tiler, cluster, tma, swizzle=1, raster="m"):
        original_init(self, acc, two_cta, tiler, cluster, tma)
        self.swizzle_size, self.raster_along = swizzle, raster

    def grid(c, tile, cluster, swizzle, raster, active):
        ctas = blackwell.cute.zipped_divide(
            c, tiler=blackwell.cute.slice_(tile, (None, None, 0))
        )[(0, (None, None, None))].shape
        params = blackwell.utils.PersistentTileSchedulerParams(
            ctas, (*cluster, 1), swizzle, raster == "m")
        return params, blackwell.utils.StaticPersistentTileScheduler.get_grid_shape(params, active)

    base.__init__ = init
    base._compute_grid = staticmethod(grid)


def run(size, config):
    patch_base()
    original = kernel.prepare_tensors

    def uniform(*pos, **kw):
        torch.manual_seed(SEED)
        torch.cuda.manual_seed_all(SEED)
        tensors = original(*pos, **kw)
        for tensor in tensors[:2]:
            tensor.uniform_(*FP8_RANGE)
            tensor.copy_(tensor.to(torch.float8_e4m3fn).to(torch.float32))
        return tensors

    kernel.prepare_tensors = uniform
    return kernel.run(
        (size, size, size, 1),
        cutlass.Float8E4M3FN, cutlass.Float8E4M3FN,
        cutlass.BFloat16, cutlass.Float32, "k", "k", "n",
        shape(config["mma_tiler"]), shape(config["mma_inst"]),
        shape(config["cluster"]), int(config.get("swizzle", 1)),
        config.get("raster", "m"), config.get("two_cta", "0") == "1",
        True, 0.1, WARMUPS, ITERATIONS, True, COLD_L2, True)


if __name__ == "__main__":
    print(f"CUTEDSL_RESULT_US={float(run(int(sys.argv[1]), json.loads(sys.argv[2])))}")
