#!/usr/bin/env python3
import argparse
import csv
import itertools
import json
import os
from pathlib import Path
import re
import subprocess
import sys

HERE = Path(__file__).resolve().parent
CUTLASS = HERE.parents[1] / "submodules/cutlass"
sys.path.insert(0, str(HERE.parent))
from protocol import (COLD_L2, FP8_RANGE, ITERATIONS, NVFP4_CODES,
                      NVFP4_SCALE_RANGE, SAMPLES, SEED, SIZES, WARMUPS)
TIME = re.compile(r"CUTEDSL_RESULT_US=([0-9.eE+-]+)")


def configs(family):
    with (HERE / f"{family}.csv").open(newline="") as f:
        rows = list(csv.DictReader(line for line in f if not line.startswith("#")))
    result, seen = [], set()
    for row in rows:
        variant = row.get("variant") or "persistent"
        knobs = []
        if family == "fp8" or variant == "persistent":
            knobs += [("swizzle", (1, 2, 4, 8)), ("raster", ("m", "n"))]
        if family == "nvfp4":
            if variant == "persistent":
                knobs += [("scheduler", ("static_persistent", "clc_dynamic_persistent"))]
            knobs += [("prefetch", ("auto", 0, 1, 2, 3, 4))]
        for values in itertools.product(*(values for _, values in knobs)):
            cfg = row | {name: str(value) for (name, _), value in zip(knobs, values)}
            key = tuple(sorted((k, v) for k, v in cfg.items() if k != "id"))
            if key not in seen:
                seen.add(key)
                result.append(cfg)
    return result


def pinned(family, size):
    with (HERE / "winners.csv").open(newline="") as source:
        for row in csv.DictReader(source):
            if row["family"] == family and int(row["size"]) == size:
                return row
    raise SystemExit(f"no pinned CuTeDSL winner for {family} {size}; run tune")


def shape(value):
    return tuple(map(int, value.split("x")))


def install(cutlass_dir):
    root = cutlass_dir / "examples/python/CuTeDSL"
    if not root.is_dir():
        raise SystemExit(f"missing {root}; initialize the CUTLASS submodule")
    sys.path[:0] = [str(root), str(root / "cute")]


def single(args):
    install(args.cutlass_dir)
    cfg = json.loads(args.config_json)
    inst, tiler, cluster = map(shape, (cfg["mma_inst"], cfg["mma_tiler"], cfg["cluster"]))
    import cutlass
    import torch

    if args.family == "nvfp4":
        variant = cfg.get("variant") or "persistent"
        if variant == "mixed":
            from rubin.kernel.blockscaled_gemm import dense_blockscaled_gemm_persistent_mixed_clusters as vendor
        else:
            from rubin.kernel.blockscaled_gemm import dense_blockscaled_gemm_persistent as vendor
        original = vendor.create_and_init_tensors_emulated

        def uniform(*pos, **kw):
            torch.manual_seed(SEED)
            torch.cuda.manual_seed_all(SEED)
            a, b, c, sfa, sfb = original(*pos, **kw)
            values = torch.tensor(NVFP4_CODES, dtype=torch.float32, device="cuda")
            for tensor in (a, b):
                codes = torch.randint(0, len(NVFP4_CODES), tensor.shape, device="cuda")
                tensor.copy_(values[codes])
            for tensor in (sfa, sfb):
                tensor.copy_(torch.empty_like(tensor).uniform_(*NVFP4_SCALE_RANGE))
            return a, b, c, sfa, sfb

        vendor.create_and_init_tensors_emulated = uniform
        common = (
            (args.size, args.size, args.size, 1),
            cutlass.Float4E2M1FN,
            cutlass.Float4E2M1FN,
            cutlass.Float8E4M3FN,
            16,
            cutlass.BFloat16,
            "k", "k", "n",
            tiler,
            inst,
        )
        prefetch = None if cfg.get("prefetch", "auto") == "auto" else int(cfg["prefetch"])
        if variant == "mixed":
            elapsed = vendor.run(
                *common, cluster, shape(cfg["fallback"]), 0.1,
                WARMUPS, ITERATIONS, True,
                COLD_L2, prefetch_dist=prefetch)
        else:
            elapsed = vendor.run(
                *common, cluster, int(cfg.get("swizzle", 1)), cfg.get("raster", "m"),
                cfg.get("scheduler", "static_persistent"), 0.1,
                WARMUPS, ITERATIONS, True,
                COLD_L2, prefetch_dist=prefetch)
    else:
        from blackwell.kernel.dense_gemm import dense_gemm_persistent as blackwell
        base = blackwell.PersistentDenseGemmKernel
        original_init = base.__init__

        def compatible_init(
            self, acc_dtype, two_cta, mma_tiler, cluster_shape, tma_store,
            swizzle=1, raster="m"):
            original_init(self, acc_dtype, two_cta, mma_tiler, cluster_shape, tma_store)
            self.swizzle_size, self.raster_along = swizzle, raster

        def compatible_grid(c, tile, cluster_shape, swizzle, raster, active):
            c_shape = blackwell.cute.slice_(tile, (None, None, 0))
            ctas = blackwell.cute.zipped_divide(c, tiler=c_shape)[(0, (None, None, None))].shape
            params = blackwell.utils.PersistentTileSchedulerParams(
                ctas, (*cluster_shape, 1), swizzle, raster == "m")
            return params, blackwell.utils.StaticPersistentTileScheduler.get_grid_shape(params, active)

        base.__init__ = compatible_init
        base._compute_grid = staticmethod(compatible_grid)
        from rubin.kernel.dense_gemm import dense_gemm_persistent as vendor
        original = vendor.prepare_tensors

        def uniform(*pos, **kw):
            torch.manual_seed(SEED)
            torch.cuda.manual_seed_all(SEED)
            tensors = original(*pos, **kw)
            for tensor in tensors[:2]:
                tensor.copy_(
                    torch.empty_like(tensor).uniform_(*FP8_RANGE)
                    .to(torch.float8_e4m3fn).to(torch.float32))
            return tensors

        vendor.prepare_tensors = uniform
        elapsed = vendor.run(
            (args.size, args.size, args.size, 1),
            cutlass.Float8E4M3FN, cutlass.Float8E4M3FN,
            cutlass.BFloat16, cutlass.Float32,
            "k", "k", "n", tiler, inst, cluster,
            int(cfg.get("swizzle", 1)), cfg.get("raster", "m"),
            cfg.get("two_cta", "0") == "1", True, 0.1, WARMUPS, ITERATIONS,
            True, COLD_L2, True)

    print(f"VENDOR_SOURCE={Path(vendor.__file__).resolve()}")
    print(f"CUTEDSL_RESULT_US={float(elapsed)}")


def measure(args, cfg):
    command = [
        sys.executable, __file__, "--family", args.family, "--size", str(args.size),
        "--cutlass-dir", str(args.cutlass_dir), "--single",
        "--config-json", json.dumps(cfg, separators=(",", ":")),
    ]
    run = subprocess.run(command, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    match = TIME.search(run.stdout)
    if run.returncode or not match:
        print(run.stdout, end="")
        print(f"FAIL id={cfg['id']} rc={run.returncode}")
        return None
    return float(match.group(1))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--family", choices=("fp8", "nvfp4"), required=True)
    parser.add_argument("--size", type=int, choices=SIZES, default=8192)
    parser.add_argument("--cutlass-dir", type=Path, default=Path(os.getenv("CUTLASS_DIR", CUTLASS)))
    parser.add_argument("--single", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--config-json", help=argparse.SUPPRESS)
    parser.add_argument("--tune", action="store_true")
    args = parser.parse_args()
    args.cutlass_dir = args.cutlass_dir.resolve()
    if args.single:
        single(args)
        return

    kernel = "dense_gemm/dense_gemm_persistent.py" if args.family == "fp8" else "blockscaled_gemm"
    print(f"VENDOR_ROOT={args.cutlass_dir / 'examples/python/CuTeDSL/cute/rubin/kernel' / kernel}")
    if args.tune:
        successful = []
        for cfg in configs(args.family):
            elapsed = measure(args, cfg)
            if elapsed is not None:
                successful.append((elapsed, cfg))
                print(f"CANDIDATE id={cfg['id']} us={elapsed:.6f}")
        if not successful:
            raise SystemExit("all candidates failed")
        _, winner = min(successful, key=lambda item: item[0])
        print("SELECTED " + " ".join(f"{k}={v}" for k, v in winner.items()))
    else:
        winner = pinned(args.family, args.size)
        print("PINNED " + " ".join(f"{k}={v}" for k, v in winner.items()))
    samples = [measure(args, winner) for _ in range(SAMPLES)]
    if any(value is None for value in samples):
        raise SystemExit("selected configuration failed")
    mean = sum(samples) / len(samples)
    tflops = 2 * args.size**3 / (mean * 1e6)
    print(f"SAMPLES_US={','.join(map(str, samples))}")
    print(f"Performance: {tflops:.6f} TFLOP/s")


if __name__ == "__main__":
    main()
