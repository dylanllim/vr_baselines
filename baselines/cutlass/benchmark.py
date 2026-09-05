#!/usr/bin/env python3
import argparse
import csv
from datetime import datetime, timezone
import os
from pathlib import Path
import subprocess
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from protocol import (COLD_L2, FP8_RANGE, ITERATIONS, NVFP4_PACKED_RANGE,
                      NVFP4_RANGE, NVFP4_SCALE_RANGE, SAMPLES, SEED, SIZES,
                      WARMUPS)


def uniform(bounds):
    return f"uniform,min:{bounds[0]:g},max:{bounds[1]:g},scale:-1"


FAMILIES = {
    "fp8": (
        "gemm",
        "cutlass3x_sm107_tensorop_gemm_e4m3_e4m3_f32_bf16_bf16*",
        uniform(FP8_RANGE),
    ),
    "nvfp4": (
        "block_scaled_gemm",
        "cutlass3x_sm107_bstensorop_gemm_ue4m3xe2m1_ue4m3xe2m1_f32_bf16_bf16*",
        uniform(NVFP4_RANGE),
    ),
}


def command(profiler, family, size, kernel, output, final=False):
    operation, _, distribution = FAMILIES[family]
    return [
        str(profiler), f"--operation={operation}", f"--kernels={kernel}",
        f"--m={size}", f"--n={size}", f"--k={size}",
        f"--warmup-iterations={WARMUPS}",
        f"--profiling-iterations={ITERATIONS if final else 0}",
        "--profiling-duration=2000", "--min-iterations=20",
        "--sleep-duration=0", "--verification-enabled=false",
        "--providers=cutlass", "--initialization-provider=device",
        f"--workspace-count={0 if COLD_L2 else 1}",
        f"--seed={SEED}", f"--dist={distribution}",
        f"--output={output}",
    ]


def run(command, log, family):
    print("COMMAND " + " ".join(map(str, command)), flush=True)
    env = os.environ.copy()
    if family == "nvfp4":
        env |= {
            "BASELINE_NVFP4_PACKED_LOW": str(NVFP4_PACKED_RANGE[0]),
            "BASELINE_NVFP4_PACKED_HIGH": str(NVFP4_PACKED_RANGE[1]),
            "BASELINE_NVFP4_SCALE_LOW": str(NVFP4_SCALE_RANGE[0]),
            "BASELINE_NVFP4_SCALE_HIGH": str(NVFP4_SCALE_RANGE[1]),
        }
    with log.open("w") as output:
        result = subprocess.run(
            command, stdout=output, stderr=subprocess.STDOUT, env=env)
    if result.returncode:
        raise SystemExit(f"profiler failed; see {log}")


def rows(prefix):
    files = list(prefix.parent.glob(prefix.name + "*.csv"))
    if len(files) != 1:
        raise SystemExit(f"expected one CSV for {prefix}, found {files}")
    with files[0].open(newline="") as source:
        result = [
            row for row in csv.DictReader(source)
            if "success" in row.get("Status", "").lower() and row.get("GFLOPs")
        ]
    if not result:
        raise SystemExit(f"no successful kernels in {files[0]}")
    return result


def pinned(family, size):
    with (HERE / "winners.csv").open(newline="") as source:
        for row in csv.DictReader(source):
            if row["family"] == family and int(row["size"]) == size:
                return row["kernel"]
    raise SystemExit(f"no pinned CUTLASS winner for {family} {size}")


def benchmark(args, family, directory):
    if args.tune:
        prefix = directory / f"{family}_screen"
        run(command(args.profiler, family, args.size, FAMILIES[family][1], prefix),
            prefix.with_suffix(".log"), family)
        candidates = rows(prefix)
        winner = max(candidates, key=lambda row: float(row["GFLOPs"]))["Operation"]
        print(f"SELECTED family={family} candidates={len(candidates)} kernel={winner}")
    else:
        winner = pinned(family, args.size)
        print(f"PINNED family={family} kernel={winner}")
    samples = []
    for rep in range(SAMPLES):
        prefix = directory / f"{family}_rep{rep + 1}"
        run(command(args.profiler, family, args.size, winner, prefix, True),
            prefix.with_suffix(".log"), family)
        samples.append(float(rows(prefix)[0]["GFLOPs"]) / 1000)
    print(f"SAMPLES_TFLOPS={','.join(map(str, samples))}")
    print(f"Performance: {sum(samples) / len(samples):.6f} TFLOP/s")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--family", choices=("all", *FAMILIES), default="all")
    parser.add_argument("--size", type=int, choices=SIZES, default=8192)
    parser.add_argument("--profiler", type=Path)
    parser.add_argument("--results", type=Path, default=HERE / "results")
    parser.add_argument("--tune", action="store_true")
    args = parser.parse_args()
    if args.profiler is None:
        cutlass = Path(os.getenv("CUTLASS_DIR", HERE.parents[1] / "submodules/cutlass"))
        args.profiler = cutlass / "build/tools/profiler/cutlass_profiler"
    args.profiler = args.profiler.resolve()
    if not args.profiler.is_file():
        raise SystemExit(f"profiler not found: {args.profiler}")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    directory = args.results / f"{stamp}_{args.size}"
    directory.mkdir(parents=True)
    for family in FAMILIES if args.family == "all" else (args.family,):
        benchmark(args, family, directory)


if __name__ == "__main__":
    main()
