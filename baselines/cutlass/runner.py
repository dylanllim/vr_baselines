#!/usr/bin/env python3
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
    with (HERE / "configs/winners.csv").open(newline="") as source:
        for row in csv.DictReader(source):
            if row["family"] == family and int(row["size"]) == size:
                return row["kernel"]
    raise SystemExit(f"no pinned CUTLASS winner for {family} {size}")


def benchmark(family, size, tune, profiler, directory):
    if tune:
        prefix = directory / f"{family}_screen"
        run(command(profiler, family, size, FAMILIES[family][1], prefix),
            prefix.with_suffix(".log"), family)
        candidates = rows(prefix)
        winner = max(candidates, key=lambda row: float(row["GFLOPs"]))["Operation"]
        print(f"ALGORITHM family={family} candidates={len(candidates)} kernel={winner}")
    else:
        winner = pinned(family, size)
    samples = []
    for rep in range(SAMPLES):
        prefix = directory / f"{family}_rep{rep + 1}"
        run(command(profiler, family, size, winner, prefix, True),
            prefix.with_suffix(".log"), family)
        samples.append(float(rows(prefix)[0]["GFLOPs"]) / 1000)
    values = ",".join(f"{value:.6f}" for value in samples)
    print(f"RESULT family={family} size={size} "
          f"mean_tflops={sum(samples) / len(samples):.6f} samples_tflops={values}")


def main(argv=sys.argv[1:]):
    if len(argv) not in (2, 3) or (len(argv) == 3 and argv[2] != "tune"):
        raise SystemExit("usage: runner.py FAMILY SIZE [tune]")
    family, size = argv[:2]
    if family not in ("all", *FAMILIES) or (size := int(size)) not in SIZES:
        raise SystemExit("unsupported family or size")
    profiler = HERE.parents[1] / "submodules/cutlass/build/tools/profiler/cutlass_profiler"
    if not profiler.is_file():
        raise SystemExit("CUTLASS profiler not found; run make build")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    directory = HERE / "results" / f"{stamp}_{size}"
    directory.mkdir(parents=True)
    for selected in FAMILIES if family == "all" else (family,):
        benchmark(selected, size, len(argv) == 3, profiler, directory)


if __name__ == "__main__":
    main()
