#!/usr/bin/env python3
import csv
import itertools
import json
import os
from pathlib import Path
import re
import statistics
import subprocess
import sys

HERE = Path(__file__).resolve().parent
CSV_DIR = HERE / "configs"
CUTEDSL = HERE.parents[1] / "submodules/cutlass/examples/python/CuTeDSL"
sys.path.insert(0, str(HERE.parent))
from protocol import SAMPLES, SIZES

TIME = re.compile(r"CUTEDSL_RESULT_US=([0-9.eE+-]+)")
ENV = os.environ.copy()
ENV["PYTHONPATH"] = os.pathsep.join(filter(None, (
    str(HERE.parent), str(CUTEDSL), str(CUTEDSL / "cute"), ENV.get("PYTHONPATH"))))
BASE_FAMILY = {
    "fp8": "fp8", "mxfp8": "mxfp8", "nvfp4": "nvfp4",
    "fp8_fp8": "fp8", "nvfp4_fp4": "nvfp4",
}


def csv_rows(name):
    with (CSV_DIR / name).open(newline="") as source:
        return list(csv.DictReader(source))


def grouped_candidates(size):
    for inst_m in (128, 256):
        clusters = [(m, n) for m in (1, 2, 4) for n in (1, 2, 4)
                    if m * n <= 16 and (inst_m == 128 or m % 2 == 0)]
        for inst_n in (64, 128, 192, 256):
            for tiler_m in (inst_m, 2 * inst_m):
                common = {
                    "mma_inst": f"{inst_m}x{inst_n}x128",
                    "mma_tiler": f"{tiler_m}x{inst_n}x256",
                }
                for cluster in clusters:
                    label = f"{cluster[0]}x{cluster[1]}"
                    for cached in ("0", "1"):
                        yield common | {
                            "id": f"grouped_{inst_m}_{inst_n}_{tiler_m}_{label}_{cached}",
                            "cluster": label, "variant": "grouped", "cached": cached,
                        }
                    if size < tiler_m * cluster[0] or size < inst_n * cluster[1]:
                        continue
                    for fallback in clusters:
                        if fallback == cluster or any(
                                p % f for p, f in zip(cluster, fallback)):
                            continue
                        fallback_label = f"{fallback[0]}x{fallback[1]}"
                        yield common | {
                            "id": f"mixed_{inst_m}_{inst_n}_{tiler_m}_{label}_{fallback_label}",
                            "cluster": label, "variant": "grouped_mixed",
                            "fallback": fallback_label,
                        }


def candidates(family, size):
    if family == "nvfp4_fp4":
        yield from grouped_candidates(size)
        return
    seen = set()
    base = BASE_FAMILY[family]
    for row in csv_rows(f"{base}.csv"):
        persistent = base == "fp8" or (row.get("variant") or "persistent") == "persistent"
        knobs = []
        if persistent:
            knobs += [("swizzle", (1, 2, 4, 8)), ("raster", ("m", "n"))]
        if base in ("mxfp8", "nvfp4"):
            if persistent:
                knobs += [("scheduler", ("static_persistent", "clc_dynamic_persistent"))]
            knobs += [("prefetch", ("auto", 0, 1, 2, 3, 4))]
        for values in itertools.product(*(values for _, values in knobs)):
            config = row | {key: str(value) for (key, _), value in zip(knobs, values)}
            key = tuple(sorted((k, v) for k, v in config.items() if k != "id"))
            if key not in seen:
                seen.add(key)
                yield config


def pinned(family, size):
    for config in csv_rows("winners.csv"):
        if config["family"] == family and int(config["size"]) == size:
            return config
    raise SystemExit(f"no pinned CuTeDSL winner for {family} {size}; run tune")


def measure(family, size, config, report_failure=False, screen=False):
    source = HERE / "src" / f"{BASE_FAMILY[family]}.py"
    command = [sys.executable, source, str(size),
               json.dumps(config, separators=(",", ":")), family]
    env = ENV | ({"BASELINE_TUNE_SCREEN": "1"} if screen else {})
    result = subprocess.run(command, text=True, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, env=env)
    match = TIME.search(result.stdout)
    if report_failure and (result.returncode or not match):
        print(result.stdout, end="", file=sys.stderr)
    return float(match.group(1)) if result.returncode == 0 and match else None


def main(argv=sys.argv[1:]):
    if len(argv) not in (2, 3) or (len(argv) == 3 and argv[2] != "tune"):
        raise SystemExit("usage: runner.py FAMILY SIZE [tune]")
    family, size = argv[:2]
    if family not in BASE_FAMILY or (size := int(size)) not in SIZES:
        raise SystemExit("unsupported family or size")

    if len(argv) == 3:
        configs = list(candidates(family, size))
        timings = [(elapsed, config) for config in configs
                   if (elapsed := measure(
                       family, size, config, screen=True)) is not None]
        if not timings:
            raise SystemExit("all candidates failed")
        finalists = sorted(timings, key=lambda item: item[0])[:50]
        finalists = [(elapsed, config) for _, config in finalists
                     if (elapsed := measure(family, size, config)) is not None]
        top = sorted(finalists, key=lambda item: item[0])[:10]
        ranked = []
        for _, config in top:
            repeats = [measure(family, size, config) for _ in range(3)]
            if all(value is not None for value in repeats):
                ranked.append((statistics.median(repeats), config))
        if not ranked:
            raise SystemExit("all top candidates failed")
        _, winner = min(ranked, key=lambda item: item[0])
        print(f"SEARCH attempted={len(configs)} supported={len(timings)} screen=1+3 "
              f"finalists={len(finalists)}x5+10 top={len(top)}x3x5+10")
        print("ALGORITHM " + " ".join(f"{key}={value}" for key, value in winner.items()))
    else:
        winner = pinned(family, size)

    samples = [measure(family, size, winner, True) for _ in range(SAMPLES)]
    if any(value is None for value in samples):
        raise SystemExit("selected configuration failed")
    mean = sum(samples) / len(samples)
    values = ",".join(f"{value:.6f}" for value in samples)
    print(f"RESULT size={size} mean_tflops={2 * size**3 / (mean * 1e6):.6f} "
          f"samples_us={values}")


if __name__ == "__main__":
    main()
