#!/usr/bin/env python3
import csv
import itertools
import json
import os
from pathlib import Path
import re
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


def csv_rows(name):
    with (CSV_DIR / name).open(newline="") as source:
        return list(csv.DictReader(source))


def candidates(family):
    seen = set()
    for row in csv_rows(f"{family}.csv"):
        persistent = family == "fp8" or (row.get("variant") or "persistent") == "persistent"
        knobs = []
        if persistent:
            knobs += [("swizzle", (1, 2, 4, 8)), ("raster", ("m", "n"))]
        if family == "nvfp4":
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


def measure(family, size, config, report_failure=False):
    command = [sys.executable, HERE / "src" / f"{family}.py", str(size),
               json.dumps(config, separators=(",", ":"))]
    result = subprocess.run(command, text=True, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, env=ENV)
    match = TIME.search(result.stdout)
    if report_failure and (result.returncode or not match):
        print(result.stdout, end="", file=sys.stderr)
    return float(match.group(1)) if result.returncode == 0 and match else None


def main(argv=sys.argv[1:]):
    if len(argv) not in (2, 3) or (len(argv) == 3 and argv[2] != "tune"):
        raise SystemExit("usage: runner.py FAMILY SIZE [tune]")
    family, size = argv[:2]
    if family not in ("fp8", "nvfp4") or (size := int(size)) not in SIZES:
        raise SystemExit("unsupported family or size")

    if len(argv) == 3:
        timings = [(elapsed, config) for config in candidates(family)
                   if (elapsed := measure(family, size, config)) is not None]
        if not timings:
            raise SystemExit("all candidates failed")
        _, winner = min(timings, key=lambda item: item[0])
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
