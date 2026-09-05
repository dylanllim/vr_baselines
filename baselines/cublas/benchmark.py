#!/usr/bin/env python3
import argparse
from pathlib import Path
import re
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from protocol import SAMPLES, SIZES

RESULT = re.compile(r"Performance: ([0-9.eE+-]+) TFLOP/s")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--size", type=int, choices=SIZES, required=True)
    parser.add_argument("--tune", action="store_true")
    args = parser.parse_args()
    samples = []
    for rep in range(SAMPLES):
        command = [str(args.binary.resolve()), str(args.size)]
        if args.tune:
            command.append("--tune")
        run = subprocess.run(
            command, text=True,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        print(run.stdout, end="")
        match = RESULT.search(run.stdout)
        if run.returncode or not match:
            raise SystemExit(f"sample {rep + 1} failed (exit {run.returncode})")
        samples.append(float(match.group(1)))
    print(f"SAMPLES_TFLOPS={','.join(map(str, samples))}")
    print(f"Performance: {sum(samples) / len(samples):.6f} TFLOP/s")


if __name__ == "__main__":
    main()
