#!/usr/bin/env python3
from pathlib import Path
import re
import subprocess
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from protocol import SAMPLES, SIZES

RESULT = re.compile(r"\bRESULT\b.*\btflops=([0-9.eE+-]+)")


def main(argv=sys.argv[1:]):
    if len(argv) not in (2, 3) or (len(argv) == 3 and argv[2] != "tune"):
        raise SystemExit("usage: runner.py FAMILY SIZE [tune]")
    family, size = argv[:2]
    if family not in ("fp8", "nvfp4") or (size := int(size)) not in SIZES:
        raise SystemExit("unsupported family or size")
    binary = HERE / "build" / family
    if not binary.is_file():
        raise SystemExit(f"missing {binary}; run make build")
    samples = []
    for sample in range(1, SAMPLES + 1):
        command = [str(binary), str(size)]
        if len(argv) == 3:
            command.append("--tune")
        run = subprocess.run(command, text=True, capture_output=True)
        match = RESULT.search(run.stdout)
        if run.returncode or not match:
            print(run.stdout, end="")
            print(run.stderr, end="", file=sys.stderr)
            raise SystemExit(f"sample {sample} failed (exit {run.returncode})")
        if len(argv) == 3:
            selected = next(
                line for line in run.stdout.splitlines()
                if line.startswith("ALGORITHM "))
            print(f"TUNE sample={sample} {selected}")
        samples.append(float(match.group(1)))
    values = ",".join(f"{value:.6f}" for value in samples)
    print(f"RESULT size={size} mean_tflops={sum(samples) / len(samples):.6f} "
          f"samples_tflops={values}")


if __name__ == "__main__":
    main()
