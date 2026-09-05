#!/usr/bin/env python3
"""Single source of truth for every reported benchmark sample."""

SIZES = (1024, 2048, 4096, 8192, 16384, 32768)
SEED = 2024
WARMUPS = 5
ITERATIONS = 10
SAMPLES = 5
COLD_L2 = True
FP8_RANGE = (-448.0, 448.0)
NVFP4_RANGE = (-6.0, 6.0)
NVFP4_CODES = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
                -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0)
NVFP4_PACKED_RANGE = (0.0, 255.0)
NVFP4_SCALE_RANGE = (0.1, 10.0)


def c_header():
    values = {
        "BENCHMARK_SIZES": "{" + ",".join(map(str, SIZES)) + "}",
        "BENCHMARK_SEED": f"{SEED}ULL",
        "BENCHMARK_WARMUPS": WARMUPS,
        "BENCHMARK_ITERATIONS": ITERATIONS,
        "BENCHMARK_SAMPLES": SAMPLES,
        "BENCHMARK_COLD_L2": int(COLD_L2),
        "BENCHMARK_FP8_LOW": f"{FP8_RANGE[0]}f",
        "BENCHMARK_FP8_HIGH": f"{FP8_RANGE[1]}f",
        "BENCHMARK_NVFP4_PACKED_LOW": f"{NVFP4_PACKED_RANGE[0]}f",
        "BENCHMARK_NVFP4_PACKED_HIGH": f"{NVFP4_PACKED_RANGE[1]}f",
        "BENCHMARK_NVFP4_SCALE_LOW": f"{NVFP4_SCALE_RANGE[0]}f",
        "BENCHMARK_NVFP4_SCALE_HIGH": f"{NVFP4_SCALE_RANGE[1]}f",
    }
    return "#pragma once\n" + "".join(f"#define {k} {v}\n" for k, v in values.items())


if __name__ == "__main__":
    print(c_header(), end="")
