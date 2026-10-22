import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import psutil

STUDENT_ID = 230103278
N = 2_000_000
WORKERS = psutil.cpu_count(logical=False)
REPEATS = 5


def serial_scan(x):
    return np.cumsum(x)


def parallel_scan(x, pool):
    out = np.empty_like(x)
    x_chunks, out_chunks = np.array_split(x, WORKERS), np.array_split(out, WORKERS)
    list(pool.map(lambda c: np.cumsum(c[0], out=c[1]), zip(x_chunks, out_chunks)))
    offsets = np.concatenate(([0], np.cumsum([c[-1] for c in out_chunks[:-1]])))
    list(pool.map(lambda c: np.add(c[0], c[1], out=c[0]), zip(out_chunks, offsets)))
    return out


def best_ms(fn):
    best = float("inf")
    for _ in range(REPEATS):
        start = time.perf_counter()
        result = fn()
        best = min(best, time.perf_counter() - start)
    return best * 1000, result


x = np.random.default_rng(STUDENT_ID).integers(1, 11, N)

print(f"Student ID: {STUDENT_ID} | N = {N:,} ints in [1, 10] | Workers: {WORKERS} (physical cores) | best of {REPEATS} runs\n")

with ThreadPoolExecutor(max_workers=WORKERS) as pool:
    serial_ms, serial_out = best_ms(lambda: serial_scan(x))
    parallel_ms, parallel_out = best_ms(lambda: parallel_scan(x, pool))

verified = parallel_out[-1] == serial_out[-1] and np.array_equal(parallel_out, serial_out)

print(f"{'Serial Time (ms)':>16} | {'Parallel Time (ms)':>18} | {'Speedup':>8} | Verification Status")
print("-" * 70)
print(f"{serial_ms:>16.3f} | {parallel_ms:>18.3f} | {serial_ms / parallel_ms:>7.2f}x | {'PASS' if verified else 'FAIL'}")
print(f"\nLast element: serial = {serial_out[-1]:,} | parallel = {parallel_out[-1]:,}")

try:
    assert verified, "parallel scan does not match serial scan"
except AssertionError as e:
    print(f"FAIL: {e}", file=sys.stderr)
    sys.exit(1)
