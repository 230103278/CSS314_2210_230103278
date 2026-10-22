import os
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np

STUDENT_ID = 230103278
N = 30_000_000
MAX_CORES = os.cpu_count()
CORES = [1, 2, MAX_CORES]
REPEATS = 5


def copy_kernel(a, b):
    np.copyto(b, a)


def trig_kernel(a, b):
    np.sin(a, out=b)
    b *= np.cos(a)
    b += np.tan(a)


def best_ms(kernel, a, b, workers):
    a_chunks, b_chunks = np.array_split(a, workers), np.array_split(b, workers)
    best = float("inf")
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for _ in range(REPEATS):
            start = time.perf_counter()
            list(pool.map(kernel, a_chunks, b_chunks))
            best = min(best, time.perf_counter() - start)
    return best * 1000


a = np.random.default_rng(STUDENT_ID).random(N)
b = np.empty(N)
reference = {"A: copy": a.copy(), "B: sin*cos+tan": np.sin(a) * np.cos(a) + np.tan(a)}

print(f"Student ID: {STUDENT_ID} | N = {N:,} float64 | Max cores: {MAX_CORES} | best of {REPEATS} runs\n")
print(f"{'Kernel':<16} | {'Cores':>5} | {'Wall (ms)':>10} | {'Speedup':>8} | {'Efficiency':>10} | {'RAM traffic':>11}")
print("-" * 78)

efficiency_at_max = {}
for (name, kernel) in [("A: copy", copy_kernel), ("B: sin*cos+tan", trig_kernel)]:
    t1 = None
    for cores in CORES:
        ms = best_ms(kernel, a, b, cores)
        assert np.allclose(b, reference[name]), f"{name} wrong result"
        t1 = t1 or ms
        speedup = t1 / ms
        efficiency = speedup / cores
        gbps = 2 * a.nbytes / (ms / 1000) / 1e9  # read A + write B
        print(f"{name:<16} | {cores:>5} | {ms:>10.2f} | {speedup:>7.2f}x | {efficiency:>9.1%} | {gbps:>6.1f} GB/s")
    efficiency_at_max[name] = efficiency
    print("-" * 78)

stalled = min(efficiency_at_max, key=efficiency_at_max.get)
scaled = max(efficiency_at_max, key=efficiency_at_max.get)
print(f"\nVerdict: Kernel {stalled} stalled in scaling due to memory channel saturation "
      f"({efficiency_at_max[stalled]:.1%} efficiency at {MAX_CORES} cores).")
print(f"Kernel {scaled} is compute-bound and kept scaling "
      f"({efficiency_at_max[scaled]:.1%} efficiency at {MAX_CORES} cores).")
