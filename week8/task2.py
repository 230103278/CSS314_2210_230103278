import os
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np

STUDENT_ID = 230103278
N = 20_000_000
P = 0.95
REPEATS = 3


def amdahl(p, cores):
    return 1 / ((1 - p) + p / cores)


def parallel_sum(data, workers):
    chunks = np.array_split(data, workers)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        return sum(pool.map(np.sum, chunks))


def best_time_ms(data, workers):
    best = float("inf")
    for _ in range(REPEATS):
        start = time.perf_counter()
        total = parallel_sum(data, workers)
        best = min(best, (time.perf_counter() - start) * 1000)
    return best, total


max_cores = os.cpu_count()
data = np.random.default_rng(STUDENT_ID).random(N, dtype=np.float64)
expected = data.sum()

print(f"Student ID: {STUDENT_ID} | Logical cores: {max_cores} | N = {N:,} | P = {P}")
print(f"Wall time = best of {REPEATS} runs\n")
print(f"{'Cores':>6} | {'Wall Time (ms)':>14} | {'Speedup':>8} | {'Amdahl Upper Bound':>18}")
print("-" * 56)

t1 = None
for workers in [1, 2, 4, max_cores, max_cores * 2]:
    ms, total = best_time_ms(data, workers)
    assert np.isclose(total, expected)
    t1 = t1 or ms
    print(f"{workers:>6} | {ms:>14.2f} | {t1 / ms:>7.2f}x | {amdahl(P, workers):>17.2f}x")
