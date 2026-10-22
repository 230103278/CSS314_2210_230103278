import os
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import psutil

STUDENT_ID = 230103278
N = 8_000
WORKERS = psutil.cpu_count(logical=False)
REPEATS = 3


def row_major(m, rows):
    return sum(m[r].sum() for r in rows)


def column_major(m, cols):
    return sum(m[:, c].sum() for c in cols)


def best_ms(kernel, m, pool):
    chunks = np.array_split(np.arange(N), WORKERS)
    best = float("inf")
    for _ in range(REPEATS):
        start = time.perf_counter()
        total = sum(pool.map(lambda idx: kernel(m, idx), chunks))
        best = min(best, time.perf_counter() - start)
    return best * 1000, total


m = np.random.default_rng(STUDENT_ID).random((N, N))
reference = m.sum()

print(f"Student ID: {STUDENT_ID} | Logical cores: {os.cpu_count()} | Physical cores (workers): {WORKERS}")
print(f"Matrix: {N:,} x {N:,} float64 ({m.nbytes / 2**20:,.0f} MiB) | best of {REPEATS} runs\n")

with ThreadPoolExecutor(max_workers=WORKERS) as pool:
    a_ms, a_sum = best_ms(row_major, m, pool)
    b_ms, b_sum = best_ms(column_major, m, pool)

assert np.isclose(a_sum, reference) and np.isclose(b_sum, reference), "sum mismatch"

print(f"Mode A (Row-Major):            {a_ms:>10.2f} ms  sum = {a_sum:,.4f}")
print(f"Mode B (Column-Major/Strided): {b_ms:>10.2f} ms  sum = {b_sum:,.4f}")
print(f"Cache penalty factor (B / A):  {b_ms / a_ms:>10.2f}x")
