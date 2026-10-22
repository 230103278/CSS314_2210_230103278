import os
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np

STUDENT_ID = 230103278
SIZES = [100, 1_000, 10_000, 100_000, 1_000_000, 10_000_000]
A, B = 2.5, 1.0
WORKERS = os.cpu_count()


def axpb(x, y):
    np.multiply(x, A, out=y)
    np.add(y, B, out=y)


def serial(x, y):
    axpb(x, y)


def parallel(pool, xs, ys):
    list(pool.map(axpb, xs, ys))


def best_ms(fn, repeats):
    best = float("inf")
    for _ in range(repeats):
        start = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - start)
    return best * 1000


rng = np.random.default_rng(STUDENT_ID)
print(f"Student ID: {STUDENT_ID} | Workers: {WORKERS} threads (persistent pool) | y = {A} * x + {B}")
print("Wall time = best of many run\n")
print(f"{'N':>12} | {'Serial (ms)':>12} | {'Parallel (ms)':>13} | {'Speedup':>8} | Winner")
print("-" * 66)

faster = []
with ThreadPoolExecutor(max_workers=WORKERS) as pool:
    for n in SIZES:
        x = rng.random(n)
        y_serial, y_parallel = np.empty(n), np.empty(n)
        xs, ys = np.array_split(x, WORKERS), np.array_split(y_parallel, WORKERS)
        repeats = max(5, min(2000, 10_000_000 // n))

        t_serial = best_ms(lambda: serial(x, y_serial), repeats)
        t_parallel = best_ms(lambda: parallel(pool, xs, ys), repeats)
        assert np.array_equal(y_serial, y_parallel), "parallel result differs"

        faster.append(t_parallel < t_serial)
        print(f"{n:>12,} | {t_serial:>12.4f} | {t_parallel:>13.4f} | {t_serial / t_parallel:>7.2f}x | "
              f"{'parallel' if faster[-1] else 'serial'}")

k = next((i for i in range(len(SIZES)) if all(faster[i:])), None)
print()
if k is None:
    print(f"Parallel never stays faster than serial up to N = {SIZES[-1]:,} on this CPU.")
elif k == 0:
    print(f"Parallel is already faster at the smallest size tested (N = {SIZES[0]:,}).")
else:
    print(f"Crossover: parallel turns from SLOWER to FASTER between N = {SIZES[k - 1]:,} and N = {SIZES[k]:,}.")
