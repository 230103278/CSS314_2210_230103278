import math
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np

STUDENT_ID = 230103278
SAMPLES = 40_000_000
THREADS = 4
BATCH = 100_000
REPEATS = 3
PER_THREAD = SAMPLES // THREADS


def count_hits(xy):
    return np.count_nonzero(xy[0] * xy[0] + xy[1] * xy[1] <= 1.0)


def shared_prng():
    rng = np.random.default_rng(STUDENT_ID)
    lock = threading.Lock()

    def worker(_):
        hits = 0
        for _ in range(PER_THREAD // BATCH):
            with lock:
                xy = rng.random((2, BATCH))
            hits += count_hits(xy)
        return hits

    return worker


def independent_prng():
    def worker(t):
        rng = np.random.default_rng(STUDENT_ID * 10 + t)
        return sum(count_hits(rng.random((2, BATCH))) for _ in range(PER_THREAD // BATCH))

    return worker


def best_run(make_worker):
    best = float("inf")
    for _ in range(REPEATS):
        worker = make_worker()
        with ThreadPoolExecutor(max_workers=THREADS) as pool:
            start = time.perf_counter()
            hits = sum(pool.map(worker, range(THREADS)))
            best = min(best, time.perf_counter() - start)
    return 4 * hits / SAMPLES, best * 1000


print(f"Student ID: {STUDENT_ID} | Samples: {SAMPLES:,} | Threads: {THREADS} | Batch: {BATCH:,} | best of {REPEATS} runs\n")

shared_pi, shared_ms = best_run(shared_prng)
indep_pi, indep_ms = best_run(independent_prng)

for pi in (shared_pi, indep_pi):
    assert abs(pi - math.pi) < 2e-3, "estimate too far from pi"

print(f"Shared PRNG (mutex):             pi = {shared_pi:.8f}  |error| = {abs(shared_pi - math.pi):.8f}  {shared_ms:>9.2f} ms")
print(f"Independent PRNG (seed=ID*10+t): pi = {indep_pi:.8f}  |error| = {abs(indep_pi - math.pi):.8f}  {indep_ms:>9.2f} ms")
print(f"Speedup from removing generator locking: {shared_ms / indep_ms:.2f}x")
