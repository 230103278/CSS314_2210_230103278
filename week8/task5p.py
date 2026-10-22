import hashlib
import os
import time
from concurrent.futures import ThreadPoolExecutor

STUDENT_ID = 230103278
TASKS = 10_000
BLOCK = STUDENT_ID.to_bytes(8, "little") * (64 * 1024 // 8)
HASHES_PER_TASK = 8
CORES = os.cpu_count()
POOL_SIZES = [1, CORES // 2, CORES, CORES * 2, CORES * 4, CORES * 16, 256]
REPEATS = 3
TOLERANCE = 1.05


def task(_):
    for _ in range(HASHES_PER_TASK):
        digest = hashlib.sha256(BLOCK).digest()
    return digest


def best_ms(workers):
    best = float("inf")
    for _ in range(REPEATS):
        with ThreadPoolExecutor(max_workers=workers) as pool:
            start = time.perf_counter()
            digests = list(pool.map(task, range(TASKS)))
            best = min(best, time.perf_counter() - start)
    assert len(digests) == TASKS and len(set(digests)) == 1, "tasks lost or wrong digest"
    return best * 1000


print(f"Student ID: {STUDENT_ID} | Tasks: {TASKS:,} x sha256({len(BLOCK) // 1024} KB) x {HASHES_PER_TASK} "
      f"| Logical cores: {CORES} | best of {REPEATS} runs\n")

times = [best_ms(p) for p in POOL_SIZES]
t1 = times[0]
best_i = min(range(len(times)), key=times.__getitem__)
worsen_i = next((i for i in range(best_i + 1, len(times)) if times[i] > times[best_i] * TOLERANCE), None)

print(f"{'Pool Size':>9} | {'Execution Time (ms)':>19} | {'Speedup vs 1 thread':>19} | Status")
print("-" * 80)
for i, (p, ms) in enumerate(zip(POOL_SIZES, times)):
    if i < best_i:
        status = "scaling"
    elif i == best_i:
        status = "OPTIMAL (fastest)"
    elif times[i] > times[best_i] * TOLERANCE:
        status = f"DEGRADED (+{ms / times[best_i] - 1:.1%} vs best)"
    else:
        status = "plateau (within 5% of best)"
    if i == worsen_i:
        status += "  <-- worsening starts here"
    print(f"{p:>9} | {ms:>19.2f} | {t1 / ms:>18.2f}x | {status}")

print()
if worsen_i is None:
    print(f"No pool size up to {POOL_SIZES[-1]} was more than 5% slower than the best ({POOL_SIZES[best_i]} threads).")
else:
    print(f"Execution time starts worsening at pool size {POOL_SIZES[worsen_i]} "
          f"(best was {POOL_SIZES[best_i]} threads on {CORES} logical cores).")
