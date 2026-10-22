import ctypes
import os
import subprocess
import tempfile
import time
from multiprocessing import Lock, Process, shared_memory

STUDENT_ID = 230103278
WORKERS = 8
INCREMENTS = 1_000_000
EXPECTED = WORKERS * INCREMENTS
REPEATS = 3

CAS_SOURCE = r"""
#include <stdint.h>
void cas_increment(int64_t *p) {
    int64_t old = __atomic_load_n(p, __ATOMIC_RELAXED);
    while (!__atomic_compare_exchange_n(p, &old, old + 1, 1, __ATOMIC_SEQ_CST, __ATOMIC_RELAXED)) {
        /* failed: another core won, old now holds the fresh value, retry */
    }
}
"""


def mutex_worker(shm_name, lock):
    shm = shared_memory.SharedMemory(name=shm_name)
    counter = shm.buf.cast("q")
    for _ in range(INCREMENTS):
        with lock:
            counter[0] += 1
    counter.release()
    shm.close()


def cas_worker(shm_name, so_path):
    shm = shared_memory.SharedMemory(name=shm_name)
    cell = ctypes.c_int64.from_buffer(shm.buf)
    lib = ctypes.CDLL(so_path)
    lib.cas_increment.argtypes = [ctypes.c_void_p]
    lib.cas_increment.restype = None
    increment, addr = lib.cas_increment, ctypes.addressof(cell)
    for _ in range(INCREMENTS):
        increment(addr)
    del cell
    shm.close()


def run(target, extra):
    shm = shared_memory.SharedMemory(create=True, size=8)
    counter = shm.buf.cast("q")
    try:
        counter[0] = 0
        procs = [Process(target=target, args=(shm.name, extra)) for _ in range(WORKERS)]
        start = time.perf_counter()
        for p in procs:
            p.start()
        for p in procs:
            p.join()
        ms = (time.perf_counter() - start) * 1000
        assert counter[0] == EXPECTED, f"lost updates: {counter[0]:,} != {EXPECTED:,}"
        return ms
    finally:
        counter.release()
        shm.close()
        shm.unlink()


if __name__ == "__main__":
    with tempfile.TemporaryDirectory() as tmp:
        src, so_path = os.path.join(tmp, "cas.c"), os.path.join(tmp, "cas.so")
        with open(src, "w") as f:
            f.write(CAS_SOURCE)
        subprocess.run(["gcc", "-O2", "-shared", "-fPIC", "-o", so_path, src], check=True)

        print(f"Student ID: {STUDENT_ID} | Workers: {WORKERS} processes | Increments/worker: {INCREMENTS:,} "
              f"| best of {REPEATS} runs\n")

        mutex_ms = min(run(mutex_worker, Lock()) for _ in range(REPEATS))
        cas_ms = min(run(cas_worker, so_path) for _ in range(REPEATS))

    mutex_tp = EXPECTED / (mutex_ms / 1000)
    cas_tp = EXPECTED / (cas_ms / 1000)
    print(f"{'Primitive':<36} | {'Time (ms)':>10} | {'Throughput (ops/sec)':>20}")
    print("-" * 74)
    print(f"{'Mutex (multiprocessing.Lock, futex)':<36} | {mutex_ms:>10.2f} | {mutex_tp:>20,.0f}")
    print(f"{'Atomic CAS loop (lock cmpxchg)':<36} | {cas_ms:>10.2f} | {cas_tp:>20,.0f}")
    print(f"\nThroughput Ratio (CAS / Mutex): {cas_tp / mutex_tp:.2f}x")
    print(f"Both counters verified at {EXPECTED:,}.")
