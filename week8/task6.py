import threading
import time
from multiprocessing import Lock, Process, shared_memory

STUDENT_ID = 230103278
WORKERS = 4
INCREMENTS = 2_000_000
EXPECTED = WORKERS * INCREMENTS


def racy(shm_name, _lock):
    shm = shared_memory.SharedMemory(name=shm_name)
    counter = shm.buf.cast("q")
    for _ in range(INCREMENTS):
        counter[0] += 1
    counter.release()
    shm.close()


def locked(shm_name, lock):
    shm = shared_memory.SharedMemory(name=shm_name)
    counter = shm.buf.cast("q")
    for _ in range(INCREMENTS):
        with lock:
            counter[0] += 1
    counter.release()
    shm.close()


def run_processes(target):
    shm = shared_memory.SharedMemory(create=True, size=8)
    counter = shm.buf.cast("q")
    try:
        lock = Lock()
        procs = [Process(target=target, args=(shm.name, lock)) for _ in range(WORKERS)]
        start = time.perf_counter()
        for p in procs:
            p.start()
        for p in procs:
            p.join()
        return counter[0], (time.perf_counter() - start) * 1000
    finally:
        counter.release()
        shm.close()
        shm.unlink()


def run_threads():
    counter = [0]

    def work():
        for _ in range(INCREMENTS):
            counter[0] += 1

    threads = [threading.Thread(target=work) for _ in range(WORKERS)]
    start = time.perf_counter()
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    return counter[0], (time.perf_counter() - start) * 1000


def report(name, actual, ms):
    lost = EXPECTED - actual
    print(f"{name:<32} | {EXPECTED:>10,} | {actual:>10,} | {lost / EXPECTED:>7.2%} | {ms:>10.2f}")


if __name__ == "__main__":
    print(f"Student ID: {STUDENT_ID} | Workers: {WORKERS} | Increments/worker: {INCREMENTS:,}\n")
    print(f"{'Version':<32} | {'Expected':>10} | {'Actual':>10} | {'Lost':>7} | {'Wall (ms)':>10}")
    print("-" * 82)

    racy_value, racy_ms = run_processes(racy)
    locked_value, locked_ms = run_processes(locked)
    thread_value, thread_ms = run_threads()

    report("Processes, no lock (race)", racy_value, racy_ms)
    report("Processes, multiprocessing.Lock", locked_value, locked_ms)
    report("Threads, no lock (GIL, ref.)", thread_value, thread_ms)

    assert locked_value == EXPECTED, "locked version lost updates"
    print(f"\nLock penalty: +{locked_ms - racy_ms:.2f} ms ({locked_ms / racy_ms:.2f}x slower than the racy version)")
