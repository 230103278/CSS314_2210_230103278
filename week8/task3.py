import os
import time
from multiprocessing import Process, shared_memory

STUDENT_ID = 230103278
ITERATIONS = 50_000_000
STRIDE = 16
REPEATS = 3


def physical_cores():
    try:
        cores, phys = set(), "0"
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.startswith("physical id"):
                    phys = line.split(":")[1].strip()
                elif line.startswith("core id"):
                    cores.add((phys, line.split(":")[1].strip()))
        if cores:
            return len(cores)
    except OSError:
        pass
    return os.cpu_count()


def worker(shm_name, idx):
    shm = shared_memory.SharedMemory(name=shm_name)
    counts = shm.buf.cast("q")
    for _ in range(ITERATIONS):
        counts[idx] += 1
    counts.release()
    shm.close()


def run(workers, stride):
    shm = shared_memory.SharedMemory(create=True, size=workers * stride * 8)
    counts = shm.buf.cast("q")
    try:
        procs = [Process(target=worker, args=(shm.name, i * stride)) for i in range(workers)]
        start = time.perf_counter()
        for p in procs:
            p.start()
        for p in procs:
            p.join()
        ms = (time.perf_counter() - start) * 1000
        assert all(counts[i * stride] == ITERATIONS for i in range(workers)), "lost increments"
        return ms
    finally:
        counts.release()
        shm.close()
        shm.unlink()


def best(workers, stride):
    return min(run(workers, stride) for _ in range(REPEATS))


if __name__ == "__main__":
    n = physical_cores()
    print(f"Student ID: {STUDENT_ID} | Physical cores (workers): {n} | Iterations/worker: {ITERATIONS:,}")
    print(f"Wall time = best of {REPEATS} runs\n")

    unpadded = best(n, 1)
    padded = best(n, STRIDE)

    print(f"Unpadded Shared: {unpadded:10.2f} ms")
    print(f"Padded ({STRIDE * 8} B apart) : {padded:10.2f} ms")
    print(f"Ratio: {unpadded / padded:10.2f}x")
