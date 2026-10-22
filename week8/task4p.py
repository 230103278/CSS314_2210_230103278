import time
from multiprocessing import Pipe, Process, shared_memory

import numpy as np

STUDENT_ID = 230103278
SIZES = [100_000, 1_000_000, 10_000_000]
REPEATS = 5


def consumer(conn):
    while (msg := conn.recv()) is not None:
        if msg[0] == "pickle":
            arr = msg[1]
            conn.send((arr.size, arr[0], arr[-1]))
        else:
            _, name, n = msg
            shm = shared_memory.SharedMemory(name=name)
            arr = np.ndarray((n,), dtype=np.float64, buffer=shm.buf)  # view, no copy
            conn.send((arr.size, arr[0], arr[-1]))
            del arr
            shm.close()


def best_ms(conn, msg):
    best = float("inf")
    for _ in range(REPEATS):
        start = time.perf_counter()
        conn.send(msg)
        reply = conn.recv()
        best = min(best, time.perf_counter() - start)
    return best * 1000, reply


if __name__ == "__main__":
    parent, child = Pipe()
    proc = Process(target=consumer, args=(child,))
    proc.start()
    rng = np.random.default_rng(STUDENT_ID)

    print(f"Student ID: {STUDENT_ID} | float64 arrays sent to a child process | best of {REPEATS} round trips")
    print("Path A: pickle through multiprocessing.Pipe | Path B: shared_memory, only the name crosses the pipe\n")
    print(f"{'N':>12} | {'Size (MB)':>9} | {'A: Pickle (ms)':>14} | {'B: Shared (ms)':>14} | {'A / B':>8}")
    print("-" * 70)

    try:
        for n in SIZES:
            x = rng.random(n)
            expected = (n, x[0], x[-1])

            a_ms, a_reply = best_ms(parent, ("pickle", x))

            shm = shared_memory.SharedMemory(create=True, size=x.nbytes)
            try:
                np.ndarray(x.shape, dtype=x.dtype, buffer=shm.buf)[:] = x
                b_ms, b_reply = best_ms(parent, ("shm", shm.name, n))
            finally:
                shm.close()
                shm.unlink()

            assert a_reply == expected and b_reply == expected, "child saw wrong data"
            print(f"{n:>12,} | {x.nbytes / 1e6:>9.1f} | {a_ms:>14.3f} | {b_ms:>14.3f} | {a_ms / b_ms:>7.1f}x")
    finally:
        parent.send(None)
        proc.join()
