import queue
import threading
import time

STUDENT_ID = 230103278
ITEMS = 1000
WORKERS = 4


def work(i):
    time.sleep(0.0001 * (i % 20))


def run(make_tasks):
    next_chunk = make_tasks()
    finish = [0.0] * WORKERS
    done = [0] * WORKERS

    def worker(w):
        while (chunk := next_chunk(w)) is not None:
            for i in chunk:
                work(i)
            done[w] += len(chunk)
        finish[w] = time.perf_counter()

    threads = [threading.Thread(target=worker, args=(w,)) for w in range(WORKERS)]
    start = time.perf_counter()
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    total = (time.perf_counter() - start) * 1000
    assert sum(done) == ITEMS, "items lost or duplicated"
    ends = [(f - start) * 1000 for f in finish]
    return total, ends, done


def static_blocks():
    size = ITEMS // WORKERS
    given = [False] * WORKERS

    def next_chunk(w):
        if given[w]:
            return None
        given[w] = True
        return range(w * size, ITEMS if w == WORKERS - 1 else (w + 1) * size)

    return next_chunk


def dynamic(chunk_size):
    def make():
        q = queue.Queue()
        for s in range(0, ITEMS, chunk_size):
            q.put(range(s, min(s + chunk_size, ITEMS)))

        def next_chunk(_):
            try:
                return q.get_nowait()
            except queue.Empty:
                return None

        return next_chunk

    return make


print(f"Student ID: {STUDENT_ID} | Items: {ITEMS} | Workers: {WORKERS} | cost(i) = sleep(0.0001 * (i % 20))\n")
print(f"{'Strategy':<26} | {'Total (ms)':>10} | {'First done':>10} | {'Last done':>10} | {'Delta (ms)':>10} | Items/worker")
print("-" * 100)
for name, strategy in [
    ("A: Static blocks (250)", static_blocks),
    ("B: Dynamic queue, chunk=1", dynamic(1)),
    ("B: Dynamic queue, chunk=5", dynamic(5)),
]:
    total, ends, done = run(strategy)
    print(f"{name:<26} | {total:>10.2f} | {min(ends):>10.2f} | {max(ends):>10.2f} | "
          f"{max(ends) - min(ends):>10.2f} | {done}")
