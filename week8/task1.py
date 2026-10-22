import os
import platform
import time

import numpy as np

STUDENT_ID = 230103278
N = 20_000_000


def cpu_model():
    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or platform.machine()


print(f"Student ID: {STUDENT_ID}")
print(f"OS: {platform.system()} {platform.release()} ({platform.platform()})")
print(f"CPU: {cpu_model()}")
print(f"Logical cores: {os.cpu_count()}")

data = np.random.default_rng(STUDENT_ID).random(N, dtype=np.float64)
print(f"Array: {data.size:,} float64 elements")

start = time.perf_counter()
total = data.sum()
std = data.std()
elapsed_ms = (time.perf_counter() - start) * 1000

assert np.isclose(std, np.sqrt(np.mean((data - data.mean()) ** 2)))
print(f"Sum: {total:.6f}")
print(f"Std dev: {std:.6f}")
print(f"Serial execution time: {elapsed_ms:.2f} ms")
