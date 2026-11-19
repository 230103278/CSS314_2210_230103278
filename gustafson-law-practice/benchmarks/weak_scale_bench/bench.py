"""Weak-scaling harness: every worker integrates K midpoints, so P workers solve a P-times larger problem.

The integrand 4/(1+x^2) on [0,1] gives pi, which doubles as a correctness check.
"""
import argparse
import csv
import math
import os
import time
from multiprocessing import Pool

import numpy as np

BLOCK = 1 << 15  # 32K doubles = 256 KiB per temporary, fits in the 1 MiB per-core L2


def integrate(args):
    lo, hi, k = args
    h = (hi - lo) / k
    total = 0.0
    for start in range(0, k, BLOCK):
        x = lo + h * (np.arange(start, min(start + BLOCK, k)) + 0.5)
        total += np.sum(4.0 / (1.0 + x * x))
    return total * h


def run(pool, p, k):
    chunks = [(i / p, (i + 1) / p, k) for i in range(p)]
    start = time.perf_counter()
    estimate = sum(pool.map(integrate, chunks, chunksize=1))  # the serial reduction is part of the timed region
    return time.perf_counter() - start, estimate


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--k", type=int, default=1 << 27, help="midpoints per worker (constant under weak scaling)")
    ap.add_argument("--runs", type=int, default=5)
    ap.add_argument("--max-workers", type=int, default=os.cpu_count())
    ap.add_argument("--out", default=os.path.join(here, "..", "results.csv"))
    a = ap.parse_args()
    if a.k < 1 or a.runs < 1 or a.max_workers < 1:
        ap.error("--k, --runs and --max-workers must be positive")

    with open(a.out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["workers", "run", "seconds", "pi_estimate"])
        for p in range(1, a.max_workers + 1):
            with Pool(p) as pool:
                run(pool, p, a.k)  # untimed full-size warm-up: forks workers and ramps the CPU clock out of idle
                for r in range(1, a.runs + 1):
                    t, pi = run(pool, p, a.k)
                    assert abs(pi - math.pi) < 1e-9, pi
                    w.writerow([p, r, f"{t:.6f}", f"{pi:.12f}"])
                    print(f"P={p:2d} run={r} t={t:.4f}s pi={pi:.12f}", flush=True)


if __name__ == "__main__":
    main()
