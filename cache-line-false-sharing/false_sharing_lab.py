import os
import re
import sys
import json
import time
import hashlib
import platform
import threading
import subprocess
from datetime import datetime, timezone
from statistics import median

import numpy as np
from numba import njit, uint64

try:
    import resource
except ImportError:
    resource = None

# Replace with your exact Git handle before executing
GIT_USERNAME = "230103278"
NUM_THREADS = 4
ITERATIONS_PER_THREAD = 60_000_000

STRIDES = (1, 4, 8, 16)
REPEATS = 3  # each configuration is timed REPEATS times; the median is reported
BUFFER_ALIGN = 128  # satisfies the 64-byte requirement and also aligns 128-byte adjacent-line pairs
GIL_DEMO_TOTAL_UPDATES = 8_000_000  # pure-Python contrast for Q5 (bytecode is ~100x slower)
MANIFEST_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache_lab_manifest.json")


def detect_cache_line_size() -> int:
    """Query L1 data cache coherency line size in bytes from OS sysfs/sysctl."""
    linux_path = "/sys/devices/system/cpu/cpu0/cache/index0/coherency_line_size"
    if os.path.exists(linux_path):
        with open(linux_path, "r", encoding="utf-8") as f:
            return int(f.read().strip())
    if sys.platform == "darwin":
        out = subprocess.check_output(["sysctl", "-n", "hw.cachelinesize"], encoding="utf-8")
        return int(out.strip())
    return 64


def read_cpu_model() -> str:
    try:
        with open("/proc/cpuinfo", encoding="utf-8") as f:
            for line in f:
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or platform.machine()


def pick_distinct_physical_cores(n):
    """One logical CPU per physical core: SMT siblings share an L1, which would hide false sharing."""
    if not hasattr(os, "sched_setaffinity"):
        return None
    available = sorted(os.sched_getaffinity(0))
    chosen, seen = [], set()
    for cpu in available:
        try:
            with open(f"/sys/devices/system/cpu/cpu{cpu}/topology/thread_siblings_list", encoding="utf-8") as f:
                core = f.read().strip()
        except OSError:
            core = str(cpu)
        if core not in seen:
            seen.add(core)
            chosen.append(cpu)
    return chosen[:n] if len(chosen) >= n else available[:n]


# ==============================================================================
# TASK 1: Hardware Topology & Student Identity Context
# ==============================================================================
def init_system_context(git_user: str) -> dict:
    assert git_user != "REPLACE_WITH_YOUR_GIT_USERNAME" and len(git_user.strip()) > 0, (
        "Set GIT_USERNAME to your real Git handle before running."
    )
    logical_cores = os.cpu_count() or 1
    if logical_cores < NUM_THREADS:
        raise RuntimeError(f"Requires at least {NUM_THREADS} logical cores; detected {logical_cores}.")

    line_size = detect_cache_line_size()
    cpu_sig = f"{platform.machine()}::{platform.processor()}::{logical_cores}"

    digest = hashlib.sha256(f"{git_user.strip()}::{cpu_sig}".encode("utf-8")).hexdigest()
    return {
        "git_user": git_user.strip(),
        "logical_cores": logical_cores,
        "line_size": line_size,
        "cpu_sig": cpu_sig,
        "cpu_model": read_cpu_model(),
        "instance_seed": int(digest[:16], 16),
        "affinity_cpus": pick_distinct_physical_cores(NUM_THREADS),
    }


def thread_seeds(instance_seed: int):
    # kept below 2**63 so Numba types them as int64 (mixing uint64 and int64 promotes to float64)
    return [(instance_seed + t * 0x9E3779B97F4A7C15) % (1 << 63) for t in range(NUM_THREADS)]


def expected_counters(seeds, iters):
    return [iters * (s % 7 + 1) for s in seeds]


def aligned_uint64_buffer(n_elems: int, align: int) -> np.ndarray:
    raw = np.zeros(n_elems * 8 + align, dtype=np.uint8)
    offset = (-raw.ctypes.data) % align
    return raw[offset:offset + n_elems * 8].view(np.uint64)  # the view keeps raw alive


# ==============================================================================
# TASK 2: GIL-Free Native Counter Kernel
# ==============================================================================
@njit(nogil=True)
def worker_strided_accumulate(buffer, slot, iters, seed):
    inc = uint64(seed % 7 + 1)
    for _ in range(iters):
        v = buffer[slot]
        # v never reaches 2**63, so (v >> 63) is 0 at run time, but LLVM cannot prove it. The store
        # address therefore depends on the loaded value, which blocks register promotion (LICM):
        # every iteration really loads and stores buffer[slot].
        buffer[slot + np.int64(v >> uint64(63))] = v + inc


# ==============================================================================
# TASK 4: Thread-Local Register Privatization Baseline
# ==============================================================================
@njit(nogil=True)
def worker_private_accumulate(buffer, slot, iters, seed):
    inc = uint64(seed % 7 + 1)
    acc = buffer[slot]
    for _ in range(iters):
        acc = acc + inc + (acc >> uint64(63))  # same loop-carried dependency, kept in a register
    buffer[slot] = acc  # the only store to shared memory


def python_strided_accumulate(buffer, slot, iters, seed):
    """Plain CPython bytecode version of the same update, used only for the Q5 GIL contrast."""
    inc = np.uint64(seed % 7 + 1)
    for _ in range(iters):
        buffer[slot] += inc


def timed_parallel(kernel, buffer, slots, seeds, iters, cpus):
    """Start one threading.Thread per slot, release them together, return wall ns and context switches."""
    barrier = threading.Barrier(len(slots) + 1)

    def body(t):
        if cpus:
            os.sched_setaffinity(0, {cpus[t]})  # pid 0 = the calling thread on Linux
        barrier.wait()
        kernel(buffer, slots[t], iters, seeds[t])

    threads = [threading.Thread(target=body, args=(t,)) for t in range(len(slots))]
    for th in threads:
        th.start()
    ru0 = resource.getrusage(resource.RUSAGE_SELF) if resource else None
    barrier.wait()
    t0 = time.perf_counter_ns()
    for th in threads:
        th.join()
    elapsed = time.perf_counter_ns() - t0
    ru1 = resource.getrusage(resource.RUSAGE_SELF) if resource else None
    if ru0 is None:
        return elapsed, None, None
    return elapsed, ru1.ru_nvcsw - ru0.ru_nvcsw, ru1.ru_nivcsw - ru0.ru_nivcsw


def run_layout(ctx, kernel, stride, iters=ITERATIONS_PER_THREAD, repeats=REPEATS):
    seeds = thread_seeds(ctx["instance_seed"])
    slots = [t * stride for t in range(NUM_THREADS)]
    runs = []
    for _ in range(repeats):
        buf = aligned_uint64_buffer(NUM_THREADS * max(STRIDES) + 8, BUFFER_ALIGN)  # fresh zeroed counters
        ns, vcs, ivcs = timed_parallel(kernel, buf, slots, seeds, iters, ctx["affinity_cpus"])
        runs.append({"elapsed_ns": ns, "voluntary_ctx_switches": vcs, "involuntary_ctx_switches": ivcs,
                     "final_counters": [int(buf[s]) for s in slots]})
    base = buf.ctypes.data
    med = int(median(r["elapsed_ns"] for r in runs))
    return {
        "stride": stride,
        "slot_byte_offsets": [8 * s for s in slots],
        "cache_lines": [(base + 8 * s) // ctx["line_size"] - base // ctx["line_size"] for s in slots],
        "line_pairs_128B": [(base + 8 * s) // 128 - base // 128 for s in slots],
        "median_ns": med,
        "ns_per_update": med / iters,
        "runs": runs,
    }


# ==============================================================================
# TASK 3: Multi-Threaded Stride Sweep
# ==============================================================================
def run_stride_benchmark(ctx) -> dict:
    return {s: run_layout(ctx, worker_strided_accumulate, s) for s in STRIDES}


def run_private_baseline(ctx) -> dict:
    return run_layout(ctx, worker_private_accumulate, 1)  # packed slots: only the final store is shared


def run_gil_contrast(ctx) -> dict:
    """Same stride=1 update in plain Python bytecode: same total work on 1 thread vs 4 threads."""
    seeds = thread_seeds(ctx["instance_seed"])
    out = {}
    for n_threads in (1, NUM_THREADS):
        buf = aligned_uint64_buffer(NUM_THREADS + 8, BUFFER_ALIGN)
        iters = GIL_DEMO_TOTAL_UPDATES // n_threads
        ns, vcs, ivcs = timed_parallel(python_strided_accumulate, buf, list(range(n_threads)),
                                       seeds, iters, ctx["affinity_cpus"])
        out[f"{n_threads}_threads"] = {"elapsed_ns": ns, "updates_per_thread": iters,
                                       "ns_per_update_wall": ns / GIL_DEMO_TOTAL_UPDATES,
                                       "voluntary_ctx_switches": vcs, "involuntary_ctx_switches": ivcs}
    out["slowdown_4_vs_1"] = out[f"{NUM_THREADS}_threads"]["elapsed_ns"] / out["1_threads"]["elapsed_ns"]
    return out


# ==============================================================================
# TASK 5: Exact State Parity & LLVM IR Verification
# ==============================================================================
def llvm_audit(dispatcher) -> dict:
    ir = "\n".join(dispatcher.inspect_llvm().values())
    functions = re.split(r"\n(?=define )", ir)
    name = dispatcher.py_func.__name__
    # the native function, not Numba's "cpython" wrapper that boxes arguments
    core = [f for f in functions if f.startswith("define") and name in f.split("(")[0]
            and "cpython" not in f.split("(")[0]]
    body = "\n".join(core)
    return {
        "native_functions": len(core),
        "store_i64": len(re.findall(r"\bstore i64\b", body)),
        "load_i64": len(re.findall(r"= load i64\b", body)),
        "python_api_calls": len(re.findall(r"@_?Py\w+", body)),
        "loop_store_lines": [ln.strip() for ln in body.splitlines() if re.search(r"\b(store|load) i64\b", ln)],
    }


def audit_llvm_and_parity(ctx, sweep, private, gil) -> dict:
    expected = expected_counters(thread_seeds(ctx["instance_seed"]), ITERATIONS_PER_THREAD)
    layouts = list(sweep.values()) + [private]
    parity = all(r["final_counters"] == expected for lay in layouts for r in lay["runs"])
    llvm = llvm_audit(worker_strided_accumulate)
    llvm_ok = llvm["native_functions"] >= 1 and llvm["store_i64"] >= 1 and llvm["python_api_calls"] == 0

    s1, s8 = sweep[1]["median_ns"], sweep[8]["median_ns"]
    manifest = {
        "git_username": ctx["git_user"],
        "instance_seed": ctx["instance_seed"],
        "cpu_model": ctx["cpu_model"],
        "cpu_sig": ctx["cpu_sig"],
        "python": platform.python_version(),
        "logical_cores": ctx["logical_cores"],
        "cache_line_size": ctx["line_size"],
        "affinity_cpus": ctx["affinity_cpus"],
        "num_threads": NUM_THREADS,
        "iterations_per_thread": ITERATIONS_PER_THREAD,
        "repeats": REPEATS,
        "strides": {str(k): v for k, v in sweep.items()},
        "private_baseline": private,
        "slowdown_stride1_vs_stride8": s1 / s8,
        "slowdown_stride4_vs_stride8": sweep[4]["median_ns"] / s8,
        "ratio_stride16_vs_stride8": sweep[16]["median_ns"] / s8,
        "speedup_private_vs_stride8": s8 / private["median_ns"],
        "gil_contrast_pure_python": gil,
        "expected_final_counters": expected,
        "parity_passed": parity,
        "llvm_audit": llvm,
        "llvm_audit_passed": llvm_ok,
        "verification_passed": parity and llvm_ok,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
    }
    with open(MANIFEST_PATH, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    return manifest


def main() -> int:
    ctx = init_system_context(GIT_USERNAME)
    print("=" * 78)
    print(f"Git user        : {ctx['git_user']}")
    print(f"CPU             : {ctx['cpu_model']} | {ctx['logical_cores']} logical cores")
    print(f"cpu_sig         : {ctx['cpu_sig']}")
    print(f"INSTANCE_SEED   : {ctx['instance_seed']} (0x{ctx['instance_seed']:016x})")
    print(f"Cache line      : {ctx['line_size']} B | pinned CPUs: {ctx['affinity_cpus']}")
    print(f"Workload        : {NUM_THREADS} threads x {ITERATIONS_PER_THREAD:,} updates, median of {REPEATS}")
    print("=" * 78, flush=True)

    warm = aligned_uint64_buffer(16, BUFFER_ALIGN)  # JIT-compile outside the timed region
    worker_strided_accumulate(warm, 0, 10, 1)
    worker_private_accumulate(warm, 0, 10, 1)

    sweep = run_stride_benchmark(ctx)
    private = run_private_baseline(ctx)
    gil = run_gil_contrast(ctx)
    m = audit_llvm_and_parity(ctx, sweep, private, gil)

    s8 = sweep[8]["median_ns"]
    print(f"{'layout':<22}{'lines':<16}{'median ns':>16}{'ns/update':>11}{'vs s=8':>9}{'vol/invol cs':>14}")
    for name, lay in [(f"stride={s}", sweep[s]) for s in STRIDES] + [("private register", private)]:
        r = lay["runs"][-1]
        print(f"{name:<22}{str(lay['cache_lines']):<16}{lay['median_ns']:>16,}{lay['ns_per_update']:>11.3f}"
              f"{lay['median_ns'] / s8:>9.2f}{str(r['voluntary_ctx_switches']) + '/' + str(r['involuntary_ctx_switches']):>14}")
    print(f"\nslowdown stride=1 vs stride=8 : {m['slowdown_stride1_vs_stride8']:.2f}x")
    print(f"stride=16 / stride=8          : {m['ratio_stride16_vs_stride8']:.3f}")
    print(f"private speedup vs stride=8   : {m['speedup_private_vs_stride8']:.2f}x")
    g1, g4 = gil["1_threads"], gil[f"{NUM_THREADS}_threads"]
    print(f"pure-Python (GIL) {GIL_DEMO_TOTAL_UPDATES:,} updates: 1 thread {g1['elapsed_ns']:,} ns, "
          f"{NUM_THREADS} threads {g4['elapsed_ns']:,} ns ({gil['slowdown_4_vs_1']:.2f}x)")
    print(f"LLVM audit      : {m['llvm_audit']['store_i64']} store i64, {m['llvm_audit']['load_i64']} load i64, "
          f"{m['llvm_audit']['python_api_calls']} Python API calls -> {'PASS' if m['llvm_audit_passed'] else 'FAIL'}")
    for ln in m["llvm_audit"]["loop_store_lines"]:
        print(f"    {ln}")
    print(f"Parity          : {'PASS' if m['parity_passed'] else 'FAIL'} (expected {m['expected_final_counters']})")
    print(f"verification_passed: {str(m['verification_passed']).lower()} -> {MANIFEST_PATH}")
    return 0 if m["verification_passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
