# Cache Line & False Sharing Mechanics — Analysis

**Run:** `python false_sharing_lab.py` (Python 3.12.13, Numba 0.68.1, NumPy 2.2.6) on an **AMD Ryzen 5 8645HS** (Zen 4, 6 cores / 12 threads in a single CCX, 32 KiB L1d and 1 MiB L2 per core, 16 MiB shared L3, 64-byte coherency line read from sysfs).
Git user `230103278`, `INSTANCE_SEED` = 9750467391610345033 (`0x87509e6eb6f6d649`).
Each layout ran 4 `threading.Thread` workers × 60,000,000 updates. The workers were pinned to logical CPUs **0, 2, 4, 6**, one per physical core, so no two workers shared an SMT core's L1. The base buffer is 128-byte aligned. Every timing is the median of 3 runs. All numbers below come from `cache_lab_manifest.json` (`"verification_passed": true`).

| layout | slot byte offsets | 64 B lines touched | 128 B pairs | median wall ns | ns / update / thread | vs stride 8 |
|---|---|---|---|---|---|---|
| A stride = 1 | 0, 8, 16, 24 | 0, 0, 0, 0 | 0, 0, 0, 0 | **708,233,655** | **11.804** | **19.66×** |
| B stride = 4 | 0, 32, 64, 96 | 0, 0, 1, 1 | 0, 0, 0, 0 | **154,485,126** | **2.575** | **4.29×** |
| C stride = 8 | 0, 64, 128, 192 | 0, 1, 2, 3 | 0, 0, 1, 1 | **36,026,883** | **0.600** | 1.00× |
| D stride = 16 | 0, 128, 256, 384 | 0, 2, 4, 6 | 0, 1, 2, 3 | **30,683,898** | **0.511** | 0.85× |
| private register | (one final store) | — | — | **29,002,111** | **0.483** | 0.81× |

All 15 strided runs and the 3 private runs ended with exactly the expected counters `[240000000, 420000000, 180000000, 360000000]`, i.e. `iters × inc` with `inc = seed_t % 7 + 1`.

---

## Q1 — MESI state transitions under stride = 1

`buffer[0]` and `buffer[1]` sit at byte offsets 0 and 8 of the same 64-byte line *L* (manifest `cache_lines: [0, 0, 0, 0]`). Start with *L* in Core 0's L1 in state **M** after its earlier updates, and Invalid in Core 1.

1. **Core 1 executes `store buffer[1]`.** The store retires into Core 1's store buffer, but it can only commit to L1 when Core 1 owns *L* in E or M. Core 1's copy is **I**, so it issues a **Read-For-Ownership** (RFO, a read-with-intent-to-modify). On Zen 4 there is no physical shared bus. The request goes to the L3 / shadow-tag directory of the CCX, which knows Core 0 holds *L* and sends Core 0 a probe.
2. **Core 0 answers the probe.** Core 0 forwards the dirty line cache-to-cache and invalidates its copy: **Core 0 M → I**. Core 1 receives the line with ownership, **I → E**, and immediately commits the store: **E → M**.
3. **Core 0 executes its next `store buffer[0]`.** Its copy is now **I**, so the same sequence runs in reverse: RFO, probe to Core 1, **Core 1 M → I**, **Core 0 I → E → M**.
4. Our loop body is load → add → store. A core that has just lost the line therefore usually misses first on the *load* (a read request). The owner downgrades **M → S**; on AMD's MOESI it becomes **M → O** and keeps the dirty data. The following store then needs a second transaction, an upgrade **S → M** that invalidates the other sharer. With 4 threads the line moves between all four cores in this way, and every move costs at least one cross-core round trip.

**Why the RFO stalls Core 1's store buffer.** x86 is TSO: stores must become globally visible in program order, so the store buffer drains strictly from its head. While the RFO for *L* is outstanding (tens of ns through the L3), the head entry cannot commit, and every younger store queues behind it. Our loop issues one store per iteration, so the 64-entry Zen 4 store buffer fills quickly and dispatch stalls. A second stall comes from the load: when Core 0's probe invalidates *L* while a younger load from *L* has already executed speculatively, the core must flush the pipeline (a memory-ordering machine clear) to keep TSO. Measured cost: **11.804 ns per update** under 4-way contention, about 50 cycles at 4.3 GHz. Uncontended, the same update costs 0.600 ns.

## Q2 — Measured slowdown and coherency overhead

| regime | median ns (wall) | ns / update / thread | aggregate updates / s |
|---|---|---|---|
| stride = 1 | 708,233,655 | 11.804 | 0.339 G |
| stride = 4 | 154,485,126 | 2.575 | 1.553 G |
| stride = 8 | 36,026,883 | 0.600 | 6.662 G |
| stride = 16 | 30,683,898 | 0.511 | 7.822 G |
| private register | 29,002,111 | 0.483 | 8.275 G |

(ns/update = median ÷ 60,000,000, because the 4 threads run concurrently; aggregate = 240,000,000 ÷ median.)

- **Slowdown of stride = 1 relative to stride = 8: 708,233,655 / 36,026,883 = 19.66×.** Against the register baseline it is 24.42×.
- **stride = 4 (2-way sharing): 4.29×** slower than stride 8. Only two cores fight over each line, and a core often gets several updates done before the line is taken away, so the cost is far below the 4-way case.
- **Coherence overhead per update:** 11.804 − 0.600 = **11.20 ns** for 4-way sharing and 2.575 − 0.600 = **1.97 ns** for 2-way sharing.
- **The run-to-run spread is small next to these gaps.** stride 1: 876.1 / 706.7 / 708.2 ms; stride 8: 36.0 / 42.8 / 35.1 ms. The first stride-1 run includes CPU clock ramp-up, which the median filters out.
- **Context switches (`getrusage`).** Voluntary switches stay at 4–10 in every native run, all from the start barrier and the joins, so the native threads never waited on a lock. Involuntary switches scale with runtime: 81–90 for the 0.7 s stride-1 runs against 3–10 for the ~30 ms runs. These are ordinary time-slice preemptions by other processes, not something false sharing causes.

## Q3 — Adjacent-line prefetcher and 128-byte pairing

Measured: stride 8 = **36,026,883 ns** (0.600 ns/update), stride 16 = **30,683,898 ns** (0.511 ns/update), a ratio of **0.852**. That is 15 % faster with 128-byte spacing, and stride 16 won in every run: 30.7 / 34.5 / 29.3 ms against 36.0 / 42.8 / 35.1 ms. Stride 16 is only 5.8 % above the pure-register floor of 29.0 ms.

The buffer is 128-byte aligned, which makes the pairing explicit. With stride 8, threads 0 and 1 use lines 0 and 1, which form **128-byte pair 0**, and threads 2 and 3 share pair 1 (manifest `line_pairs_128B: [0, 0, 1, 1]`). With stride 16 every thread has its own pair (`[0, 1, 2, 3]`).

**How the pairing works.** Intel's L2 "spatial" / adjacent-cache-line prefetcher treats memory as 128-byte-aligned pairs of 64-byte lines. On a miss to one line it also requests the *buddy* line (even ↔ odd) to complete the pair. Zen 4 has the equivalent L2 *up/down* prefetcher and the L1 next-line/stream prefetchers, which fetch line ±1 next to a miss.

**When 64-byte padding still causes secondary invalidations.** Thread 0's line is private, but whenever Core 0 misses on it (first touch, after an eviction, or after an interrupt or context switch) the prefetcher may also request line 1, which Core 1 holds in **M**. That prefetch probes Core 1 and downgrades its copy (M → S/O, or M → I). Core 1's next store then needs an upgrade/RFO even though the two threads share no data. This only happens when:
1. the two hot lines sit in the same 128-byte pair (Intel) or are adjacent (AMD up/down);
2. the prefetcher is enabled;
3. at least one core keeps missing on its line, so the prefetcher keeps firing.

In this benchmark each line mostly stays resident, so condition 3 holds rarely. The penalty is therefore a modest 15 %, not the 19.7× of real false sharing. The difference is consistent with prefetcher interference, but I did not measure probe counts with `perf`, so this attribution is inferred, not proven. 128-byte spacing (stride 16, like C++'s `std::hardware_destructive_interference_size` on Intel) removes the effect.

## Q4 — Register promotion versus real memory stores (LLVM IR)

**What an optimizer wants to do.** In the naive kernel `for _ in range(iters): buffer[slot] += inc`, the address `buffer + 8·slot` is loop-invariant, and no other memory access in the loop can alias it. LLVM's LICM pass therefore applies **scalar promotion**: it loads `buffer[slot]` once before the loop, keeps the running value in a register, and stores it once after the loop. Single-threaded semantics allow this, because no other thread exists as far as the compiler knows (a data race would be undefined behaviour). I compiled exactly that naive kernel to check. Its IR contains `%.217.promoted = load i64 …`, then `%2 = mul i64 %.67, %arg.iters`, and a single `store i64` after the loop. SCEV collapsed the whole loop into one multiply, and 60,000,000 "updates" took **3,546 ns** in total. A benchmark built on that kernel would measure nothing.

**What my kernel emits.** `worker_strided_accumulate` stores to `buffer[slot + (v >> 63)]`, where `v` is the value just loaded. The counter never reaches 2⁶³, so the offset is always 0 at run time, but LLVM cannot prove that. `worker_strided_accumulate.inspect_llvm()` shows the loop body as:

```llvm
%.218 = load i64, ptr %sunkaddr3, align 8          ; v = buffer[slot]          (real L1 load)
%.220 = add i64 %.67, %.218                        ; v + inc
%.221 = lshr i64 %.218, 63                         ; v >> 63   (depends on the loaded value)
%.223 = add nsw i64 %.221, %arg.slot               ; slot + (v >> 63)
%.237 = icmp slt i64 %.223, 0                      ; Python negative-index wraparound check
%.238 = select i1 %.237, i64 %arg.buffer.5.0, i64 0
%6    = getelementptr i64, ptr %arg.buffer.4, i64 %.223
%.252 = getelementptr i64, ptr %6, i64 %.238       ; store address derived from the loaded value
store i64 %.220, ptr %.252, align 8                ; buffer[...] = v + inc     (real L1 store)
```

The store's address is computed from `%.218`, so it is **not loop-invariant**, and LICM cannot promote it. Because that store *may* alias `buffer[slot]`, the load cannot be hoisted either. Every iteration therefore performs a real `load i64` / `store i64` pair. The audit counts **9 `store i64` and 9 `load i64`**: LLVM unrolled the loop 8× (`%.218`, `%.218.1` … `%.218.7`) and added one remainder ("prol") copy, with one load/store pair in each. It also counts **0 `@Py*` API calls** in the native function, so the kernel runs as pure `nopython` machine code with no object-mode fallback; uint64 lowers to LLVM's signless `i64`. Numba has no `volatile`, and this value-dependent address does the same job here.

The privatized control (`acc = acc + inc + (acc >> 63)`) uses the same trick to stop SCEV from folding the loop into a multiply, but keeps `acc` in a register. Comparing the two separates the costs:
- **stride 8 vs register: 0.600 − 0.483 = 0.117 ns per update.** That is the whole cost of going through L1 and store-to-load forwarding when the line is private.
- **stride 1 vs stride 8: 11.20 ns per update.** That is the cross-core coherence cost, about 100× larger.

## Q5 — CPython GIL masking versus `nogil=True`

The same stride-1 update was run as plain Python bytecode (`buffer[slot] += inc` on the same NumPy buffer, same thread pinning), with 8,000,000 updates of total work in both cases:

| | wall ns | ns / update | voluntary / involuntary ctx switches |
|---|---|---|---|
| 1 thread × 8,000,000 | 932,006,454 | 116.5 | 1 / 7 |
| 4 threads × 2,000,000 (stride 1) | 1,034,430,562 | 129.3 | **1014** / 37 |

Four threads are **1.11× slower** than one thread doing the same work, even though they are pinned to four separate physical cores.

**Why false sharing never shows up.** Only the thread holding the GIL can execute bytecode. Every 5 ms (`sys.getswitchinterval() = 0.005`) the running thread is asked to drop the GIL, and another waiting thread is woken through a condition variable. The 1,014 voluntary context switches are those sleeps and wake-ups. At any moment only one core is storing to the line, so it does not bounce between L1s on every update. It migrates about once per GIL hand-off, roughly every 40,000 updates at 116.5 ns each. Each update also costs ~116 ns of interpreter work: bytecode dispatch, boxing an `np.uint64` scalar, and `__getitem__` / `__iadd__` / `__setitem__`. That is about 240× the native 0.483 ns, so even the ~11 ns coherence penalty would disappear in the noise.

**Why more threads run slower.** Adding threads adds work without adding parallelism: GIL request/drop signalling, OS wake-ups and context switches, convoying while threads wait for the lock, and colder caches and branch predictors after each switch. The 4-thread run spends ~102 ms (11 %) more on this overhead.

**Contrast with the native runs.** `@njit(nogil=True)` releases the GIL for the whole kernel, so four cores really execute stores simultaneously. Those threads record only 4–10 voluntary switches and run up to **24× slower** when their counters share a line. That slowdown comes entirely from hardware cache-coherence traffic, which the GIL-serialized version could never expose. The GIL is software lock contention, where threads are serialized and sleep. False sharing is hardware line contention, where all cores run but stall on RFOs.
