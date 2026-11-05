# Task 09 — Workload with Three Phases

## Phases

| Phase | Share | Type |
|---|---:|---|
| A — Disk I/O and parameter broadcast | 2 % | Serial |
| B — Boundary synchronisation | 6 % | Serial (processors wait) |
| C — Flux computation | 92 % | Parallel |

```text
α = 0.02 + 0.06 = 0.08
```

## Scaled speedup

```text
S(512)  = 512  − 0.08 × 511  = 512  − 40.88  = 471.12
S(2048) = 2048 − 0.08 × 2047 = 2048 − 163.76 = 1884.24
```

| P | S(P) | E(P) |
|---:|---:|---:|
| 512 | 471.12 | 0.920 |
| 2048 | 1884.24 | 0.920 |

## Analysis

- Phase B causes most of the loss (6 % out of 8 %).
- If B is hidden by overlapping communication with computation, α = 0.02 and S(2048) = 2007.06 (+6.5 %).
- This assumes B does not grow with P. In real systems synchronisation often grows with P, which lowers the
  speedup.
