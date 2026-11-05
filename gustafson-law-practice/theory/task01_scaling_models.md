# Task 01 — Strong Scaling vs. Weak Scaling

## Comparison

| | Amdahl (1967) | Gustafson (1988) |
|---|---|---|
| Question | How much faster can I solve the same problem? | How much bigger a problem can I solve in the same time? |
| What stays fixed | Problem size | Run time |
| Speedup | S = 1 / (s + (1 − s)/P) | S = P − α(P − 1) |
| Limit as P → ∞ | 1/s | grows without limit |
| Scaling type | Strong scaling | Weak scaling |

## Problem size as P → ∞

- **Amdahl:** the problem size stays the same. Each processor gets a smaller and smaller piece. In the end only the
  serial part is left, so the speedup cannot go above 1/s.
- **Gustafson:** each processor keeps the same amount of work, so the total problem grows with P. The serial part
  stays the same size, but it becomes a smaller and smaller share of the total work. So the speedup keeps growing.

## Why "fixed-size" and "fixed-time"

- Amdahl is **fixed-size**: we keep the problem and add processors to finish faster.
- Gustafson is **fixed-time**: we keep the time budget and add processors to solve a bigger problem (finer mesh,
  more particles). This is how scientists usually use big machines.

## Conclusion

Both laws are correct. They answer different questions. Amdahl is about latency (finish sooner). Gustafson is
about throughput (do more in the same time).
