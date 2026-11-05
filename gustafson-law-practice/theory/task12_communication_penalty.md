# Task 12 — Gustafson's Law with Communication Cost

## Model

Classic Gustafson assumes the parallel time stays 1. In a real cluster, communication adds extra time C(P). On one
processor there is no communication, so C(1) = 0.

```text
Time on P processors:  T_P = α + (1 − α) + C(P) = 1 + C(P)
Time on 1 processor:   T_1 = α + (1 − α)·P
```

Modified speedup and efficiency:

```text
S(P) = (P − α(P − 1)) / (1 + C(P))
E(P) = S(P) / P
```

If C(P) = 0, this is the normal Gustafson law.

## Two types of communication cost

| Type | C(P) | Example |
|---|---|---|
| Tree-based | c · log₂ P | allreduce, barrier, broadcast |
| Square-root | c · (√P − 1) | data transfer limited by network bandwidth |

What happens as P grows:

- With log₂ P cost, speedup grows like P / log P. Efficiency falls slowly.
- With √P cost, speedup grows only like √P. Efficiency falls faster.
- In both cases efficiency goes to 0, not to 1 − α.

## Example (α = 0.04)

Log cost with c = 0.01, square-root cost with c = 0.001:

| P | Classic S | S with log cost | S with √P cost |
|---:|---:|---:|---:|
| 16 | 15.40 | 14.8 | 15.4 |
| 1024 | 983.08 | 893.7 | 953.5 |
| 16384 | 15728.68 | 13797.1 | 13956.2 |
| 65536 | 62914.60 | 54236.7 | 50131.2 |

## How to reduce the cost

- Overlap communication with computation.
- Exchange data only with neighbours.
- Use fewer global synchronisations.

## Conclusion

Gustafson's law is accurate only when communication does not grow with P.
