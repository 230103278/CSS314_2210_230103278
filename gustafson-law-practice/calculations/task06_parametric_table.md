# Task 06 — Speedup and Efficiency for α = 0.04

Formulas:

```text
S(P) = P − 0.04·(P − 1)
E(P) = S(P) / P
```

| P | α(P − 1) | S(P) | E(P) |
|---:|---:|---:|---:|
| 16 | 0.60 | 15.40 | 0.962500 |
| 64 | 2.52 | 61.48 | 0.960625 |
| 256 | 10.20 | 245.80 | 0.960156 |
| 1024 | 40.92 | 983.08 | 0.960039 |
| 4096 | 163.80 | 3932.20 | 0.960010 |
| 16384 | 655.32 | 15728.68 | 0.960002 |

Computed with Python:

```text
a = 0.04
for P in [16, 64, 256, 1024, 4096, 16384]:
    S = P - a*(P - 1)
    print(P, round(S, 2), round(S/P, 6))
```

## Evaluation

- P grows 1024 times (16 → 16384), but efficiency drops only from 0.9625 to 0.9600.
- Efficiency gets closer and closer to 1 − α = 0.96 and never goes below it.
- Speedup keeps growing almost linearly.
- For comparison, Amdahl's law with s = 0.04 can never go above 25×.
