# Task 08 — Amdahl vs. Gustafson (s = α = 0.08, P = 128)

## Amdahl

```text
S_A = 1 / (0.08 + 0.92/128)
    = 1 / (0.08 + 0.0071875)
    = 1 / 0.0871875
    = 11.47
```

## Gustafson

```text
S_G = 128 − 0.08 × 127
    = 128 − 10.16
    = 117.84
```

## Difference

| | Value |
|---|---|
| S_G / S_A | 10.27 |
| (S_G − S_A) / S_A | +927 % |
| Amdahl efficiency | 9.0 % |
| Gustafson efficiency | 92.1 % |

## Why they diverge

- Amdahl keeps the problem fixed. The serial 8 % stays the same while the parallel part shrinks, so the speedup
  stops near 1/0.08 = 12.5.
- Gustafson grows the problem with P. The serial part stays small compared to the growing parallel work, so the
  speedup keeps growing.
- At P = 1 both give 1. The gap grows as P grows.
