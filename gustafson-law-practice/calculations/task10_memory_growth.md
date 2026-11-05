# Task 10 — Memory Limit for a 3D Grid

## Formula

- Each node has 32 GB.
- A 3D grid of N × N × N cells with b bytes per cell needs b·N³ bytes.
- On P nodes the total memory is P · 32 GB.

```text
b·N³ ≤ P · 32 GB

N_max = (P · 32 GB / b)^(1/3)
```

So N grows only as the cube root of P.

**Assumptions:** 1 GB = 2³⁰ bytes, b = 8 bytes (one `double` per cell), all memory is used by the grid.

## Calculation

64 nodes:

```text
N³ ≤ 64 · 32 · 2³⁰ / 8 = 2³⁸
N  ≤ 2^(38/3) ≈ 6501.9   →   N_max = 6501
```

512 nodes:

```text
N³ ≤ 512 · 32 · 2³⁰ / 8 = 2⁴¹
N  ≤ 2^(41/3) ≈ 13003.8  →   N_max = 13003
```

## Results

| Bytes per cell | 64 nodes | 512 nodes |
|---:|---:|---:|
| 8 (1 field) | 6 501 | 13 003 |
| 40 (5 fields) | 3 802 | 7 604 |

## Conclusion

- 8 times more nodes gives only 2 times larger N.
- Memory limits how far a problem can grow under weak scaling.
