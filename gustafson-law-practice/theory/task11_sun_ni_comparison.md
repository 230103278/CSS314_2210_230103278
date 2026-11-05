# Task 11 — Gustafson vs. Sun–Ni (Memory-Bounded Speedup)

## Sun–Ni formula

Let s be the serial fraction and q = 1 − s. When we use P processors, the parallel work grows by a factor G(P):

```text
S(P) = (s + q·G(P)) / (s + q·G(P)/P)
```

| G(P) | Result |
|---|---|
| G = 1 (work does not grow) | Amdahl's law |
| G = P (work grows with P) | Gustafson's law: S = s + q·P |
| G set by memory | Sun–Ni's law |

So Gustafson's law is a special case of Sun–Ni.

## How G(P) depends on the algorithm

With P times more memory, the problem can be P times bigger in memory. How much the **work** grows depends on the
algorithm:

| Algorithm | Memory | Work | G(P) |
|---|---|---|---|
| Vector operations, simple stencils | N | N | P (same as Gustafson) |
| Matrix multiplication | N² | N³ | P^(3/2) |
| FFT, sorting | N | N·log N | slightly more than P |

## When Gustafson is wrong

A bigger G gives a bigger speedup. So:

- **G(P) > P** (for example matrix multiplication): Gustafson **underestimates** the speedup. But the run time
  also grows, so the time is no longer fixed.
- **G(P) < P** (work grows slower than memory, or a time limit stops the problem from growing): Gustafson
  **overestimates** the speedup.

Other cases:

- Overestimates: communication grows with P, serial I/O grows with problem size, part of memory is used by
  buffers.
- Underestimates: data fits into cache on more processors, which gives extra speedup.

## Conclusion

Gustafson's law fits when time is the limit and work grows like memory. Sun–Ni is more general and fits when
memory is the limit.
