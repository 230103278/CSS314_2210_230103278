# Task 05 — The nCUBE 10 Experiment (1988)

## Background

Gustafson, Montry and Benner (Sandia National Laboratories) ran three applications on an **nCUBE 10**, a hypercube
computer with **1024 processors**. Each processor had its own small memory, and processors exchanged messages.

Applications:

- beam stress analysis
- surface wave simulation
- fluid flow simulation

They reported **scaled speedups of about 1016–1021×** on 1024 processors and won the first Gordon Bell Prize.

## Why it looked impossible

The serial fraction was about 0.4 %. Amdahl's law gives:

```text
S = 1 / (0.004 + 0.996/1024) ≈ 201
```

So people expected about 200×, not 1000×.

## How they did it

1. Each processor got a problem of the same size, so the total problem was 1024 times bigger.
2. The serial part (setup, input, global sums) did not grow with the problem.
3. They measured the serial fraction on the 1024-processor run: α ≈ 0.004.
4. Gustafson's formula gives:

```text
S = 1024 − 0.004 × 1023 ≈ 1020
```

This matches the reported results.

## Why Amdahl's law was not broken

Measured on one processor, the serial fraction of the big problem is tiny (about 0.000004). So Amdahl's limit for
this problem is very high. The experiment did not break Amdahl's law. It changed the problem size.

## Notes

- The one-processor time was estimated, not measured, because the big problem did not fit in one node's memory.
- The codes were carefully tuned to keep communication small.

## Conclusion

The experiment showed that bigger machines are useful when the problem grows with the machine. This idea is
called weak scaling.
