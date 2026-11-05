# Task 13 — When Gustafson's Law Fails

Gustafson's law assumes that if the problem grows with P, the work per processor and the run time stay the same.
Below are two algorithms where this is not true.

## Case 1 — Conjugate Gradient (CG) solver

CG is used to solve large systems of equations, for example the Poisson equation on a 3D grid.

Why the time grows:

- **More iterations for bigger problems.** For a 3D grid with N_side points per side, CG needs about N_side
  iterations. Under weak scaling N_side grows as P^(1/3). So a bigger problem needs more iterations, even though
  each processor has the same number of grid points.
- **Global synchronisation.** Every iteration computes dot products, which need a global sum over all processors.
  This costs about log₂ P time and every processor must wait for it.

```text
T(P) ≈ P^(1/3) · (local work + c · log₂ P)
```

Example: from 1 to 4096 processors the number of iterations grows 16 times, so the "fixed-time" run takes about 16
times longer.

**Fix:** use multigrid, which needs about the same number of iterations for any problem size.

## Case 2 — Parallel 3D FFT

FFT is used in many simulations (turbulence, molecular dynamics).

Why the time grows:

- **Work is N·log N, not N.** When N grows with P, the work per processor also grows a little (by the log N factor).
- **All-to-all communication.** The FFT needs to swap data between all processors. Each processor sends messages to
  P − 1 others, so the number of messages grows with P.
- **Network limits.** All processors send at the same time, so the network becomes congested.

```text
T(P) ≈ (N/P)·log N + t_s·(P − 1) + (network transfer time)
```

The t_s·(P − 1) term grows without limit, so the time cannot stay constant.

## Summary

| Algorithm | Why Gustafson fails | How time grows |
|---|---|---|
| CG | More iterations + global sums | P^(1/3) · log P |
| 3D FFT | N·log N work + all-to-all | grows with P |

**Conclusion:** Gustafson's law works well only when the work grows linearly with problem size and each processor
communicates only with a few neighbours.
