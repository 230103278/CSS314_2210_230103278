# Task 04 — Limit of Parallel Efficiency

## Gustafson

```text
E_G(P) = (P − α(P − 1)) / P = 1 − α + α/P
```

As P → ∞, the term α/P → 0, so:

```text
lim E_G(P) = 1 − α
```

Proof: |E_G(P) − (1 − α)| = α/P. For any ε > 0, if P > α/ε then α/P < ε. ∎

- Efficiency only goes down, but never below 1 − α.
- Example: α = 0.05 → efficiency stays above 95 % on any number of processors.

## Amdahl

```text
E_A(P) = 1 / (s·P + 1 − s)
```

As P → ∞ the denominator grows without limit, so:

```text
lim E_A(P) = 0
```

Proof: 0 < E_A(P) ≤ 1/(s·P), and 1/(s·P) → 0. ∎

## Comparison

| | Gustafson | Amdahl |
|---|---|---|
| Speedup limit | ∞ | 1/s |
| Efficiency limit | 1 − α | 0 |
| Example: α = s = 0.05, P = 1024 | E = 0.950 | E = 0.019 |

## Conclusion

Under weak scaling the efficiency stays almost constant. Under strong scaling it always drops to zero.
