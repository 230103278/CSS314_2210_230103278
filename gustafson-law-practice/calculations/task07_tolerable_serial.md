# Task 07 — Maximum Allowed Serial Fraction

**Given:** P = 8192, need S(P) ≥ 7500.

## Derivation

```text
P − α(P − 1) ≥ 7500
α(P − 1) ≤ P − 7500
α ≤ (P − 7500) / (P − 1)
```

## Calculation

```text
α_max = (8192 − 7500) / (8192 − 1) = 692 / 8191 ≈ 0.08448
```

**Answer:** α_max ≈ 0.0845 (8.45 %).

## Check

```text
S = 8192 − (692/8191)·8191 = 8192 − 692 = 7500 ✔
```

## Meaning

- The required efficiency is 7500 / 8192 ≈ 91.6 %.
- If a run takes 60 s, at most about 5 s of it can be serial.
