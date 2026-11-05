# Task 02 — Derivation of Gustafson's Law

## Setup

The program runs on P processors. Its run time is normalised to 1:

```text
s + p = 1
```

- s — time of the serial part (one processor works)
- p — time of the parallel part (all P processors work)
- α = s — serial fraction measured on the parallel machine

## Step 1: work done on P processors

- Serial part: 1 processor × s = s
- Parallel part: P processors × p = p·P

```text
Total work = s + p·P
```

## Step 2: time on one processor

One processor must do all of this work alone:

```text
T_1 = s + p·P
```

## Step 3: speedup

```text
S(P) = T_1 / T_P = (s + p·P) / (s + p) = s + p·P
```

## Step 4: rewrite with α

Put s = α and p = 1 − α:

```text
S(P) = α + (1 − α)·P
     = α + P − α·P
     = P − α(P − 1)
```

## Checks

- P = 1 → S = 1 ✔
- α = 0 → S = P (ideal) ✔
- α = 1 → S = 1 (all serial) ✔

Efficiency (used later):

```text
E(P) = S(P) / P = 1 − α·(P − 1)/P
```
