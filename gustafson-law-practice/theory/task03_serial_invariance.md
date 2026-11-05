# Task 03 — The Serial Fraction Paradox

## The problem

With a serial fraction of 0.4 %:

- Amdahl: maximum speedup = 1 / 0.004 = 250
- Gustafson on 1024 processors: S = 1024 − 0.004·1023 ≈ 1020

Both use "0.4 %", but the results are very different.

## The answer: the fractions are measured in different places

- Amdahl measures s on **one processor** running the job.
- Gustafson measures α on **the parallel machine** running the bigger job.

If the same job is run on one processor, its time is α + (1 − α)·P. So the serial fraction seen on one processor
is:

```text
s_Amdahl = α / (α + (1 − α)·P)
```

Example with α = 0.004, P = 1024: s_Amdahl ≈ 0.0000039.

## The two laws agree

Put this s_Amdahl into Amdahl's formula:

```text
S_A = α + (1 − α)·P = S_G
```

So for the same job both laws give the same answer. The "paradox" only appears when the α from the parallel
machine is used in Amdahl's formula.

## Why the serial part does not stop scaling

The serial time (setup, input, final sum) stays the same while the parallel work grows with P. So its share of
the total work gets smaller and smaller.

## When this breaks

The serial time must stay constant. It does not when:

- serial input/output grows with the problem size
- communication grows with P (see Task 12)

## Conclusion

A serial fraction only makes sense if you also say where it was measured (machine size and problem size).
