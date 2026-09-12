# Large-Scale Scaling Benchmarks (UFLP)

This document summarizes the performance scaling of **LP-Biased Hybrid GRASP** vs. **alpha-GRASP** as instance size increases. Each size is evaluated over 5 independent seeds in parallel.

- **LP-Biased time** includes the LP Simplex solve time + constructive phase + local search.
- **alpha-GRASP time** includes constructive phase + local search only.

## Scaling Metrics Table

| Size (F x C) | Variables | LP-Biased Solve Time (s) | LP-Biased Init / Final Gap (%) | alpha-GRASP Solve Time (s) | alpha-GRASP Init / Final Gap (%) | LP Solve Portion (s) |
|---|---|---|---|---|---|---|
| 20x30 | 600 | 0.006s (± 0.000) | 0.00% / 0.0000% | 0.001s (± 0.000) | 1.86% / 0.0000% | 0.006s |
| 50x100 | 5,000 | 0.030s (± 0.001) | 3.09% / 0.0000% | 0.014s (± 0.002) | 16.41% / 0.0000% | 0.027s |
| 100x200 | 20,000 | 0.125s (± 0.011) | 10.51% / 0.0000% | 0.102s (± 0.016) | 19.56% / 0.1736% | 0.102s |
| 150x400 | 60,000 | 0.408s (± 0.035) | 5.10% / 0.0000% | 0.580s (± 0.072) | 15.52% / 0.0627% | 0.324s |
| 200x500 | 100,000 | 0.804s (± 0.062) | 7.94% / 0.0000% | 1.562s (± 0.170) | 17.26% / 0.0000% | 0.557s |
| 250x800 | 200,000 | 1.908s (± 0.208) | 8.08% / 0.0000% | 4.158s (± 0.155) | 10.86% / 0.0000% | 1.115s |

## Discussion of Scaling Behavior

1. **Initial Quality Gap**: across the 6 sizes tested, the LP-biased construction starts between **0.00%** and **10.51%** above the LP bound; alpha-GRASP starts between **1.86%** and **19.56%**.
2. **Convergence and Final Gaps**: after local search the LP-biased arm lands between 0.0000% and 0.0000%, and alpha-GRASP between 0.0000% and 0.1736%. On final gap the LP-biased arm was better on 2 sizes, worse on 0, and tied on 4.
3. **Solve Time Efficiency**: at the largest size (250x800, 200,000 variables) the LP-biased arm is **2.18x faster** (1.908s against 4.158s), with the LP solve itself accounting for 58% of its time. Gaps are measured against the LP bound, which is a lower bound and not a proven optimum, so a 0.0000% entry means the arm attained the bound.
