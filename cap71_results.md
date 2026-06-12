# Benchmarking Results on cap71.txt

This document summarizes the results of comparing **LP-Biased Hybrid GRASP** vs. **Uniform Random GRASP** across 20 different random seeds.

- **Problem Instance:** `cap71.txt` (16 facilities, 50 customers)
- **LP Optimal Lower Bound:** 932,615.75
- **Expected facilities open initially:** 11.05
- **Uniform Probability Baseline:** 0.6906

## Detailed Seed-by-Seed Results

| Seed | LP-Biased Init Cost | LP-Biased Init Gap | LP-Biased Final Cost | LP-Biased Iters | Uniform Init Cost | Uniform Init Gap | Uniform Final Cost | Uniform Iters |
|---|---|---|---|---|---|---|---|---|
| 1 | 934,622.57 | 0.2152% | 932,615.75 | 1 | 1,111,810.45 | 19.2142% | 932,615.75 | 5 |
| 2 | 932,615.75 | 0.0000% | 932,615.75 | 0 | 959,620.61 | 2.8956% | 932,615.75 | 3 |
| 3 | 932,615.75 | 0.0000% | 932,615.75 | 0 | 1,081,120.29 | 15.9234% | 932,615.75 | 5 |
| 4 | 932,615.75 | 0.0000% | 932,615.75 | 0 | 954,661.31 | 2.3638% | 932,615.75 | 4 |
| 5 | 932,615.75 | 0.0000% | 932,615.75 | 0 | 1,502,238.28 | 61.0779% | 932,615.75 | 6 |
| 6 | 938,158.11 | 0.5943% | 932,615.75 | 1 | 1,089,204.84 | 16.7903% | 932,615.75 | 5 |
| 7 | 932,615.75 | 0.0000% | 932,615.75 | 0 | 945,928.46 | 1.4275% | 932,615.75 | 4 |
| 8 | 932,615.75 | 0.0000% | 932,615.75 | 0 | 959,445.55 | 2.8768% | 932,615.75 | 4 |
| 9 | 938,158.11 | 0.5943% | 932,615.75 | 1 | 1,051,555.14 | 12.7533% | 932,615.75 | 4 |
| 10 | 932,615.75 | 0.0000% | 932,615.75 | 0 | 1,051,485.34 | 12.7458% | 932,615.75 | 3 |
| 11 | 932,615.75 | 0.0000% | 932,615.75 | 0 | 959,926.35 | 2.9284% | 932,615.75 | 2 |
| 12 | 932,615.75 | 0.0000% | 932,615.75 | 0 | 953,663.34 | 2.2568% | 932,615.75 | 4 |
| 13 | 932,615.75 | 0.0000% | 932,615.75 | 0 | 959,869.94 | 2.9223% | 932,615.75 | 4 |
| 14 | 932,615.75 | 0.0000% | 932,615.75 | 0 | 963,593.51 | 3.3216% | 932,615.75 | 4 |
| 15 | 932,615.75 | 0.0000% | 932,615.75 | 0 | 1,066,186.41 | 14.3222% | 932,615.75 | 5 |
| 16 | 932,615.75 | 0.0000% | 932,615.75 | 0 | 1,042,445.81 | 11.7766% | 932,615.75 | 3 |
| 17 | 932,615.75 | 0.0000% | 932,615.75 | 0 | 1,067,499.12 | 14.4629% | 932,615.75 | 4 |
| 18 | 932,615.75 | 0.0000% | 932,615.75 | 0 | 951,858.36 | 2.0633% | 932,615.75 | 2 |
| 19 | 932,615.75 | 0.0000% | 932,615.75 | 0 | 965,378.96 | 3.5130% | 932,615.75 | 5 |
| 20 | 932,615.75 | 0.0000% | 932,615.75 | 0 | 997,718.71 | 6.9807% | 932,615.75 | 6 |

## Summary Statistics

| Metric | LP-Biased Hybrid GRASP | Uniform Random GRASP | Improvement / Comparison |
|---|---|---|---|
| **Average Init Cost** | 933,270.33 (± 1,686.6) | 1,031,760.54 (± 121,165.2) | LP-biased is 9.5% lower |
| **Average Initial Gap** | 0.0702% (± 0.1808%) | 10.6308% (± 12.9920%) | LP-biased has 10.5606 percentage points lower gap |
| **Best Final Cost** | 932,615.75 | 932,615.75 | Both found global optimal (932,615.75) |
| **Average Final Cost** | 932,615.75 | 932,615.75 | Both always converge to optimum |
| **Average Local Search Iters** | 0.15 (range: 0-1) | 4.10 (range: 2-6) | LP-biased needs 96.3% fewer iterations |
| **Runs Requiring Move Type** | Inserts: 0.0%<br/>Deletes: 15.0%<br/>Swaps: 0.0% | Inserts: 35.0%<br/>Deletes: 55.0%<br/>Swaps: 95.0% | LP-biased avoids moves in most runs |
| **Average Solve Time** | 0.84 ms | 3.61 ms | LP-biased is 76.6% faster |
| **Optimal Solutions Found** | **20 / 20** | 20 / 20 | Both hit global optimum 100% of the time |
