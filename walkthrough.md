# Walkthrough: Hybrid LP-GRASP UFLP Solver (Experiments & Visualizations)

## Overview

We extended the testing framework of the **Hybrid LP-GRASP Solver** by introducing a comprehensive comparative benchmarking suite. This suite evaluates the solver's performance on the standard OR-Library benchmark instance [cap71.txt](file:///e:/Pesquisa_Operacional/cap71.txt) (16 facilities, 50 customers).

To highlight the value of our exact LP relaxation-based guidance (Phase 1), we compare it against a **Uniform Random GRASP** baseline. 

The baseline is calibrated mathematically to open the same expected number of facilities initially ($p \approx 0.69$), focusing the evaluation strictly on *which* facilities are selected rather than *how many*.

---

## Experimental Setup

- **Benchmark Instance:** `cap71.txt` (LP optimal bound = `932,615.75`)
- **Runs:** 20 independent runs per method (seeds 1 to 20)
- **Baseline construction:** Opens each facility with a uniform probability $p = 0.6906$ (calibrated to expected LP facility count of 11.05)
- **LP-biased construction:** Opens each facility $f$ with probability $\max(y_f, 0.01)$ based on Simplex fractional values

---

## Benchmark Results Summary

The experimental data is saved to [cap71_results.md](file:///e:/Pesquisa_Operacional/cap71_results.md). Here is the aggregated summary:

| Metric | LP-Biased Hybrid GRASP | Uniform Random GRASP | Key Takeaway / Comparison |
|---|---|---|---|
| **Average Init Cost** | 933,270.33 (± 1,686.6) | 1,031,760.54 (± 121,165.2) | LP-biased starting solutions are **9.5% cheaper** on average |
| **Average Initial Gap** | **0.0702%** (± 0.1808%) | **10.6308%** (± 12.9920%) | LP-biased starts **151x closer** to the optimal bound |
| **Best Final Cost** | 932,615.75 (Optimal) | 932,615.75 (Optimal) | Both methods find the global optimum under local search |
| **Average Final Cost** | 932,615.75 | 932,615.75 | Both converge to the exact same optimal cost |
| **Average LS Iterations** | **0.15** (range: 0-1) | **4.10** (range: 2-6) | LP-biased converges **96.3% faster** |
| **Runs Requiring Move Type** | Inserts: 0.0%<br/>Deletes: 15.0%<br/>Swaps: 0.0% | Inserts: 35.0%<br/>Deletes: 55.0%<br/>Swaps: 95.0% | LP-biased avoids local search moves in most runs |
| **Average Solve Time** | **0.84 ms** | **3.61 ms** | LP-biased is **4.2x faster** in construction + local search |
| **Optimal Solutions Found** | **20 / 20** (100%) | **20 / 20** (100%) | Both solvers achieve 100% convergence to the optimal solution |

### Key Findings

1. **Near-Perfect Initialization:** By using the Simplex relaxation solutions to guide the constructive heuristic, the LP-biased method starts with a cost that is already within **0.07%** of the global optimum on average. In 17 out of 20 runs, the initial solution constructed was *already* the exact global optimum, requiring **0 local search iterations**.
2. **Scatter Overlay Visualizes the Variance:** Because the LP-biased initial states are so consistent, the box plots for LP-Biased have zero height (collapsing into a line at the optimal cost). By overlaying all 20 runs as individual dots, we can now clearly see that 17 dots are stacked exactly on the optimal line, while only 3 dots represent non-optimal starts, explaining the "zero variance" phenomenon visually.
3. **Move Type Breakdown:** The uniform random method requires significant work during local search, requiring swaps in 95% of runs, deletions in 55% of runs, and insertions in 35% of runs to reach optimality. In contrast, the LP-biased method only requires deletions in 15% of runs and never requires insertions or swaps.
4. **Computational Efficiency:** Because the LP-biased initial solutions are positioned extremely close to the global optimum, the local search requires minimal work. This results in a massive **96.3% reduction in local search iterations** and a **4.2x speedup in solve time** compared to the uniform baseline.

---

## Visualization

A 4-panel analysis plot comparing both methods is saved as [cap71_analysis.png](file:///e:/Pesquisa_Operacional/cap71_analysis.png).

![Performance Comparison Charts](cap71_analysis.png)

* **Top-Left (Initial Cost):** Shows that LP-biased initialization generates vastly superior starting configurations with low variance, with individual runs represented by jittered dots.
* **Top-Right (Initial Optimality Gap):** Highlights the starting optimality gap (LP-biased at 0.07% vs. Uniform at 10.63%).
* **Bottom-Left (LS Iterations):** Displays the local search iterations, showing that LP-biased starts directly at the optimal solution (0 iterations) in most runs.
* **Bottom-Right (LS Moves Requirement):** Illustrates the percentage of runs requiring insert, delete, and swap moves.

---

## Files Created

1. [requirements.txt](file:///e:/Pesquisa_Operacional/requirements.txt) - Pinned dependencies (`pulp==3.3.2`, `matplotlib>=3.5.0`).
2. [run_experiments.py](file:///e:/Pesquisa_Operacional/run_experiments.py) - Standalone script to run the 20-seed comparison, compute statistics, generate results, and output plots.
3. [cap71_results.md](file:///e:/Pesquisa_Operacional/cap71_results.md) - Detailed Markdown results tables.
4. [cap71_analysis.png](file:///e:/Pesquisa_Operacional/cap71_analysis.png) - Performance visualization plots.
