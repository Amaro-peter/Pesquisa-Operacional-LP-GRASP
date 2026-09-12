import os
import sys
import time
import random
import matplotlib.pyplot as plt
import numpy as np
import concurrent.futures
from typing import List, Tuple, Dict

# Import modular components
from scripts.uflp_solver import generate_random_instance, UFLPInstance
from scripts.run_experiments import (
    get_lp_bound_and_probs,
    construct_lp_biased_solution,
    construct_alpha_grasp_solution,
    run_local_search_iter_count
)

# All generated artifacts live in output/, alongside every other report.
OUTPUT_DIR = "output"


def evaluate_seed(seed_instance_tuple):
    # lp_probs MUST travel inside the task tuple. Relying on a module-level
    # global only worked under the "fork" start method; Python 3.14 defaults to
    # "forkserver" on Linux (and "spawn" on macOS/Windows), where the worker
    # re-imports the module and would see an empty dict -- silently degrading
    # the LP-biased arm into a uniform 1% random construction.
    seed, instance, lp_bound, lp_solve_time, lp_probs = seed_instance_tuple
    
    # --- LP-Biased Heuristic ---
    random.seed(seed)
    t0 = time.perf_counter()
    init_state_lp = construct_lp_biased_solution(instance, lp_probs)
    lp_init_gap = (init_state_lp.total_cost - lp_bound) / lp_bound * 100
    
    # local search
    final_state_lp, iters_lp, _ = run_local_search_iter_count(instance, init_state_lp)
    t_search_lp = time.perf_counter() - t0
    
    t_total_lp = lp_solve_time + t_search_lp
    lp_final_gap = (final_state_lp.total_cost - lp_bound) / lp_bound * 100
    
    # --- alpha-GRASP Heuristic ---
    random.seed(seed)
    t0 = time.perf_counter()
    init_state_alpha = construct_alpha_grasp_solution(instance, alpha=0.2)
    alpha_init_gap = (init_state_alpha.total_cost - lp_bound) / lp_bound * 100
    
    # local search
    final_state_alpha, iters_alpha, _ = run_local_search_iter_count(instance, init_state_alpha)
    t_total_alpha = time.perf_counter() - t0
    alpha_final_gap = (final_state_alpha.total_cost - lp_bound) / lp_bound * 100
    
    return {
        'lp_time': t_total_lp,
        'lp_init_gap': lp_init_gap,
        'lp_final_gap': lp_final_gap,
        'alpha_time': t_total_alpha,
        'alpha_init_gap': alpha_init_gap,
        'alpha_final_gap': alpha_final_gap,
        'lp_only_time': lp_solve_time
    }


# Gaps are measured against the LP bound, so a negative gap is impossible: it
# would mean a feasible solution beat a valid lower bound. Values a hair below
# zero are floating-point noise from `(cost - bound) / bound` when the arm
# attains the bound exactly, and printing them as "-0.0000%" reads like a bug.
# Anything larger than the tolerance is NOT hidden -- a genuinely negative gap
# is a real defect and must stay visible.
GAP_NOISE_TOLERANCE = 1e-9


def fmt_gap(value, places=4):
    """Format a percentage gap, absorbing sub-tolerance negative zero."""
    # Compare on magnitude: `-0.0 < 0.0` is False (they compare equal), so a
    # strict range check lets negative zero straight through to "-0.0000".
    if abs(value) < GAP_NOISE_TOLERANCE:
        value = 0.0
    return f"{value:.{places}f}"


def describe_scaling(rows):
    """
    Describe the measured scaling behaviour, computed from `rows`.

    Every figure and every comparative word here is derived from the
    measurements. The previous version of this section was hardcoded prose that
    claimed LP-biased initial gaps were "under 1%" and alpha-GRASP's were "over
    30% to 140%" -- neither of which matched the table printed directly above
    it. That is the same defect that produced `fix_markdown.py`, and a report
    that can contradict its own table is not a report.
    """
    def rng(key):
        vals = [r[key] for r in rows]
        return min(vals), max(vals)

    lp_init_lo, lp_init_hi = rng("lp_init_gap")
    al_init_lo, al_init_hi = rng("alpha_init_gap")
    lp_fin_lo, lp_fin_hi = rng("lp_final_gap")
    al_fin_lo, al_fin_hi = rng("alpha_final_gap")

    biggest = max(rows, key=lambda r: r["vars"])
    lp_t, al_t = biggest["lp_time"], biggest["alpha_time"]
    faster = "faster" if lp_t < al_t else "slower"
    factor = (al_t / lp_t) if lp_t > 0 else float("inf")
    lp_share = (biggest["lp_only_time"] / lp_t * 100) if lp_t > 0 else 0.0

    lp_wins = sum(1 for r in rows
                  if r["lp_final_gap"] < r["alpha_final_gap"] - 1e-9)
    al_wins = sum(1 for r in rows
                  if r["alpha_final_gap"] < r["lp_final_gap"] - 1e-9)
    ties = len(rows) - lp_wins - al_wins

    out = []
    out.append(
        f"1. **Initial Quality Gap**: across the {len(rows)} sizes tested, the LP-biased "
        f"construction starts between **{lp_init_lo:.2f}%** and **{lp_init_hi:.2f}%** above the LP "
        f"bound; alpha-GRASP starts between **{al_init_lo:.2f}%** and **{al_init_hi:.2f}%**."
    )
    out.append(
        f"2. **Convergence and Final Gaps**: after local search the LP-biased arm lands between "
        f"{fmt_gap(lp_fin_lo)}% and {fmt_gap(lp_fin_hi)}%, and alpha-GRASP between "
        f"{fmt_gap(al_fin_lo)}% and {fmt_gap(al_fin_hi)}%. On final gap the LP-biased arm "
        f"was better on {lp_wins} "
        f"{'size' if lp_wins == 1 else 'sizes'}, worse on {al_wins}, and tied on {ties}."
    )
    out.append(
        f"3. **Solve Time Efficiency**: at the largest size ({biggest['size']}, "
        f"{biggest['vars']:,} variables) the LP-biased arm is **{factor:.2f}x {faster}** "
        f"({lp_t:.3f}s against {al_t:.3f}s), with the LP solve itself accounting for "
        f"{lp_share:.0f}% of its time. Gaps are measured against the LP bound, which is a lower "
        f"bound and not a proven optimum, so a 0.0000% entry means the arm attained the bound."
    )
    return out


def main():
    print("=" * 70)
    print("  Large-Scale Scaling Benchmark for UFLP")
    print("  Comparing LP-Biased Hybrid GRASP vs. alpha-GRASP (Parallelized)")
    print("=" * 70)

    sizes = [
        (20, 30),      # 600 variables
        (50, 100),     # 5,000 variables
        (100, 200),    # 20,000 variables
        (150, 400),    # 60,000 variables
        (200, 500),    # 100,000 variables
        (250, 800)     # 200,000 variables
    ]
    
    n_seeds = 5
    seeds = list(range(1, n_seeds + 1))
    
    plot_x = []
    
    lp_times_mean, lp_times_std = [], []
    lp_init_gaps_mean, lp_init_gaps_std = [], []
    lp_final_gaps_mean, lp_final_gaps_std = [], []
    
    alpha_times_mean, alpha_times_std = [], []
    alpha_init_gaps_mean, alpha_init_gaps_std = [], []
    alpha_final_gaps_mean, alpha_final_gaps_std = [], []
    
    detailed_rows = []

    max_workers = os.cpu_count() or 4

    for n_fac, n_cust in sizes:
        n_vars = n_fac * n_cust
        plot_x.append(n_vars)
        print(f"\nEvaluating instance size {n_fac}x{n_cust} ({n_vars:,} variables)...")
        
        instance = generate_random_instance(n_fac, n_cust, seed=999)
        
        t_lp_start = time.perf_counter()
        lp_bound, lp_probs = get_lp_bound_and_probs(instance)
        lp_solve_time = time.perf_counter() - t_lp_start
        print(f"  LP Solved: Bound = {lp_bound:,.2f} in {lp_solve_time:.3f}s")
        
        tasks = [(seed, instance, lp_bound, lp_solve_time, lp_probs) for seed in seeds]
        results = []
        with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
            for res in executor.map(evaluate_seed, tasks):
                results.append(res)
                
        def stats(lst):
            arr = np.array(lst)
            return float(np.mean(arr)), float(np.std(arr))
            
        lp_times = [r['lp_time'] for r in results]
        lp_init_gaps = [r['lp_init_gap'] for r in results]
        lp_final_gaps = [r['lp_final_gap'] for r in results]
        alpha_times = [r['alpha_time'] for r in results]
        alpha_init_gaps = [r['alpha_init_gap'] for r in results]
        alpha_final_gaps = [r['alpha_final_gap'] for r in results]

        lp_time_m, lp_time_s = stats(lp_times)
        lp_init_gap_m, lp_init_gap_s = stats(lp_init_gaps)
        lp_final_gap_m, lp_final_gap_s = stats(lp_final_gaps)
        
        alpha_time_m, alpha_time_s = stats(alpha_times)
        alpha_init_gap_m, alpha_init_gap_s = stats(alpha_init_gaps)
        alpha_final_gap_m, alpha_final_gap_s = stats(alpha_final_gaps)
        
        lp_times_mean.append(lp_time_m)
        lp_times_std.append(lp_time_s)
        lp_init_gaps_mean.append(lp_init_gap_m)
        lp_init_gaps_std.append(lp_init_gap_s)
        lp_final_gaps_mean.append(lp_final_gap_m)
        lp_final_gaps_std.append(lp_final_gap_s)
        
        alpha_times_mean.append(alpha_time_m)
        alpha_times_std.append(alpha_time_s)
        alpha_init_gaps_mean.append(alpha_init_gap_m)
        alpha_init_gaps_std.append(alpha_init_gap_s)
        alpha_final_gaps_mean.append(alpha_final_gap_m)
        alpha_final_gaps_std.append(alpha_final_gap_s)
        
        detailed_rows.append({
            'size': f"{n_fac}x{n_cust}",
            'vars': n_vars,
            'lp_time': lp_time_m,
            'lp_time_std': lp_time_s,
            'lp_init_gap': lp_init_gap_m,
            'lp_init_gap_std': lp_init_gap_s,
            'lp_final_gap': lp_final_gap_m,
            'lp_final_gap_std': lp_final_gap_s,
            'alpha_time': alpha_time_m,
            'alpha_time_std': alpha_time_s,
            'alpha_init_gap': alpha_init_gap_m,
            'alpha_init_gap_std': alpha_init_gap_s,
            'alpha_final_gap': alpha_final_gap_m,
            'alpha_final_gap_std': alpha_final_gap_s,
            'lp_only_time': lp_solve_time,
        })
        
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    md_path = os.path.join(OUTPUT_DIR, "scaling_results.md")
    png_path = os.path.join(OUTPUT_DIR, "scaling_analysis.png")

    print(f"\nWriting tables of results to {md_path}...")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# Large-Scale Scaling Benchmarks (UFLP)\n\n")
        f.write("This document summarizes the performance scaling of **LP-Biased Hybrid GRASP** vs. **alpha-GRASP** ")
        f.write("as instance size increases. Each size is evaluated over 5 independent seeds in parallel.\n\n")
        f.write("- **LP-Biased time** includes the LP Simplex solve time + constructive phase + local search.\n")
        f.write("- **alpha-GRASP time** includes constructive phase + local search only.\n\n")
        
        f.write("## Scaling Metrics Table\n\n")
        f.write("| Size (F x C) | Variables | LP-Biased Solve Time (s) | LP-Biased Init / Final Gap (%) | alpha-GRASP Solve Time (s) | alpha-GRASP Init / Final Gap (%) | LP Solve Portion (s) |\n")
        f.write("|---|---|---|---|---|---|---|\n")
        for r in detailed_rows:
            f.write(f"| {r['size']} | {r['vars']:,} | {r['lp_time']:.3f}s (± {r['lp_time_std']:.3f}) | {fmt_gap(r['lp_init_gap'], 2)}% / {fmt_gap(r['lp_final_gap'])}% | "
                    f"{r['alpha_time']:.3f}s (± {r['alpha_time_std']:.3f}) | {fmt_gap(r['alpha_init_gap'], 2)}% / {fmt_gap(r['alpha_final_gap'])}% | {r['lp_only_time']:.3f}s |\n")
        
        f.write("\n## Discussion of Scaling Behavior\n\n")
        for line in describe_scaling(detailed_rows):
            f.write(line + "\n")

    print("\nGenerating scaling curves and saving to scaling_analysis.png...")
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # 1. Scaling Time plot
    ax = axes[0]
    ax.errorbar(plot_x, lp_times_mean, yerr=lp_times_std, fmt='-o', color='blue', label='LP-Biased Total Time')
    ax.errorbar(plot_x, alpha_times_mean, yerr=alpha_times_std, fmt='-s', color='red', label='alpha-GRASP Total Time')
    ax.set_title('Solve Time Scaling by Instance Size')
    ax.set_xlabel('Instance Variables (Facilities x Customers)')
    ax.set_ylabel('Wall-clock Time (seconds)')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.legend()
    
    # 2. Initial Gap Quality Plot
    ax = axes[1]
    ax.errorbar(plot_x, lp_init_gaps_mean, yerr=lp_init_gaps_std, fmt='-o', color='blue', label='LP-Biased Init Gap')
    ax.errorbar(plot_x, alpha_init_gaps_mean, yerr=alpha_init_gaps_std, fmt='-s', color='red', label='alpha-GRASP Init Gap')
    ax.set_title('Constructive Phase: Initial Gap to LP Lower Bound')
    ax.set_xlabel('Instance Variables (Facilities x Customers)')
    ax.set_ylabel('Initial Gap (%)')
    ax.set_xscale('log')
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(png_path, dpi=150)

    print("\nScaling benchmark completed successfully!")
    print(f"Results table written to: {md_path}")
    print(f"Visualization saved to: {png_path}")

if __name__ == '__main__':
    main()
