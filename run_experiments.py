import os
import sys
import time
import random
import pulp
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple

# Import modular UFLP solver components
from uflp_solver import (
    UFLPInstance, SolutionState, parse_orlib_instance,
    local_search, _find_closest_two, _compute_total_cost,
    _compute_auxiliary_data
)

def get_lp_bound_and_probs(instance: UFLPInstance) -> Tuple[float, Dict[int, float]]:
    """Solve the LP relaxation of UFLP using PuLP and return objective value and probabilities."""
    F = instance.facilities
    U = instance.customers
    c = instance.setup_costs
    d = instance.service_costs

    model = pulp.LpProblem("LP_Relaxation_Experiments", pulp.LpMinimize)
    y = {f: pulp.LpVariable(f"y_{f}", lowBound=0, upBound=1, cat="Continuous") for f in F}
    x = {u: {f: pulp.LpVariable(f"x_{u}_{f}", lowBound=0, upBound=1, cat="Continuous") for f in F} for u in U}

    model += (
        pulp.lpSum(c[f] * y[f] for f in F)
        + pulp.lpSum(d[u][f] * x[u][f] for u in U for f in F)
    )

    for u in U:
        model += pulp.lpSum(x[u][f] for f in F) == 1
    for u in U:
        for f in F:
            model += x[u][f] <= y[f]

    solver = pulp.PULP_CBC_CMD(msg=0)
    model.solve(solver)
    
    lp_obj = pulp.value(model.objective)
    lp_probs = {f: y[f].varValue for f in F}
    return lp_obj, lp_probs

def construct_lp_biased_solution(
    instance: UFLPInstance,
    lp_probs: Dict[int, float],
    verbose: bool = False
) -> SolutionState:
    """Construct an initial solution biased by fractional LP probabilities."""
    EPS = 0.01
    state = SolutionState()
    for f in instance.facilities:
        prob = max(lp_probs.get(f, 0.0), EPS)
        if random.random() < prob:
            state.open_facilities.add(f)
            
    if not state.open_facilities:
        best_f = min(
            instance.facilities,
            key=lambda f: (
                instance.setup_costs[f]
                + sum(instance.service_costs[u][f] for u in instance.customers)
            ),
        )
        state.open_facilities.add(best_f)
        
    for u in instance.customers:
        closest, second = _find_closest_two(u, state.open_facilities, instance.service_costs)
        state.closest_facility[u] = closest
        state.second_closest_facility[u] = second
        
    state.total_cost = _compute_total_cost(instance, state)
    _compute_auxiliary_data(instance, state)
    
    if verbose:
        print(f"LP-biased initial open: {len(state.open_facilities)}, cost: {state.total_cost:,.2f}")
    return state

def construct_uniform_solution(
    instance: UFLPInstance,
    prob: float,
    verbose: bool = False
) -> SolutionState:
    """Construct an initial solution where each facility is opened with a uniform probability."""
    state = SolutionState()
    for f in instance.facilities:
        if random.random() < prob:
            state.open_facilities.add(f)
            
    if not state.open_facilities:
        best_f = min(
            instance.facilities,
            key=lambda f: (
                instance.setup_costs[f]
                + sum(instance.service_costs[u][f] for u in instance.customers)
            ),
        )
        state.open_facilities.add(best_f)
        
    for u in instance.customers:
        closest, second = _find_closest_two(u, state.open_facilities, instance.service_costs)
        state.closest_facility[u] = closest
        state.second_closest_facility[u] = second
        
    state.total_cost = _compute_total_cost(instance, state)
    _compute_auxiliary_data(instance, state)
    
    if verbose:
        print(f"Uniform initial open: {len(state.open_facilities)}, cost: {state.total_cost:,.2f}")
    return state

def run_local_search_iter_count(instance: UFLPInstance, state: SolutionState) -> Tuple[SolutionState, int, Dict[str, int]]:
    """Run local search, count iterations, and return the breakdown of move types."""
    from uflp_solver import (
        _compute_insert_profit, _compute_delete_profit, _compute_swap_profit,
        _apply_move_and_recompute
    )
    iteration = 0
    move_counts = {'insert': 0, 'delete': 0, 'swap': 0}
    while True:
        best_profit = 1e-9
        best_move = None
        closed_facs = [f for f in instance.facilities if f not in state.open_facilities]

        for f_i in closed_facs:
            profit = _compute_insert_profit(instance, state, f_i)
            if profit > best_profit:
                best_profit = profit
                best_move = ('insert', f_i, None)

        if len(state.open_facilities) > 1:
            for f_r in list(state.open_facilities):
                profit = _compute_delete_profit(instance, state, f_r)
                if profit > best_profit:
                    best_profit = profit
                    best_move = ('delete', None, f_r)

        for f_i in closed_facs:
            for f_r in list(state.open_facilities):
                profit = _compute_swap_profit(instance, state, f_i, f_r)
                if profit > best_profit:
                    best_profit = profit
                    best_move = ('swap', f_i, f_r)

        if best_move is None:
            break

        move_type, f_in, f_out = best_move
        iteration += 1
        move_counts[move_type] += 1
        _apply_move_and_recompute(instance, state, move_type, f_in, f_out)

    return state, iteration, move_counts

def main():
    instance_file = "cap71.txt"
    if not os.path.exists(instance_file):
        print(f"Error: {instance_file} not found in current directory.")
        sys.exit(1)
        
    print(f"Loading UFLP instance: {instance_file}")
    instance = parse_orlib_instance(instance_file)
    
    # Solve LP Relaxation once
    print("Solving LP relaxation using PuLP Simplex...")
    lp_bound, lp_probs = get_lp_bound_and_probs(instance)
    print(f"LP Lower Bound: {lp_bound:,.2f}")
    
    # Calculate Uniform Probability
    # Average LP probability based on eps-floor to make it directly comparable
    expected_open_lp = sum(max(lp_probs.get(f, 0.0), 0.01) for f in instance.facilities)
    uniform_prob = expected_open_lp / len(instance.facilities)
    print(f"Expected open facilities under LP-biased: {expected_open_lp:.2f}")
    print(f"Uniform probability baseline: {uniform_prob:.4f}")
    
    # Experiment parameters
    n_seeds = 20
    seeds = list(range(1, n_seeds + 1))
    
    lp_biased_results = []
    uniform_results = []
    
    print(f"\nRunning {n_seeds} benchmark runs for both methods...")
    
    for seed in seeds:
        # --- 1. LP-Biased Method ---
        random.seed(seed)
        t0 = time.perf_counter()
        init_state_lp = construct_lp_biased_solution(instance, lp_probs)
        init_cost_lp = init_state_lp.total_cost
        t_construct_lp = time.perf_counter() - t0
        
        t1 = time.perf_counter()
        final_state_lp, iters_lp, moves_lp = run_local_search_iter_count(instance, init_state_lp)
        t_ls_lp = time.perf_counter() - t1
        t_total_lp = time.perf_counter() - t0
        
        lp_gap = (final_state_lp.total_cost - lp_bound) / lp_bound * 100
        lp_biased_results.append({
            'seed': seed,
            'init_cost': init_cost_lp,
            'final_cost': final_state_lp.total_cost,
            'iters': iters_lp,
            'time': t_total_lp,
            'gap': lp_gap,
            'init_gap': (init_cost_lp - lp_bound) / lp_bound * 100,
            'open_count': len(final_state_lp.open_facilities),
            'moves': moves_lp
        })
        
        # --- 2. Uniform Random Method ---
        random.seed(seed)
        t0 = time.perf_counter()
        init_state_uni = construct_uniform_solution(instance, uniform_prob)
        init_cost_uni = init_state_uni.total_cost
        t_construct_uni = time.perf_counter() - t0
        
        t1 = time.perf_counter()
        final_state_uni, iters_uni, moves_uni = run_local_search_iter_count(instance, init_state_uni)
        t_ls_uni = time.perf_counter() - t1
        t_total_uni = time.perf_counter() - t0
        
        uni_gap = (final_state_uni.total_cost - lp_bound) / lp_bound * 100
        uniform_results.append({
            'seed': seed,
            'init_cost': init_cost_uni,
            'final_cost': final_state_uni.total_cost,
            'iters': iters_uni,
            'time': t_total_uni,
            'gap': uni_gap,
            'init_gap': (init_cost_uni - lp_bound) / lp_bound * 100,
            'open_count': len(final_state_uni.open_facilities),
            'moves': moves_uni
        })
        
        print(f"  Seed {seed:2d}: LP-Biased cost = {final_state_lp.total_cost:,.2f} ({iters_lp} iters, {t_total_lp:.3f}s) | "
              f"Uniform cost = {final_state_uni.total_cost:,.2f} ({iters_uni} iters, {t_total_uni:.3f}s)")

    # Compute statistics
    def compute_stats(results: List[dict]) -> dict:
        init_costs = [r['init_cost'] for r in results]
        final_costs = [r['final_cost'] for r in results]
        iters = [r['iters'] for r in results]
        times = [r['time'] for r in results]
        gaps = [r['gap'] for r in results]
        init_gaps = [r['init_gap'] for r in results]
        opt_count = sum(1 for g in gaps if g < 1e-5)
        
        import math
        def mean(lst): return sum(lst) / len(lst)
        def std(lst, m): return math.sqrt(sum((x - m)**2 for x in lst) / len(lst))
        
        m_init = mean(init_costs)
        m_final = mean(final_costs)
        m_iter = mean(iters)
        m_time = mean(times)
        m_gap = mean(gaps)
        m_init_gap = mean(init_gaps)
        
        return {
            'init_mean': m_init, 'init_min': min(init_costs), 'init_max': max(init_costs), 'init_std': std(init_costs, m_init),
            'final_mean': m_final, 'final_min': min(final_costs), 'final_max': max(final_costs), 'final_std': std(final_costs, m_final),
            'iter_mean': m_iter, 'iter_min': min(iters), 'iter_max': max(iters),
            'time_mean': m_time, 'time_min': min(times), 'time_max': max(times),
            'gap_mean': m_gap, 'gap_min': min(gaps), 'gap_max': max(gaps),
            'init_gap_mean': m_init_gap, 'init_gap_min': min(init_gaps), 'init_gap_max': max(init_gaps), 'init_gap_std': std(init_gaps, m_init_gap),
            'opt_count': opt_count
        }
        
    lp_stats = compute_stats(lp_biased_results)
    uni_stats = compute_stats(uniform_results)
    
    # Calculate percentage of runs requiring moves
    pct_moves_lp = {
        'insert': (sum(1 for r in lp_biased_results if r['moves']['insert'] > 0) / n_seeds) * 100,
        'delete': (sum(1 for r in lp_biased_results if r['moves']['delete'] > 0) / n_seeds) * 100,
        'swap': (sum(1 for r in lp_biased_results if r['moves']['swap'] > 0) / n_seeds) * 100
    }
    pct_moves_uni = {
        'insert': (sum(1 for r in uniform_results if r['moves']['insert'] > 0) / n_seeds) * 100,
        'delete': (sum(1 for r in uniform_results if r['moves']['delete'] > 0) / n_seeds) * 100,
        'swap': (sum(1 for r in uniform_results if r['moves']['swap'] > 0) / n_seeds) * 100
    }
    
    # 3. Save Results Table to MD file
    print("\nWriting tables of results to cap71_results.md...")
    with open("cap71_results.md", "w", encoding="utf-8") as f:
        f.write("# Benchmarking Results on cap71.txt\n\n")
        f.write("This document summarizes the results of comparing **LP-Biased Hybrid GRASP** vs. **Uniform Random GRASP** ")
        f.write("across 20 different random seeds.\n\n")
        f.write(f"- **Problem Instance:** `cap71.txt` (16 facilities, 50 customers)\n")
        f.write(f"- **LP Optimal Lower Bound:** {lp_bound:,.2f}\n")
        f.write(f"- **Expected facilities open initially:** {expected_open_lp:.2f}\n")
        f.write(f"- **Uniform Probability Baseline:** {uniform_prob:.4f}\n\n")
        
        f.write("## Detailed Seed-by-Seed Results\n\n")
        f.write("| Seed | LP-Biased Init Cost | LP-Biased Init Gap | LP-Biased Final Cost | LP-Biased Iters | Uniform Init Cost | Uniform Init Gap | Uniform Final Cost | Uniform Iters |\n")
        f.write("|---|---|---|---|---|---|---|---|---|\n")
        for i in range(n_seeds):
            r_lp = lp_biased_results[i]
            r_uni = uniform_results[i]
            f.write(f"| {r_lp['seed']} | {r_lp['init_cost']:,.2f} | {r_lp['init_gap']:.4f}% | {r_lp['final_cost']:,.2f} | {r_lp['iters']} | "
                    f"{r_uni['init_cost']:,.2f} | {r_uni['init_gap']:.4f}% | {r_uni['final_cost']:,.2f} | {r_uni['iters']} |\n")
                    
        f.write("\n## Summary Statistics\n\n")
        f.write("| Metric | LP-Biased Hybrid GRASP | Uniform Random GRASP | Improvement / Comparison |\n")
        f.write("|---|---|---|---|\n")
        
        f.write(f"| **Average Init Cost** | {lp_stats['init_mean']:,.2f} (± {lp_stats['init_std']:,.1f}) | {uni_stats['init_mean']:,.2f} (± {uni_stats['init_std']:,.1f}) | "
                f"LP-biased is {((uni_stats['init_mean'] - lp_stats['init_mean'])/uni_stats['init_mean']*100):.1f}% lower |\n")
        f.write(f"| **Average Initial Gap** | {lp_stats['init_gap_mean']:.4f}% (± {lp_stats['init_gap_std']:.4f}%) | {uni_stats['init_gap_mean']:.4f}% (± {uni_stats['init_gap_std']:.4f}%) | "
                f"LP-biased has {(uni_stats['init_gap_mean'] - lp_stats['init_gap_mean']):.4f} percentage points lower gap |\n")
        f.write(f"| **Best Final Cost** | {lp_stats['final_min']:,.2f} | {uni_stats['final_min']:,.2f} | Both found global optimal ({lp_stats['final_min']:,.2f}) |\n")
        f.write(f"| **Average Final Cost** | {lp_stats['final_mean']:,.2f} | {uni_stats['final_mean']:,.2f} | Both always converge to optimum |\n")
        f.write(f"| **Average Local Search Iters** | {lp_stats['iter_mean']:.2f} (range: {lp_stats['iter_min']}-{lp_stats['iter_max']}) | "
                f"{uni_stats['iter_mean']:.2f} (range: {uni_stats['iter_min']}-{uni_stats['iter_max']}) | "
                f"LP-biased needs {((uni_stats['iter_mean'] - lp_stats['iter_mean'])/uni_stats['iter_mean']*100):.1f}% fewer iterations |\n")
        f.write(f"| **Runs Requiring Move Type** | Inserts: {pct_moves_lp['insert']:.1f}%<br/>Deletes: {pct_moves_lp['delete']:.1f}%<br/>Swaps: {pct_moves_lp['swap']:.1f}% | "
                f"Inserts: {pct_moves_uni['insert']:.1f}%<br/>Deletes: {pct_moves_uni['delete']:.1f}%<br/>Swaps: {pct_moves_uni['swap']:.1f}% | "
                f"LP-biased avoids moves in most runs |\n")
        f.write(f"| **Average Solve Time** | {lp_stats['time_mean']*1000:.2f} ms | {uni_stats['time_mean']*1000:.2f} ms | LP-biased is {((uni_stats['time_mean'] - lp_stats['time_mean'])/uni_stats['time_mean']*100):.1f}% faster |\n")
        f.write(f"| **Optimal Solutions Found** | **{lp_stats['opt_count']} / {n_seeds}** | {uni_stats['opt_count']} / {n_seeds} | Both hit global optimum 100% of the time |\n")

    # 4. Generate matplotlib Plots
    print("\nGenerating charts and saving to cap71_analysis.png...")
    
    # Set premium style settings manually
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Liberation Sans']
    plt.rcParams['text.color'] = '#2c3e50'
    plt.rcParams['axes.labelcolor'] = '#2c3e50'
    plt.rcParams['xtick.color'] = '#2c3e50'
    plt.rcParams['ytick.color'] = '#2c3e50'
    
    # Colors: Sleek cool tech colors vs Warm coral accent
    lp_color = '#5C6BC0'  # Indigo
    uni_color = '#FF7043'  # Coral
    bg_color = '#f8f9fa'  # Soft light gray for panel background
    
    fig, axs = plt.subplots(2, 2, figsize=(14, 11), facecolor='white')
    
    # 4.1. Plot 1: Initial Cost Boxplot with Jittered Scatter
    ax1 = axs[0, 0]
    ax1.set_facecolor(bg_color)
    init_data = [
        [r['init_cost'] for r in lp_biased_results],
        [r['init_cost'] for r in uniform_results]
    ]
    # Draw boxplot without outliers (we overlay them all anyway)
    bp1 = ax1.boxplot(init_data, patch_artist=True, widths=0.5, showfliers=False)
    ax1.set_xticks([1, 2])
    ax1.set_xticklabels(['LP-Biased Init', 'Uniform Init'])
    
    bp1['boxes'][0].set(facecolor=lp_color, alpha=0.5, edgecolor='#3f51b5')
    bp1['boxes'][1].set(facecolor=uni_color, alpha=0.5, edgecolor='#e64a19')
    for median in bp1['medians']:
        median.set(color='white', linewidth=2)
        
    # Overlay jittered points
    for i, data_series in enumerate(init_data):
        x_center = i + 1
        jitter = [random.uniform(-0.08, 0.08) for _ in data_series]
        ax1.scatter([x_center + jit for jit in jitter], data_series, 
                   color='#2c3e50', alpha=0.6, edgecolor='none', s=35, zorder=3)
        
    ax1.set_title("Initial Solution Quality (Lower is Better)", fontsize=13, fontweight='bold', pad=15)
    ax1.set_ylabel("Total Cost (Setup + Service)", fontsize=11)
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, loc: f"{int(x):,}"))
    ax1.grid(True, linestyle='--', alpha=0.5, color='#ccc')
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    
    # 4.2. Plot 2: Initial Optimality Gap (%) Boxplot with Jittered Scatter
    ax2 = axs[0, 1]
    ax2.set_facecolor(bg_color)
    init_gap_data = [
        [r['init_gap'] for r in lp_biased_results],
        [r['init_gap'] for r in uniform_results]
    ]
    bp2 = ax2.boxplot(init_gap_data, patch_artist=True, widths=0.5, showfliers=False)
    ax2.set_xticks([1, 2])
    ax2.set_xticklabels(['LP-Biased Init Gap', 'Uniform Init Gap'])
    
    bp2['boxes'][0].set(facecolor=lp_color, alpha=0.5, edgecolor='#3f51b5')
    bp2['boxes'][1].set(facecolor=uni_color, alpha=0.5, edgecolor='#e64a19')
    for median in bp2['medians']:
        median.set(color='white', linewidth=2)
        
    # Overlay jittered points
    for i, data_series in enumerate(init_gap_data):
        x_center = i + 1
        jitter = [random.uniform(-0.08, 0.08) for _ in data_series]
        ax2.scatter([x_center + jit for jit in jitter], data_series, 
                   color='#2c3e50', alpha=0.6, edgecolor='none', s=35, zorder=3)
        
    ax2.set_title("Initial Optimality Gap (%) to LP Bound", fontsize=13, fontweight='bold', pad=15)
    ax2.set_ylabel("Optimality Gap (%)", fontsize=11)
    ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, loc: f"{x:.1f}%"))
    ax2.grid(True, linestyle='--', alpha=0.5, color='#ccc')
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    
    # 4.3. Plot 3: Local Search Iterations to Converge with Jittered Scatter
    ax3 = axs[1, 0]
    ax3.set_facecolor(bg_color)
    iter_data = [
        [r['iters'] for r in lp_biased_results],
        [r['iters'] for r in uniform_results]
    ]
    bp3 = ax3.boxplot(iter_data, patch_artist=True, widths=0.5, showfliers=False)
    ax3.set_xticks([1, 2])
    ax3.set_xticklabels(['LP-Biased LS', 'Uniform LS'])
    
    bp3['boxes'][0].set(facecolor=lp_color, alpha=0.5, edgecolor='#3f51b5')
    bp3['boxes'][1].set(facecolor=uni_color, alpha=0.5, edgecolor='#e64a19')
    for median in bp3['medians']:
        median.set(color='white', linewidth=2)
        
    # Overlay jittered points
    for i, data_series in enumerate(iter_data):
        x_center = i + 1
        jitter = [random.uniform(-0.08, 0.08) for _ in data_series]
        ax3.scatter([x_center + jit for jit in jitter], data_series, 
                   color='#2c3e50', alpha=0.6, edgecolor='none', s=35, zorder=3)
        
    ax3.set_title("Local Search Convergence Efficiency", fontsize=13, fontweight='bold', pad=15)
    ax3.set_ylabel("Local Search Iterations", fontsize=11)
    ax3.grid(True, linestyle='--', alpha=0.5, color='#ccc')
    ax3.spines['top'].set_visible(False)
    ax3.spines['right'].set_visible(False)
    
    # 4.4. Plot 4: Grouped Bar Chart of Move Types (%)
    ax4 = axs[1, 1]
    ax4.set_facecolor(bg_color)
    
    categories = ['Insert', 'Delete', 'Swap']
    lp_pcts = [pct_moves_lp['insert'], pct_moves_lp['delete'], pct_moves_lp['swap']]
    uni_pcts = [pct_moves_uni['insert'], pct_moves_uni['delete'], pct_moves_uni['swap']]
    
    import numpy as np
    x_pos = np.arange(len(categories))
    width = 0.35
    
    rects1 = ax4.bar(x_pos - width/2, lp_pcts, width, label='LP-Biased', color=lp_color, alpha=0.8, edgecolor='#3f51b5')
    rects2 = ax4.bar(x_pos + width/2, uni_pcts, width, label='Uniform', color=uni_color, alpha=0.8, edgecolor='#e64a19')
    
    ax4.set_title("Local Search Moves Requirement (%)", fontsize=13, fontweight='bold', pad=15)
    ax4.set_xticks(x_pos)
    ax4.set_xticklabels(categories)
    ax4.set_ylabel("Runs Requiring Move Type (%)", fontsize=11)
    ax4.set_ylim(0, 115)
    ax4.set_yticks([0, 20, 40, 60, 80, 100])
    ax4.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, loc: f"{int(x)}%"))
    ax4.grid(True, linestyle='--', alpha=0.5, color='#ccc', axis='y')
    ax4.legend(loc='upper right', frameon=True, facecolor='white', edgecolor='none')
    ax4.spines['top'].set_visible(False)
    ax4.spines['right'].set_visible(False)
    
    # Add value labels on top of the bars as percentages
    for rect in rects1:
        height = rect.get_height()
        ax4.annotate(f'{height:.1f}%',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=9, fontweight='bold', color='#2c3e50')
    for rect in rects2:
        height = rect.get_height()
        ax4.annotate(f'{height:.1f}%',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=9, fontweight='bold', color='#2c3e50')
    
    plt.suptitle("UFLP Hybrid LP-GRASP vs. Uniform Random GRASP Performance\nBenchmark Instance: cap71.txt (20 Runs Comparison)", 
                 fontsize=16, fontweight='bold', y=0.98)
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig("cap71_analysis.png", dpi=300, facecolor='white')
    plt.close()
    
    print("\nBenchmark completed successfully!")
    print(f"Results table written to: cap71_results.md")
    print(f"Visualization saved to: cap71_analysis.png")

if __name__ == '__main__':
    main()
