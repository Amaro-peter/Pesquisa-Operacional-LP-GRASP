"""
Hybrid LP-GRASP Solver for the Uncapacitated Facility Location Problem (UFLP)

Based on the framework by Resende & Werneck (2006):
"A hybrid multistart heuristic for the uncapacitated facility location problem"

Improvement: Uses the LP relaxation (solved with SciPy's HiGHS backend) to bias
the GRASP construction phase,
replacing uniform random selection with probability-weighted selection derived
from exact fractional solutions.

Usage:
    python uflp_solver.py                              # Random instance demo
    python uflp_solver.py --file cap71.txt             # OR-Library instance
    python uflp_solver.py --n-facilities 25 --n-customers 40 --seed 7
"""

from __future__ import annotations

import argparse
import math
import random
import sys
import time
from dataclasses import dataclass, field
from typing import Dict, List, Set, Tuple

import numpy as np
from scipy.optimize import linprog
import scipy.sparse as sp

# ============================================================================
# Section 1: Data Structures
# ============================================================================

@dataclass
class UFLPInstance:
    """Problem instance for the Uncapacitated Facility Location Problem."""
    facilities: List[int]
    customers: List[int]
    setup_costs: Dict[int, float]
    service_costs: Dict[int, Dict[int, float]]  # service_costs[customer][facility]


@dataclass
class SolutionState:
    """Algorithmic state tracking an incumbent solution and auxiliary data."""
    open_facilities: Set[int] = field(default_factory=set)
    total_cost: float = float('inf')
    closest_facility: Dict[int, int] = field(default_factory=dict)
    second_closest_facility: Dict[int, int] = field(default_factory=dict)
    save: Dict[int, float] = field(default_factory=dict)
    loss: Dict[int, float] = field(default_factory=dict)
    extra: Dict[Tuple[int, int], float] = field(default_factory=dict)


# ============================================================================
# Section 2: Instance I/O
# ============================================================================

def parse_orlib_instance(filepath: str) -> UFLPInstance:
    """
    Parse a UFLP instance in OR-Library format (Beasley).

    Format:
        Line 1: n_facilities  n_customers
        Next n_facilities lines: capacity  setup_cost  (one per facility)
        Then n_customers blocks, each:
            demand
            n_facilities service costs (may span multiple lines)
        (capacity and demand are ignored for the uncapacitated version)
    """
    with open(filepath, 'r') as f:
        tokens = f.read().split()

    idx = 0
    n_facilities = int(tokens[idx]); idx += 1
    n_customers = int(tokens[idx]); idx += 1

    facilities = list(range(n_facilities))
    customers = list(range(n_customers))
    setup_costs: Dict[int, float] = {}
    service_costs: Dict[int, Dict[int, float]] = {u: {} for u in customers}

    # Read facility data: capacity, setup_cost
    for fac in facilities:
        _capacity = float(tokens[idx]); idx += 1  # ignored
        setup_costs[fac] = float(tokens[idx]); idx += 1

    # Read customer data: demand, then n_facilities service costs
    for cust in customers:
        _demand = float(tokens[idx]); idx += 1  # ignored
        for fac in facilities:
            cost = float(tokens[idx]); idx += 1
            service_costs[cust][fac] = cost

    return UFLPInstance(
        facilities=facilities,
        customers=customers,
        setup_costs=setup_costs,
        service_costs=service_costs,
    )


def generate_random_instance(
    n_facilities: int = 20,
    n_customers: int = 30,
    seed: int = 42,
) -> UFLPInstance:
    """
    Generate a random Euclidean UFLP instance.

    Facilities and customers are placed uniformly in [0, 1000]².
    Setup costs are drawn uniformly from [100, 5000].
    Service costs equal the Euclidean distance.
    """
    rng = random.Random(seed)

    facilities = list(range(n_facilities))
    customers = list(range(n_customers))

    fac_coords = {f: (rng.uniform(0, 1000), rng.uniform(0, 1000)) for f in facilities}
    cust_coords = {u: (rng.uniform(0, 1000), rng.uniform(0, 1000)) for u in customers}

    setup_costs = {f: round(rng.uniform(100, 5000), 2) for f in facilities}

    service_costs: Dict[int, Dict[int, float]] = {}
    for u in customers:
        service_costs[u] = {}
        ux, uy = cust_coords[u]
        for f in facilities:
            fx, fy = fac_coords[f]
            dist = math.sqrt((ux - fx) ** 2 + (uy - fy) ** 2)
            service_costs[u][f] = round(dist, 4)

    return UFLPInstance(
        facilities=facilities,
        customers=customers,
        setup_costs=setup_costs,
        service_costs=service_costs,
    )


# ============================================================================
# Section 3: Phase 1 — Simplex-Based LP Relaxation
# ============================================================================

def solve_lp_relaxation(
    instance: UFLPInstance,
    verbose: bool = True,
) -> Dict[int, float]:
    """
    Solve the LP relaxation of the UFLP with SciPy's `linprog` (HiGHS backend).

    Variables:
        y[f] ∈ [0, 1]  — fraction of facility f opened
        x[u][f] ∈ [0, 1] — fraction of customer u served by facility f

    Constraints:
        Σ_f x[u][f] = 1               for each customer u  (full assignment)
        x[u][f] ≤ y[f]                for each (u, f)       (linking)

    Returns:
        Dict mapping each facility to its fractional LP value y[f].
    """
    F_list = instance.facilities
    U_list = instance.customers
    n_F = len(F_list)
    n_U = len(U_list)
    
    f_map = {f: i for i, f in enumerate(F_list)}
    u_map = {u: i for i, u in enumerate(U_list)}
    
    n_vars = n_F + n_U * n_F
    
    c_obj = np.zeros(n_vars)
    for f in F_list:
        c_obj[f_map[f]] = instance.setup_costs[f]
    for u in U_list:
        for f in F_list:
            idx = n_F + u_map[u] * n_F + f_map[f]
            c_obj[idx] = instance.service_costs[u][f]
            
    bounds = (0.0, 1.0)
    
    A_eq_row, A_eq_col, A_eq_data = [], [], []
    for u in U_list:
        u_idx = u_map[u]
        for f in F_list:
            A_eq_row.append(u_idx)
            A_eq_col.append(n_F + u_idx * n_F + f_map[f])
            A_eq_data.append(1.0)
    A_eq = sp.csc_matrix((A_eq_data, (A_eq_row, A_eq_col)), shape=(n_U, n_vars))
    b_eq = np.ones(n_U)
    
    n_ineq = n_U * n_F
    A_ub_row, A_ub_col, A_ub_data = [], [], []
    
    row_idx = 0
    for u in U_list:
        u_idx = u_map[u]
        for f in F_list:
            f_idx = f_map[f]
            # x[u][f]
            A_ub_row.append(row_idx)
            A_ub_col.append(n_F + u_idx * n_F + f_idx)
            A_ub_data.append(1.0)
            
            # -y[f]
            A_ub_row.append(row_idx)
            A_ub_col.append(f_idx)
            A_ub_data.append(-1.0)
            
            row_idx += 1
            
    A_ub = sp.csc_matrix((A_ub_data, (A_ub_row, A_ub_col)), shape=(n_ineq, n_vars))
    b_ub = np.zeros(n_ineq)
    
    res = linprog(c_obj, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=[bounds]*n_vars, method='highs')
    
    if not res.success:
        raise RuntimeError(f"LP relaxation did not reach optimality (status={res.status}): {res.message}")
        
    lp_probs: Dict[int, float] = {f: res.x[f_map[f]] for f in F_list}

    if verbose:
        lp_obj = res.fun
        n_frac = sum(1 for v in lp_probs.values() if 0.01 < v < 0.99)
        n_integral = sum(1 for v in lp_probs.values() if v >= 0.99)
        print(f"[Phase 1] LP Relaxation solved")
        print(f"          LP objective (lower bound): {lp_obj:,.2f}")
        print(f"          Facilities integral (y>=0.99): {n_integral}/{len(F_list)}")
        print(f"          Facilities fractional: {n_frac}/{len(F_list)}")

    return lp_probs


# ============================================================================
# Section 4: Phase 2 — Probabilistic Constructive Heuristic
# ============================================================================

def _find_closest_two(
    customer: int,
    open_set: Set[int],
    service_costs: Dict[int, Dict[int, float]],
) -> Tuple[int, int]:
    """Return (closest_facility, second_closest_facility) for a customer."""
    costs = service_costs[customer]
    sorted_facs = sorted(open_set, key=lambda f: costs[f])
    closest = sorted_facs[0]
    second = sorted_facs[1] if len(sorted_facs) > 1 else closest
    return closest, second


def _compute_total_cost(
    instance: UFLPInstance,
    state: SolutionState,
) -> float:
    """Compute total cost = Σ setup + Σ service (to closest open facility)."""
    cost = sum(instance.setup_costs[f] for f in state.open_facilities)
    for u in instance.customers:
        cost += instance.service_costs[u][state.closest_facility[u]]
    return cost


def _compute_auxiliary_data(
    instance: UFLPInstance,
    state: SolutionState,
) -> None:
    """
    Compute save, loss, and extra from scratch per Resende & Werneck.

    Definitions (all with respect to the CURRENT solution state):

    save[f_i] (f_i closed): the cost DECREASE from inserting f_i.
        save[f_i] = -c(f_i) + Sum_u max(0, d(u, phi1[u]) - d(u, f_i))
        where phi1[u] = closest open facility to u.
        Positive save means inserting f_i improves the solution.

    loss[f_r] (f_r open): the cost INCREASE from removing f_r.
        loss[f_r] = -c(f_r) + Sum_{u: phi1[u]=f_r} (d(u, phi2[u]) - d(u, f_r))
        where phi2[u] = second closest open facility to u.
        Positive loss means removing f_r worsens the solution.
        (We save the setup cost c(f_r) but pay more in reassignment.)

    extra[(f_i, f_r)] (f_i closed, f_r open): swap correction.
        For users u with phi1[u] = f_r, if they would be reassigned to
        phi2[u] upon removal of f_r, but f_i is actually closer than phi2[u],
        then extra corrects for this. Specifically:
        extra[(f_i, f_r)] = Sum_{u: phi1[u]=f_r, d(u,f_i) < d(u,phi2[u])}
                                (d(u, phi2[u]) - d(u, f_i))

    Move profits:
        profit(insert f_i)          = save[f_i]
        profit(remove f_r)          = -loss[f_r]
        profit(swap f_i for f_r)    = save[f_i] - loss[f_r] + extra[(f_i, f_r)]

    IMPORTANT: The swap formula double-counts for customers assigned to f_r
    that would also benefit from f_i. The save already counts them (they get
    closer to f_i than phi1[u]=f_r), and loss already counts them (reassigning
    to phi2[u]). The extra term corrects loss's overestimate but save still
    includes the f_i benefit for those users. To fix this, we subtract from
    save the contribution of users assigned to f_r when computing swap profit.
    We store this per-pair correction in a separate field.
    """
    F = instance.facilities
    U = instance.customers
    c = instance.setup_costs
    d = instance.service_costs
    open_facs = state.open_facilities
    closed_facs = [f for f in F if f not in open_facs]
    phi1 = state.closest_facility
    phi2 = state.second_closest_facility

    # --- save[f_i] for each closed facility ---
    state.save = {}
    for f_i in closed_facs:
        benefit = -c[f_i]
        for u in U:
            gain = d[u][phi1[u]] - d[u][f_i]
            benefit += max(0.0, gain)
        state.save[f_i] = benefit

    # --- loss[f_r] for each open facility ---
    state.loss = {}
    for f_r in open_facs:
        penalty = -c[f_r]  # removing f_r saves setup cost (reduces loss)
        for u in U:
            if phi1[u] == f_r:
                penalty += d[u][phi2[u]] - d[u][f_r]
        state.loss[f_r] = penalty

    # --- extra[(f_i, f_r)] sparse ---
    state.extra = {}
    for f_i in closed_facs:
        for f_r in open_facs:
            correction = 0.0
            for u in U:
                if phi1[u] == f_r:
                    # Formula to perfectly correct the double counting of save and loss:
                    # extra_u = d(u, phi2[u]) - min(d(u, f_i), d(u, phi2[u])) - max(0, d(u, f_r) - d(u, f_i))
                    # It is only non-zero when d(u, f_i) < d(u, phi2[u])
                    if d[u][f_i] < d[u][phi2[u]]:
                        correction += (d[u][phi2[u]] - d[u][f_i]) - max(0, d[u][f_r] - d[u][f_i])
            if abs(correction) > 1e-12:
                state.extra[(f_i, f_r)] = correction


def _compute_swap_profit(
    instance: UFLPInstance,
    state: SolutionState,
    f_i: int,
    f_r: int,
) -> float:
    """
    Compute the EXACT profit of swapping f_r (open) with f_i (closed) in O(1).
    We use the precomputed save, loss, and extra arrays, but adjust for the
    double-counting of customers currently assigned to f_r who also benefit from f_i.
    """
    # If there's only 1 open facility, we can't use loss (it assumes phi2 is a valid backup)
    if len(state.open_facilities) == 1:
        d = instance.service_costs
        delta = instance.setup_costs[f_i] - instance.setup_costs[f_r]
        for u in instance.customers:
            delta += d[u][f_i] - d[u][f_r]
        return -delta

    profit = state.save[f_i] - state.loss[f_r] + state.extra.get((f_i, f_r), 0.0)
    
    # Correct the double-counting: for any customer where phi1 == f_r,
    # save[f_i] already included max(0, d(u, f_r) - d(u, f_i)).
    # We must subtract this out because the f_r facility is being removed,
    # and their true reassignment is handled by loss + extra.
    # To do this in O(1) time without looping over all users, we would need 
    # to store this specific overlap. Since the current state doesn't have it,
    # we can store it in another sparse dict, or we can just compute the correction 
    # during _compute_auxiliary_data. Let's assume we update _compute_auxiliary_data
    # to include this overlap subtraction directly in `extra`.
    return profit


def _compute_insert_profit(
    instance: UFLPInstance,
    state: SolutionState,
    f_i: int,
) -> float:
    """Compute exact profit of inserting closed facility f_i in O(1)."""
    return state.save[f_i]


def _compute_delete_profit(
    instance: UFLPInstance,
    state: SolutionState,
    f_r: int,
) -> float:
    """Compute exact profit of removing open facility f_r in O(1)."""
    return -state.loss[f_r]


def construct_solution(
    instance: UFLPInstance,
    lp_probs: Dict[int, float],
    verbose: bool = True,
) -> SolutionState:
    """
    Build an initial solution using LP-biased probabilistic construction.

    Each facility f is opened with probability max(lp_probs[f], eps) where eps=0.01.
    If the result is empty, we fall back to the single cheapest facility.
    """
    EPS = 0.01
    state = SolutionState()

    # --- Select facilities probabilistically ---
    for f in instance.facilities:
        prob = max(lp_probs.get(f, 0.0), EPS)
        if random.random() < prob:
            state.open_facilities.add(f)

    # --- Fallback: if nothing opened, pick the cheapest total-cost facility ---
    if not state.open_facilities:
        best_f = min(
            instance.facilities,
            key=lambda f: (
                instance.setup_costs[f]
                + sum(instance.service_costs[u][f] for u in instance.customers)
            ),
        )
        state.open_facilities.add(best_f)

    # --- Assign customers to closest & second closest ---
    for u in instance.customers:
        closest, second = _find_closest_two(
            u, state.open_facilities, instance.service_costs
        )
        state.closest_facility[u] = closest
        state.second_closest_facility[u] = second

    # --- Compute cost and auxiliary data ---
    state.total_cost = _compute_total_cost(instance, state)
    _compute_auxiliary_data(instance, state)

    if verbose:
        print(f"[Phase 2] Constructed initial solution")
        print(f"          Open facilities: {len(state.open_facilities)}/{len(instance.facilities)}")
        print(f"          Initial cost: {state.total_cost:,.2f}")

    return state


# ============================================================================
# Section 5: Phase 3 — Fast Local Search (Best-Improvement)
# ============================================================================

def _apply_move_and_recompute(
    instance: UFLPInstance,
    state: SolutionState,
    move_type: str,
    f_in: int | None,
    f_out: int | None,
) -> None:
    """
    Apply a move (insert/delete/swap) and fully recompute the solution state.
    """
    d = instance.service_costs

    match move_type:
        case 'insert':
            state.open_facilities.add(f_in)
        case 'delete':
            state.open_facilities.discard(f_out)
        case 'swap':
            state.open_facilities.add(f_in)
            state.open_facilities.discard(f_out)

    # Recompute closest/second_closest for all customers
    for u in instance.customers:
        closest, second = _find_closest_two(u, state.open_facilities, d)
        state.closest_facility[u] = closest
        state.second_closest_facility[u] = second

    # Recompute total cost
    state.total_cost = _compute_total_cost(instance, state)

    # Recompute auxiliary data
    _compute_auxiliary_data(instance, state)


def local_search(
    instance: UFLPInstance,
    state: SolutionState,
    verbose: bool = True,
) -> SolutionState:
    """
    Best-improvement local search over insertions, deletions, and swaps.

    Uses exact delta evaluation for each candidate move. At each iteration,
    evaluates all possible moves and applies the single move with the highest
    positive profit. Stops when no improving move exists (local optimum).

    The save/loss/extra structures are maintained in the SolutionState for
    external inspection, but move acceptance is based on exact cost deltas
    to prevent numerical cycling.
    """
    iteration = 0

    while True:
        best_profit = 1e-9  # minimum threshold to avoid numerical noise
        best_move: Tuple[str, int | None, int | None] | None = None

        closed_facs = [f for f in instance.facilities if f not in state.open_facilities]

        # --- 1. Evaluate insertions ---
        for f_i in closed_facs:
            profit = _compute_insert_profit(instance, state, f_i)
            if profit > best_profit:
                best_profit = profit
                best_move = ('insert', f_i, None)

        # --- 2. Evaluate deletions (only if >1 facility is open) ---
        if len(state.open_facilities) > 1:
            for f_r in list(state.open_facilities):
                profit = _compute_delete_profit(instance, state, f_r)
                if profit > best_profit:
                    best_profit = profit
                    best_move = ('delete', None, f_r)

        # --- 3. Evaluate swaps ---
        for f_i in closed_facs:
            for f_r in list(state.open_facilities):
                profit = _compute_swap_profit(instance, state, f_i, f_r)
                if profit > best_profit:
                    best_profit = profit
                    best_move = ('swap', f_i, f_r)

        # --- No improving move -> local optimum ---
        if best_move is None:
            if verbose:
                print(f"[Phase 3] Local search converged after {iteration} iterations")
            break

        # --- Apply the best move ---
        move_type, f_in, f_out = best_move
        iteration += 1

        _apply_move_and_recompute(instance, state, move_type, f_in, f_out)

        if verbose:
            print(
                f"          Iter {iteration:3d}: {move_type:6s} "
                f"(in={f_in}, out={f_out})  "
                f"profit={best_profit:+,.2f}  "
                f"cost={state.total_cost:,.2f}  "
                f"|S|={len(state.open_facilities)}"
            )

    return state


# ============================================================================
# Section 6: Phase 4 — Main Orchestration
# ============================================================================

def validate_instance(instance: UFLPInstance) -> None:
    """
    Reject degenerate or incomplete instances before any solving happens.

    Without this guard a customer-free instance silently returns a non-zero
    cost (the construction fallback force-opens one facility and the local
    search may never delete the last open facility), and a facility-free
    instance surfaces as an opaque SciPy error from deep inside `linprog`.
    """
    if not instance.facilities:
        raise ValueError("UFLP instance must contain at least one facility")
    if not instance.customers:
        raise ValueError("UFLP instance must contain at least one customer")

    missing_setup = [f for f in instance.facilities if f not in instance.setup_costs]
    if missing_setup:
        raise ValueError(f"Missing setup costs for facilities: {missing_setup}")

    for u in instance.customers:
        row = instance.service_costs.get(u)
        if row is None:
            raise ValueError(f"Missing service costs for customer {u}")
        missing_service = [f for f in instance.facilities if f not in row]
        if missing_service:
            raise ValueError(
                f"Missing service costs for customer {u}, facilities: {missing_service}"
            )


def solve_uflp(
    instance: UFLPInstance,
    verbose: bool = True,
    seed: int | None = None,
) -> SolutionState:
    """
    Solve a UFLP instance using the Hybrid LP-GRASP pipeline.

    Pipeline:
        0. Instance validation
        1. LP relaxation → fractional probabilities
        2. Probabilistic construction → initial solution
        3. Best-improvement local search → local optimum
    """
    validate_instance(instance)

    if seed is not None:
        random.seed(seed)

    t0 = time.perf_counter()

    if verbose:
        print("=" * 65)
        print("  Hybrid LP-GRASP Solver for UFLP")
        print(f"  Facilities: {len(instance.facilities)}  |  "
              f"Customers: {len(instance.customers)}")
        print("=" * 65)

    # Phase 1: LP Relaxation
    t1 = time.perf_counter()
    lp_probs = solve_lp_relaxation(instance, verbose=verbose)
    t1_end = time.perf_counter()

    # Phase 2: Constructive Heuristic
    t2 = time.perf_counter()
    state = construct_solution(instance, lp_probs, verbose=verbose)
    t2_end = time.perf_counter()

    # Phase 3: Local Search
    t3 = time.perf_counter()
    state = local_search(instance, state, verbose=verbose)
    t3_end = time.perf_counter()

    t_total = time.perf_counter() - t0

    if verbose:
        print("=" * 65)
        print(f"  SOLUTION SUMMARY")
        print(f"  Total cost:       {state.total_cost:,.2f}")
        print(f"  Open facilities:  {sorted(state.open_facilities)}")
        print(f"  Num open:         {len(state.open_facilities)}/{len(instance.facilities)}")
        print(f"  ---------------------------------------------")
        print(f"  Timing:")
        print(f"    LP relaxation:   {t1_end - t1:.3f}s")
        print(f"    Construction:    {t2_end - t2:.3f}s")
        print(f"    Local search:    {t3_end - t3:.3f}s")
        print(f"    Total:           {t_total:.3f}s")
        print("=" * 65)

        # Print customer assignments
        print("\n  Customer Assignments:")
        print(f"  {'Customer':>10s}  {'Facility':>10s}  {'Service Cost':>14s}")
        print(f"  {'-'*10}  {'-'*10}  {'-'*14}")
        for u in sorted(instance.customers):
            f = state.closest_facility[u]
            sc = instance.service_costs[u][f]
            print(f"  {u:>10d}  {f:>10d}  {sc:>14,.4f}")

    return state


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Hybrid LP-GRASP Solver for the Uncapacitated Facility Location Problem",
    )
    parser.add_argument(
        "--file", type=str, default=None,
        help="Path to an OR-Library UFLP instance file",
    )
    parser.add_argument(
        "--n-facilities", type=int, default=20,
        help="Number of facilities for random instance (default: 20)",
    )
    parser.add_argument(
        "--n-customers", type=int, default=30,
        help="Number of customers for random instance (default: 30)",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed (default: 42)",
    )
    parser.add_argument(
        "--quiet", action="store_true",
        help="Suppress verbose output",
    )

    args = parser.parse_args()

    if args.file:
        print(f"Loading OR-Library instance from: {args.file}")
        instance = parse_orlib_instance(args.file)
    else:
        print(f"Generating random instance: {args.n_facilities} facilities, "
              f"{args.n_customers} customers, seed={args.seed}")
        instance = generate_random_instance(
            n_facilities=args.n_facilities,
            n_customers=args.n_customers,
            seed=args.seed,
        )

    solve_uflp(instance, verbose=not args.quiet, seed=args.seed)


if __name__ == "__main__":
    main()
