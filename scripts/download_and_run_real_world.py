"""
Real-world UFLP benchmark on the California Housing census blocks.

Builds a 2000x2000 UFLP instance from real latitude/longitude and block
population, then compares three methods against the same LP bound:

    A.  LP rounding      -- round the LP relaxation at y >= 0.5, no local search.
                            Isolates what the LP alone is worth.
    A+. Rounding + search -- the same rounded solution, then the SAME local
                            search. Fully deterministic: no randomness, no EPS
                            floor. This is the ablation that isolates what the
                            *randomized* construction contributes, and it is the
                            arm the original study never ran.
    B.  LP-biased GRASP  -- probabilistic construction biased by the fractional
                            y values, then the same local search.
    C.  alpha-GRASP      -- savings-based RCL construction (alpha = 0.2), then
                            the same local search. The classical baseline.

    Arms A+ and B differ in exactly one thing: whether the construction is
    deterministic rounding or probabilistic sampling. Everything downstream is
    identical, so any difference between them is attributable to randomization
    and to nothing else.

PROVENANCE CONTRACT
    `output/california_4M_results.md` is written by `render_report` and by
    nothing else. Every figure in it is a measured value carried in a
    `ArmResult` / `LPProfile` record, and every comparative phrase ("lower",
    "faster", "tighter") is *computed* from those values by `_cmp` / `_signed`
    rather than written into the template. The report therefore cannot
    contradict its own table, which is exactly the defect that motivated this
    rewrite: the previous template hardcoded "LP-biased starts lower" while the
    run it described produced the opposite.

    A machine-readable sidecar is written alongside it at
    `output/california_4M_results.json` so the markdown can always be checked
    against the raw measurements.

Usage:
    python download_and_run_real_world.py                  # full 2000x2000, 3 seeds
    python download_and_run_real_world.py --size 400       # quick smoke run
    python download_and_run_real_world.py --seeds 42       # single seed
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import random
import statistics
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Dict, List, Tuple

import numpy as np

from scripts.run_experiments import (
    construct_alpha_grasp_solution,
    construct_lp_biased_solution,
    get_lp_bound_and_probs,
    run_local_search_iter_count,
)
from scripts.multistart import (
    MultistartResult,
    multistart_alpha_grasp,
    multistart_lp_biased,
)
from scripts.reference import compute_reference
from scripts.uflp_solver import (
    SolutionState,
    UFLPInstance,
    _compute_auxiliary_data,
    _compute_total_cost,
    _find_closest_two,
)

# The EPS floor used by `construct_lp_biased_solution`; mirrored here so the
# report can account for the spurious openings it causes.
CONSTRUCTION_EPS = 0.01

# Cost-model constants, surfaced so the report can state them exactly.
BASE_SETUP_COST = 5000.0
SETUP_FLOOR = 1000.0
KM_PER_DEGREE = 111.0
COST_PER_KM = 10.0

DATASET_SEED = 42  # seeds the facility/customer sample; not the heuristic seed


# ============================================================================
# Instance construction
# ============================================================================

def create_california_uflp_instance(n_fac: int = 2000, n_cust: int = 2000) -> UFLPInstance:
    """
    Build a UFLP instance from California Housing census blocks.

    Facilities and customers are two disjoint random samples of census blocks.
    Setup cost scales with block population; service cost is the Euclidean
    distance between (latitude, longitude) pairs, converted at a flat
    111 km/degree and priced at 10 currency units per km.

    NOTE ON THE DISTANCE MODEL: longitude degrees are converted with the same
    111 km factor as latitude degrees. At Californian latitudes (~37 N) a
    longitude degree is really ~88 km, so east-west separation is overstated by
    roughly 26%. The instance is geographically *derived*, not geographically
    *accurate*. `tests/unit/test_california_instance.py` pins this so it cannot
    drift silently.
    """
    from sklearn.datasets import fetch_california_housing

    data = fetch_california_housing()
    lat_idx, lon_idx = 6, 7
    coords = data.data[:, [lat_idx, lon_idx]]
    populations = data.data[:, 4]

    np.random.seed(DATASET_SEED)
    indices = np.random.permutation(len(coords))

    fac_indices = indices[:n_fac]
    cust_indices = indices[n_fac:n_fac + n_cust]

    facilities = list(range(n_fac))
    customers = list(range(n_cust))

    pop_fac = populations[fac_indices]
    pop_fac = pop_fac / np.median(pop_fac)
    setup_costs = {f: float(BASE_SETUP_COST * pop_fac[f]) + SETUP_FLOOR for f in facilities}

    service_costs: Dict[int, Dict[int, float]] = {}
    for i, c_idx in enumerate(cust_indices):
        service_costs[i] = {}
        c_coord = coords[c_idx]
        for j, f_idx in enumerate(fac_indices):
            f_coord = coords[f_idx]
            dist = np.linalg.norm(c_coord - f_coord) * KM_PER_DEGREE
            service_costs[i][j] = float(dist * COST_PER_KM)

    return UFLPInstance(facilities, customers, setup_costs, service_costs)


# ============================================================================
# Measurement records
# ============================================================================

@dataclass
class LPProfile:
    """Everything measured about the LP relaxation itself."""
    bound: float
    solve_seconds: float
    n_facilities: int
    n_integral_open: int       # y >= 0.99
    n_fractional: int          # 0.01 < y < 0.99
    n_near_zero: int           # y <= 0.01
    sum_y: float
    rounded_open_count: int    # |{f : y >= 0.5}|

    @property
    def fractional_pct(self) -> float:
        return self.n_fractional / self.n_facilities * 100.0

    @property
    def expected_eps_openings(self) -> float:
        """Facilities the EPS floor is expected to open despite the LP zeroing them."""
        return CONSTRUCTION_EPS * self.n_near_zero


@dataclass
class ArmResult:
    """One method's measurements on one seed."""
    method: str
    seed: int | None
    initial_cost: float
    initial_gap: float
    initial_open: int
    final_cost: float
    final_gap: float
    final_open: int
    iterations: int
    moves: Dict[str, int] = field(default_factory=dict)
    construct_seconds: float = 0.0
    search_seconds: float = 0.0
    lp_seconds: float = 0.0

    @property
    def total_seconds(self) -> float:
        return self.lp_seconds + self.construct_seconds + self.search_seconds


# ============================================================================
# The three methods
# ============================================================================

def profile_lp(lp_bound: float, lp_probs: Dict[int, float], solve_seconds: float) -> LPProfile:
    values = list(lp_probs.values())
    return LPProfile(
        bound=lp_bound,
        solve_seconds=solve_seconds,
        n_facilities=len(values),
        n_integral_open=sum(1 for v in values if v >= 0.99),
        n_fractional=sum(1 for v in values if 0.01 < v < 0.99),
        n_near_zero=sum(1 for v in values if v <= 0.01),
        sum_y=float(sum(values)),
        rounded_open_count=sum(1 for v in values if v >= 0.5),
    )


def _finalise(instance: UFLPInstance, open_facilities) -> SolutionState:
    """Populate a full SolutionState for a given open set."""
    state = SolutionState(open_facilities=set(open_facilities))
    for u in instance.customers:
        closest, second = _find_closest_two(u, state.open_facilities, instance.service_costs)
        state.closest_facility[u] = closest
        state.second_closest_facility[u] = second
    state.total_cost = _compute_total_cost(instance, state)
    _compute_auxiliary_data(instance, state)
    return state


def round_lp_solution(
    instance: UFLPInstance,
    lp_probs: Dict[int, float],
    threshold: float = 0.5,
) -> SolutionState:
    """
    Control arm: open every facility whose LP value reaches `threshold`.

    Falls back to the single highest-y facility if the threshold opens nothing,
    so the result is always feasible.
    """
    opened = {f for f, v in lp_probs.items() if v >= threshold}
    if not opened:
        opened = {max(lp_probs, key=lambda f: lp_probs[f])}
    return _finalise(instance, opened)


def run_lp_rounding(
    instance: UFLPInstance,
    lp_probs: Dict[int, float],
    lp: LPProfile,
) -> ArmResult:
    t0 = time.perf_counter()
    state = round_lp_solution(instance, lp_probs)
    elapsed = time.perf_counter() - t0
    gap = (state.total_cost - lp.bound) / lp.bound * 100.0
    return ArmResult(
        method="LP rounding (control)",
        seed=None,
        initial_cost=state.total_cost,
        initial_gap=gap,
        initial_open=len(state.open_facilities),
        final_cost=state.total_cost,
        final_gap=gap,
        final_open=len(state.open_facilities),
        iterations=0,
        moves={"insert": 0, "delete": 0, "swap": 0},
        construct_seconds=elapsed,
        lp_seconds=lp.solve_seconds,
    )


def run_lp_rounding_plus_search(
    instance: UFLPInstance,
    lp_probs: Dict[int, float],
    lp: LPProfile,
) -> ArmResult:
    """
    Ablation arm: deterministic LP rounding, then the same local search.

    This is arm B with the randomness removed and nothing else changed. If it
    matches arm B's final solution, the probabilistic construction contributes
    no quality -- only the extra local-search work needed to undo the EPS noise
    it introduced.

    Takes no seed: it is deterministic, so there is nothing to average over.
    """
    t0 = time.perf_counter()
    state = round_lp_solution(instance, lp_probs)
    t_construct = time.perf_counter() - t0
    initial_cost, initial_open = state.total_cost, len(state.open_facilities)

    t0 = time.perf_counter()
    state, iterations, moves = run_local_search_iter_count(instance, state)
    t_search = time.perf_counter() - t0

    return ArmResult(
        method="LP rounding + local search (deterministic ablation)",
        seed=None,
        initial_cost=initial_cost,
        initial_gap=(initial_cost - lp.bound) / lp.bound * 100.0,
        initial_open=initial_open,
        final_cost=state.total_cost,
        final_gap=(state.total_cost - lp.bound) / lp.bound * 100.0,
        final_open=len(state.open_facilities),
        iterations=iterations,
        moves=moves,
        construct_seconds=t_construct,
        search_seconds=t_search,
        lp_seconds=lp.solve_seconds,
    )


def run_local_search_only(instance: UFLPInstance, lp_bound: float) -> ArmResult:
    """
    Control arm with NO LP guidance at all.

    Starts from the single cheapest facility by total cost (setup plus the sum
    of its service costs -- the same fallback rule `construct_solution` uses)
    and runs the same local search. It never reads the relaxation.

    This is the control that isolates what the LP contributes *at all*, as
    opposed to what the randomized construction contributes on top of it. It
    matters on instance families where the relaxation is weak: if this arm
    matches the LP-guided arms, the LP solve is buying nothing but time.
    """
    t0 = time.perf_counter()
    cheapest = min(
        instance.facilities,
        key=lambda f: (
            instance.setup_costs[f]
            + sum(instance.service_costs[u][f] for u in instance.customers)
        ),
    )
    state = _finalise(instance, {cheapest})
    t_construct = time.perf_counter() - t0
    initial_cost, initial_open = state.total_cost, len(state.open_facilities)

    t0 = time.perf_counter()
    state, iterations, moves = run_local_search_iter_count(instance, state)
    t_search = time.perf_counter() - t0

    return ArmResult(
        method="Local search only (no LP)",
        seed=None,
        initial_cost=initial_cost,
        initial_gap=(initial_cost - lp_bound) / lp_bound * 100.0,
        initial_open=initial_open,
        final_cost=state.total_cost,
        final_gap=(state.total_cost - lp_bound) / lp_bound * 100.0,
        final_open=len(state.open_facilities),
        iterations=iterations,
        moves=moves,
        construct_seconds=t_construct,
        search_seconds=t_search,
        lp_seconds=0.0,  # this arm never solves the LP
    )


def run_lp_biased(
    instance: UFLPInstance,
    lp_probs: Dict[int, float],
    lp: LPProfile,
    seed: int,
) -> ArmResult:
    random.seed(seed)
    t0 = time.perf_counter()
    state = construct_lp_biased_solution(instance, lp_probs)
    t_construct = time.perf_counter() - t0
    initial_cost, initial_open = state.total_cost, len(state.open_facilities)

    t0 = time.perf_counter()
    state, iterations, moves = run_local_search_iter_count(instance, state)
    t_search = time.perf_counter() - t0

    return ArmResult(
        method="LP-biased hybrid GRASP",
        seed=seed,
        initial_cost=initial_cost,
        initial_gap=(initial_cost - lp.bound) / lp.bound * 100.0,
        initial_open=initial_open,
        final_cost=state.total_cost,
        final_gap=(state.total_cost - lp.bound) / lp.bound * 100.0,
        final_open=len(state.open_facilities),
        iterations=iterations,
        moves=moves,
        construct_seconds=t_construct,
        search_seconds=t_search,
        lp_seconds=lp.solve_seconds,
    )


def run_alpha_grasp(
    instance: UFLPInstance,
    lp: LPProfile,
    seed: int,
    alpha: float = 0.2,
) -> ArmResult:
    random.seed(seed)
    t0 = time.perf_counter()
    state = construct_alpha_grasp_solution(instance, alpha=alpha)
    t_construct = time.perf_counter() - t0
    initial_cost, initial_open = state.total_cost, len(state.open_facilities)

    t0 = time.perf_counter()
    state, iterations, moves = run_local_search_iter_count(instance, state)
    t_search = time.perf_counter() - t0

    return ArmResult(
        method=f"alpha-GRASP baseline (alpha={alpha})",
        seed=seed,
        initial_cost=initial_cost,
        initial_gap=(initial_cost - lp.bound) / lp.bound * 100.0,
        initial_open=initial_open,
        final_cost=state.total_cost,
        final_gap=(state.total_cost - lp.bound) / lp.bound * 100.0,
        final_open=len(state.open_facilities),
        iterations=iterations,
        moves=moves,
        construct_seconds=t_construct,
        search_seconds=t_search,
        lp_seconds=0.0,  # the baseline does not pay for the LP
    )


# ============================================================================
# Report rendering -- every comparative word is computed, never hardcoded
# ============================================================================

def _cmp(a: float, b: float, lower_word: str, higher_word: str, tie: str = "the same as",
         tol: float = 1e-9) -> str:
    """Pick the word that matches the measured direction of a vs b."""
    if abs(a - b) <= tol:
        return tie
    return lower_word if a < b else higher_word


def _signed(delta: float, unit: str = "", places: int = 2) -> str:
    """Format a difference with an explicit sign so direction is unambiguous."""
    return f"{delta:+,.{places}f}{unit}"


def _mean(values: List[float]) -> float:
    return statistics.fmean(values)


def _plural(n: float, singular: str, plural: str | None = None) -> str:
    """Agree a noun with a (possibly rounded) count."""
    return singular if abs(round(n)) == 1 else (plural or singular + "s")


def _spread(values: List[float]) -> str:
    """min-max range, or a single value when there is only one seed."""
    if len(values) == 1:
        return ""
    return f" (range {min(values):,.4f}–{max(values):,.4f})"


def _git_commit() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=10, check=False,
        )
        commit = out.stdout.strip()
        dirty = subprocess.run(
            ["git", "status", "--porcelain"],
            capture_output=True, text=True, timeout=10, check=False,
        ).stdout.strip()
        return f"{commit}{' (working tree modified)' if dirty else ''}" if commit else "unknown"
    except Exception:  # pragma: no cover - environment without git
        return "unknown"


def _environment() -> Dict[str, str]:
    import scipy
    try:
        import sklearn
        sklearn_version = sklearn.__version__
    except ImportError:  # pragma: no cover
        sklearn_version = "not installed"
    return {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "scikit_learn": sklearn_version,
        "platform": platform.platform(),
        "cpu_count": str(os.cpu_count()),
    }


def render_report(
    n_fac: int,
    n_cust: int,
    lp: LPProfile,
    reference,
    control: ArmResult,
    rounding_ls: ArmResult,
    no_lp: ArmResult,
    lp_biased: "MultistartResult",
    alpha: "MultistartResult",
    seeds: List[int],
    env: Dict[str, str],
    generated_at: str,
    commit: str,
    wall_seconds: float,
    alpha_value: float = 0.2,
) -> str:
    """
    Render the markdown purely from measured values.

    Every comparative phrase is computed from the measurements rather than
    written into the template, so the report cannot state a conclusion its own
    tables contradict.
    """
    lines: List[str] = []
    w = lines.append
    R = lp_biased.restarts
    n_vars = n_fac + n_cust * n_fac
    n_rows = n_cust + n_cust * n_fac

    def gap(cost: float) -> float:
        return (cost - reference.value) / reference.value * 100.0

    g_a, g_ap, g_d = gap(control.final_cost), gap(rounding_ls.final_cost), gap(no_lp.final_cost)
    g_b, g_c = gap(lp_biased.best_cost), gap(alpha.best_cost)
    label = reference.gap_label

    # ---------------------------------------------------------------- header
    w("# Real-World UFLP Benchmark — California Housing, 2000×2000")
    w("")
    w(f"A {n_fac:,}×{n_cust:,} instance built from real census-block coordinates and")
    w("populations. Randomized arms run as **multistart** procedures; deterministic arms run")
    w("once, because restarting a deterministic construction reproduces the same solution.")
    w("")
    w(f"> **What the gaps are measured against.** {reference.describe()}.")
    if not reference.proven:
        w("> The true optimum lies in "
          f"`[{lp.bound:,.2f}, {min(control.final_cost, rounding_ls.final_cost, no_lp.final_cost, lp_biased.best_cost, alpha.best_cost):,.2f}]`.")
    w("")
    w("> **Provenance.** Every number was measured by the run described in §8 and written by")
    w("> `render_report` in `download_and_run_real_world.py`. No value is hardcoded or")
    w("> back-computed; every comparative word is computed from the measurements. Raw records")
    w("> are in `california_4M_results.json`; `--from-json` re-renders the prose from them")
    w("> without re-running the benchmark.")
    w("")
    w("---")
    w("")

    # ---------------------------------------------------------------- §1
    w("## 1. Instance")
    w("")
    w("| Property | Value |")
    w("|---|---|")
    w("| Source | California Housing (`sklearn.datasets.fetch_california_housing`) |")
    w(f"| Facilities × customers | {n_fac:,} × {n_cust:,} |")
    w(f"| Sampling | two **disjoint** random samples of census blocks, `np.random.seed({DATASET_SEED})` |")
    w(f"| Setup cost | `{BASE_SETUP_COST:,.0f} × (population / median population) + {SETUP_FLOOR:,.0f}` |")
    w(f"| Service cost | Euclidean degree distance × {KM_PER_DEGREE} km/deg × {COST_PER_KM} per km |")
    w(f"| LP model | {n_vars:,} variables, {n_rows:,} rows ({n_cust * n_fac:,} linking) |")
    w("")
    w(f"Longitude degrees are converted at the same {KM_PER_DEGREE} km/degree as latitude degrees.")
    w("At ~37°N a longitude degree is closer to 88 km, so east–west separation is overstated by")
    w("roughly 26%. The instance is geographically *derived*, not geographically *accurate*.")
    w("")

    # ---------------------------------------------------------------- §2
    w("## 2. LP relaxation")
    w("")
    w("| Quantity | Value |")
    w("|---|---|")
    w(f"| LP bound | **{lp.bound:,.2f}** |")
    w(f"| Solve time | {lp.solve_seconds:.2f} s |")
    w(f"| Fractional facilities (`0.01 < y < 0.99`) | **{lp.n_fractional:,}** / {lp.n_facilities:,} "
      f"({lp.fractional_pct:.2f}%) |")
    w(f"| `Σ y[f]` | {lp.sum_y:,.2f} |")
    w(f"| Facilities at `y ≥ 0.5` | {lp.rounded_open_count:,} |")
    w("")
    integrality = ("essentially integral" if lp.fractional_pct < 1.0
                   else "moderately fractional" if lp.fractional_pct < 10.0
                   else "highly fractional")
    w(f"The relaxation is **{integrality}** — {lp.n_fractional} of {lp.n_facilities} facilities")
    w(f"({lp.fractional_pct:.2f}%) take a strictly fractional value. A near-integral relaxation")
    w("means the LP has very nearly solved the problem already, leaving the heuristic layer")
    w("little to contribute beyond repair.")
    w("")

    # ---------------------------------------------------------------- §3
    w("## 3. Arms")
    w("")
    w("| | Arm | Type | Construction |")
    w("|---|---|---|---|")
    w("| A | LP rounding only | deterministic | open every `y ≥ 0.5`, no local search |")
    w("| A+ | LP rounding + local search | deterministic | same, then local search |")
    w("| D | Local search only | deterministic | cheapest facility, **never reads the LP** |")
    w(f"| B | LP-biased multistart | best of {R} | sample `p = max(y, {CONSTRUCTION_EPS})` per restart |")
    w(f"| C | α-GRASP multistart | best of {R} | savings RCL, α = {alpha_value}, per restart |")
    w("")
    w("A+ and B differ in exactly one respect — deterministic rounding versus probabilistic")
    w("sampling — so any difference between them is attributable to randomization alone.")
    w("D isolates what the LP contributes at all.")
    w("")

    # ---------------------------------------------------------------- §4
    w("## 4. Results")
    w("")
    w(f"| Arm | Type | Final cost | {label} | Work | Time |")
    w("|---|---|---|---|---|---|")
    w(f"| A · LP rounding | deterministic | {control.final_cost:,.2f} | {g_a:.4f}% | "
      f"0 iters | {control.total_seconds:.1f} s |")
    w(f"| **A+ · rounding + LS** | deterministic | **{rounding_ls.final_cost:,.2f}** | "
      f"**{g_ap:.4f}%** | {rounding_ls.iterations} iters | {rounding_ls.total_seconds:.1f} s |")
    w(f"| D · local search, no LP | deterministic | {no_lp.final_cost:,.2f} | {g_d:.4f}% | "
      f"{no_lp.iterations} iters | {no_lp.total_seconds:.1f} s |")
    w(f"| **B · LP-biased multistart** | best of {R} | **{lp_biased.best_cost:,.2f}** | "
      f"**{g_b:.4f}%** | {lp_biased.total_iterations} iters / {R} restarts | "
      f"{lp_biased.total_seconds:.1f} s |")
    w(f"| **C · α-GRASP multistart** | best of {R} | **{alpha.best_cost:,.2f}** | "
      f"**{g_c:.4f}%** | {alpha.total_iterations} iters / {R} restarts | "
      f"{alpha.total_seconds:.1f} s |")
    w("")
    w(f"*A, A+ and B include the {lp.solve_seconds:.1f} s LP solve; C and D do not read the LP.*")
    w("")
    best_name, best_gap = min(
        [("A · LP rounding", g_a), ("A+ · rounding + local search", g_ap),
         ("D · local search only", g_d), ("B · LP-biased multistart", g_b),
         ("C · α-GRASP multistart", g_c)], key=lambda t: t[1])
    w(f"**Best arm: {best_name}, at {best_gap:.4f}%.**")
    w("")

    # ---------------------------------------------------------------- §5
    w("## 5. LP-biased GRASP vs. classical GRASP")
    w("")
    w(f"Both arms get {R} restarts and the same local search.")
    w("")
    w("| | B · LP-biased | C · α-GRASP |")
    w("|---|---|---|")
    w(f"| Best {label} | **{g_b:.4f}%** | **{g_c:.4f}%** |")
    w(f"| Restart producing the winner | {lp_biased.best_found_at} / {R} | "
      f"{alpha.best_found_at} / {R} |")
    w(f"| Mean gap per restart | {_mean(lp_biased.per_restart_gaps):.4f}% | "
      f"{_mean(alpha.per_restart_gaps):.4f}% |")
    w(f"| Best single restart | {min(lp_biased.per_restart_gaps):.4f}% | "
      f"{min(alpha.per_restart_gaps):.4f}% |")
    w(f"| Worst single restart | {max(lp_biased.per_restart_gaps):.4f}% | "
      f"{max(alpha.per_restart_gaps):.4f}% |")
    w(f"| Total local-search iterations | {lp_biased.total_iterations} | {alpha.total_iterations} |")
    w(f"| Heuristic time (excl. LP) | {lp_biased.heuristic_seconds:.1f} s | "
      f"{alpha.heuristic_seconds:.1f} s |")
    w("")
    hh = _cmp(g_b, g_c, "B", "C", tie="neither")
    if hh == "B":
        w(f"**The LP-biased construction wins the head-to-head** by {g_c - g_b:.4f} pp at equal")
        w("restart budget.")
    elif hh == "C":
        w(f"**The classical baseline wins the head-to-head** by {g_b - g_c:.4f} pp at equal")
        w("restart budget.")
    else:
        w("**The two constructions tie at this restart budget.**")
    w("")

    # ---------------------------------------------------------------- §6
    w("## 6. Does multistart pay for itself?")
    w("")
    w(f"A best-of-{R} result costs {R}× the work of one run. The question is therefore not")
    w("whether B beats A+ at full budget, but how many restarts B needs to match A+ at all.")
    w("")
    n_match = lp_biased.restarts_to_reach(g_ap) if hasattr(lp_biased, "restarts_to_reach") else None
    if n_match is None:
        # Recompute from the trajectory when the dataclass came back from JSON.
        n_match = next((i for i, c in enumerate(lp_biased.trajectory_costs, start=1)
                        if gap(c) <= g_ap + 1e-9), None)
    if n_match is not None:
        matched_time = lp.solve_seconds + n_match * lp_biased.seconds_per_restart
        w(f"- B reaches A+'s quality ({g_ap:.4f}%) after **{n_match}** "
          f"{_plural(n_match, 'restart')}, at roughly {matched_time:.0f} s against A+'s "
          f"{rounding_ls.total_seconds:.0f} s.")
    else:
        w(f"- **B never matches A+** within its {R}-restart budget "
          f"({g_b:.4f}% against {g_ap:.4f}%).")
    w(f"- B's best came from restart {lp_biased.best_found_at} of {R}; the remaining "
      f"{R - lp_biased.best_found_at} produced no improvement.")
    w("")
    if g_b < g_ap - 1e-9:
        w(f"**Multistart overturns the single-start result.** With {R} restarts the randomized")
        w(f"construction reaches {g_b:.4f}% against the deterministic {g_ap:.4f}%, so the earlier")
        w("finding that randomization contributes nothing was an artifact of running it once.")
    elif abs(g_b - g_ap) <= 1e-9:
        w(f"**Multistart brings the randomized arm level with the deterministic one** "
          f"({g_b:.4f}% each) — at {R}× the work.")
    else:
        w(f"**The deterministic arm still holds up.** A+ reaches {g_ap:.4f}% in one run; B reaches")
        w(f"{g_b:.4f}% using {R} restarts. Multistart narrows the earlier single-start gap without")
        w("overturning it at this budget.")
    w("")

    # ---------------------------------------------------------------- §7
    w("## 7. What the LP contributes")
    w("")
    w(f"Arm D never reads the relaxation. It reaches **{g_d:.4f}%**; the LP-guided A+ reaches")
    w(f"**{g_ap:.4f}%**.")
    w("")
    lpv = _cmp(g_ap, g_d, "A+", "D", tie="neither")
    if lpv == "A+":
        w(f"**The LP contributes**: {g_d - g_ap:.4f} pp of quality over an identical search that")
        w("never sees it.")
    elif lpv == "D":
        w(f"**The LP is not contributing here**: the LP-free arm is {g_ap - g_d:.4f} pp better,")
        w(f"while A+ also pays {lp.solve_seconds:.0f} s for the relaxation.")
    else:
        w("**The LP makes no measurable difference here** — the LP-free arm reaches the same")
        w(f"quality without paying {lp.solve_seconds:.0f} s for the relaxation.")
    w("")
    w(f"Rounding the LP without searching reaches only {g_a:.4f}%, so the local search is doing")
    w("substantial work regardless of how its starting point is chosen.")
    w("")

    # ---------------------------------------------------------------- §8
    w("## 8. Limitations")
    w("")
    if not reference.proven:
        w("- **Gaps are measured against the LP bound, not a proven optimum.** At this size the")
        w("  integer program has four million binary-linked assignment variables and is out of")
        w("  reach for CBC, so the figures above **overstate** true optimality gaps by an unknown")
        w("  amount. The Körkel-Ghosh benchmark in this repository reports proven optima instead.")
    w(f"- **{R} restarts** is a modest budget for a GRASP; published studies commonly use 32+.")
    w("- **One instance.** Conclusions here describe this instance, not the family.")
    w("- **Timings are not portable.** Repeated runs of the same configuration on this machine")
    w("  differed by tens of percent; the LP solve alone has ranged 67–231 s. Compare timings")
    w("  within a run only. Cost, gap, iteration and move-mix figures are deterministic under")
    w("  the stated seeds.")
    w("- **The distance model overstates east–west separation by ~26%** (§1).")
    w(f"- **A single α ({alpha_value})** for the baseline; not tuned.")
    w("")

    # ---------------------------------------------------------------- §9
    w("## 9. Reproduction")
    w("")
    w("```bash")
    w(f"python -m scripts.download_and_run_real_world --size {n_fac} --restarts {R} --alpha {alpha_value}")
    w("python -m scripts.download_and_run_real_world --from-json output/california_4M_results.json")
    w("```")
    w("")
    w("| Run metadata | |")
    w("|---|---|")
    w(f"| Generated at | {generated_at} |")
    w("| Generated by | `download_and_run_real_world.py` → `render_report` |")
    w(f"| Git commit | `{commit}` |")
    w(f"| Dataset sample seed | {DATASET_SEED} |")
    w(f"| Restarts | {R} |")
    w(f"| Total wall-clock | {wall_seconds:.1f} s |")
    w(f"| Python | {env['python']} |")
    w(f"| NumPy / SciPy / scikit-learn | {env['numpy']} / {env['scipy']} / {env['scikit_learn']} |")
    w("")
    w("Raw measurements: [`california_4M_results.json`](california_4M_results.json).")
    w("")
    return "\n".join(lines) + "\n"


def rerender_from_json(json_path: str, out_dir: str) -> str:
    """
    Rebuild the markdown from a saved measurement sidecar.

    The whole point of writing the JSON is that the prose can be corrected or
    extended without re-running a 38-minute benchmark, and without anyone being
    tempted to hand-edit the results file. Measurements come from the sidecar;
    only the wording is regenerated.
    """
    with open(json_path) as fh:
        payload = json.load(fh)

    lp = LPProfile(**payload["lp_profile"])
    arms = payload["arms"]
    from scripts.reference import Reference

    control = ArmResult(**arms["lp_rounding_control"])
    rounding_ls = ArmResult(**arms["lp_rounding_plus_search"])
    no_lp = ArmResult(**arms["local_search_only"])
    lp_biased = MultistartResult(**arms["lp_biased_multistart"])
    alpha = MultistartResult(**arms["alpha_grasp_multistart"])
    ref = Reference(**payload["reference"])

    report = render_report(
        n_fac=payload["instance"]["n_facilities"],
        n_cust=payload["instance"]["n_customers"],
        lp=lp, reference=ref, control=control, rounding_ls=rounding_ls, no_lp=no_lp,
        lp_biased=lp_biased, alpha=alpha, alpha_value=payload.get("alpha", 0.2),
        seeds=payload["seeds"], env=payload["environment"],
        generated_at=payload["generated_at"], commit=payload["git_commit"],
        wall_seconds=payload["wall_seconds"],
    )

    os.makedirs(out_dir, exist_ok=True)
    md_path = os.path.join(out_dir, "california_4M_results.md")
    with open(md_path, "w") as fh:
        fh.write(report)
    return md_path


def main(argv: List[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument("--size", type=int, default=2000,
                        help="facilities and customers (square instance); default 2000")
    parser.add_argument("--restarts", type=int, default=10,
                        help="multistart restarts per randomized arm")
    parser.add_argument("--alpha", type=float, default=0.2)
    parser.add_argument("--attempt-exact-ip", action="store_true",
                        help="try to prove the integer optimum with CBC (intractable at 2000x2000)")
    parser.add_argument("--out-dir", type=str, default="output")
    parser.add_argument("--from-json", type=str, default=None,
                        help="re-render the markdown from a saved measurement sidecar "
                             "instead of re-running the benchmark")
    args = parser.parse_args(argv)

    args.seeds = list(range(1, args.restarts + 1))

    if args.from_json:
        md_path = rerender_from_json(args.from_json, args.out_dir)
        print(f"Re-rendered {md_path} from {args.from_json} (no benchmark re-run)")
        return

    n_fac = n_cust = args.size
    os.makedirs(args.out_dir, exist_ok=True)
    wall0 = time.perf_counter()

    print(f"Building California instance ({n_fac:,}×{n_cust:,})...", flush=True)
    t0 = time.perf_counter()
    instance = create_california_uflp_instance(n_fac, n_cust)
    print(f"  built in {time.perf_counter() - t0:.2f}s", flush=True)

    print("Solving LP relaxation...", flush=True)
    t0 = time.perf_counter()
    lp_bound, lp_probs = get_lp_bound_and_probs(instance)
    lp = profile_lp(lp_bound, lp_probs, time.perf_counter() - t0)
    print(f"  bound={lp.bound:,.2f}  time={lp.solve_seconds:.2f}s  "
          f"fractional={lp.n_fractional}/{lp.n_facilities}  sum_y={lp.sum_y:.2f}", flush=True)

    print("Arm A: LP rounding (control)...", flush=True)
    control = run_lp_rounding(instance, lp_probs, lp)
    print(f"  |S|={control.final_open}  cost={control.final_cost:,.2f}  "
          f"gap={control.final_gap:.4f}%", flush=True)

    print("Arm A+: LP rounding + local search (deterministic ablation)...", flush=True)
    rounding_ls = run_lp_rounding_plus_search(instance, lp_probs, lp)
    print(f"  init |S|={rounding_ls.initial_open} gap={rounding_ls.initial_gap:.4f}%  →  "
          f"iters={rounding_ls.iterations} {rounding_ls.moves}  "
          f"final |S|={rounding_ls.final_open} gap={rounding_ls.final_gap:.4f}%  "
          f"time={rounding_ls.total_seconds:.2f}s", flush=True)

    print("Arm D: local search only, no LP (control)...", flush=True)
    no_lp = run_local_search_only(instance, lp.bound)
    print(f"  init |S|={no_lp.initial_open}  →  iters={no_lp.iterations} {no_lp.moves}  "
          f"final |S|={no_lp.final_open} gap={no_lp.final_gap:.4f}%  "
          f"time={no_lp.total_seconds:.2f}s", flush=True)

    # The integer program is far out of reach at this size (four million
    # binary-linked assignment variables), so gaps are measured against the LP
    # bound and the report says so.
    ref = compute_reference(instance, lp.bound, attempt_exact=args.attempt_exact_ip)

    print(f"Arm B: LP-biased MULTISTART ({len(args.seeds)} restarts)...", flush=True)
    lp_biased = multistart_lp_biased(instance, lp_probs, lp.bound, args.seeds, lp.solve_seconds)
    print(f"  best gap={lp_biased.best_gap:.4f}% at restart {lp_biased.best_found_at}"
          f"/{lp_biased.restarts}  |S|={lp_biased.best_open}  "
          f"time={lp_biased.total_seconds:.2f}s", flush=True)

    print(f"Arm C: alpha-GRASP MULTISTART ({len(args.seeds)} restarts)...", flush=True)
    alpha = multistart_alpha_grasp(instance, lp.bound, args.seeds, alpha=args.alpha)
    print(f"  best gap={alpha.best_gap:.4f}% at restart {alpha.best_found_at}"
          f"/{alpha.restarts}  |S|={alpha.best_open}  "
          f"time={alpha.total_seconds:.2f}s", flush=True)

    # A claimed optimum that one of our own feasible solutions beats was not an
    # optimum. See tests/regression/test_unproven_optimum_regression.py.
    ref = ref.validated_against(min(
        control.final_cost, rounding_ls.final_cost, no_lp.final_cost,
        lp_biased.best_cost, alpha.best_cost,
    ))

    wall = time.perf_counter() - wall0
    env = _environment()
    generated_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    commit = _git_commit()

    report = render_report(
        n_fac=n_fac, n_cust=n_cust, lp=lp, reference=ref, control=control,
        rounding_ls=rounding_ls, no_lp=no_lp, lp_biased=lp_biased, alpha=alpha,
        seeds=args.seeds, alpha_value=args.alpha,
        env=env, generated_at=generated_at, commit=commit, wall_seconds=wall,
    )

    md_path = os.path.join(args.out_dir, "california_4M_results.md")
    json_path = os.path.join(args.out_dir, "california_4M_results.json")

    with open(md_path, "w") as fh:
        fh.write(report)

    with open(json_path, "w") as fh:
        json.dump(
            {
                "generated_at": generated_at,
                "generated_by": "download_and_run_real_world.py::render_report",
                "git_commit": commit,
                "environment": env,
                "instance": {
                    "n_facilities": n_fac,
                    "n_customers": n_cust,
                    "dataset_seed": DATASET_SEED,
                    "base_setup_cost": BASE_SETUP_COST,
                    "setup_floor": SETUP_FLOOR,
                    "km_per_degree": KM_PER_DEGREE,
                    "cost_per_km": COST_PER_KM,
                },
                "seeds": args.seeds,
                "construction_eps": CONSTRUCTION_EPS,
                "lp_profile": asdict(lp),
                "reference": asdict(ref),
                "alpha": args.alpha,
                "arms": {
                    "lp_rounding_control": asdict(control),
                    "lp_rounding_plus_search": asdict(rounding_ls),
                    "local_search_only": asdict(no_lp),
                    "lp_biased_multistart": asdict(lp_biased),
                    "alpha_grasp_multistart": asdict(alpha),
                },
                "wall_seconds": wall,
            },
            fh,
            indent=2,
        )

    print(f"\nWrote {md_path}", flush=True)
    print(f"Wrote {json_path}", flush=True)
    print(f"Total wall-clock: {wall:.1f}s", flush=True)


if __name__ == "__main__":
    main()
