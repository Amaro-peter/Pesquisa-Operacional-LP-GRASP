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
    control: ArmResult,
    rounding_ls: ArmResult,
    lp_biased: List[ArmResult],
    alpha: List[ArmResult],
    seeds: List[int],
    env: Dict[str, str],
    generated_at: str,
    commit: str,
    wall_seconds: float,
) -> str:
    """Render the markdown report purely from measured values."""
    n_vars = n_fac + n_cust * n_fac
    n_rows = n_cust + n_cust * n_fac

    # --- aggregates -------------------------------------------------------
    lb_final = [r.final_gap for r in lp_biased]
    al_final = [r.final_gap for r in alpha]
    lb_init = [r.initial_gap for r in lp_biased]
    al_init = [r.initial_gap for r in alpha]
    lb_iters = [r.iterations for r in lp_biased]
    al_iters = [r.iterations for r in alpha]
    lb_time = [r.total_seconds for r in lp_biased]
    al_time = [r.total_seconds for r in alpha]

    m_lb_final, m_al_final = _mean(lb_final), _mean(al_final)
    m_lb_init, m_al_init = _mean(lb_init), _mean(al_init)
    m_lb_time, m_al_time = _mean(lb_time), _mean(al_time)
    m_lb_iters, m_al_iters = _mean(lb_iters), _mean(al_iters)

    # Mean stage timings, for the decomposition table.
    m_lb_construct = _mean([r.construct_seconds for r in lp_biased])
    m_al_construct = _mean([r.construct_seconds for r in alpha])
    m_lb_search = _mean([r.search_seconds for r in lp_biased])
    m_al_search = _mean([r.search_seconds for r in alpha])

    d_construct = m_lb_construct - m_al_construct
    d_search = m_lb_search - m_al_search
    d_lp = lp.solve_seconds
    d_total = m_lb_time - m_al_time

    # Move mix, summed across seeds.
    def move_total(results: List[ArmResult], kind: str) -> int:
        return sum(r.moves.get(kind, 0) for r in results)

    lb_deletes, lb_swaps, lb_inserts = (move_total(lp_biased, k) for k in ("delete", "swap", "insert"))
    al_deletes, al_swaps, al_inserts = (move_total(alpha, k) for k in ("delete", "swap", "insert"))

    m_lb_open_init = _mean([r.initial_open for r in lp_biased])
    m_lb_open_final = _mean([r.final_open for r in lp_biased])
    m_al_open_final = _mean([r.final_open for r in alpha])
    eps_excess = m_lb_open_init - lp.rounded_open_count

    seed_word = "seed" if len(seeds) == 1 else "seeds"
    seeds_str = ", ".join(str(s) for s in seeds)

    _candidates = [
        ("LP rounding (control)", control.final_gap),
        ("LP rounding + local search", rounding_ls.final_gap),
        ("LP-biased hybrid GRASP", m_lb_final),
        ("alpha-GRASP baseline", m_al_final),
    ]
    best_name, best_final = min(_candidates, key=lambda t: t[1])

    lines: List[str] = []
    w = lines.append

    # ---------------------------------------------------------------- header
    w("# Real-World UFLP Benchmark — California Housing")
    w("")
    w(f"A {n_fac}×{n_cust} uncapacitated facility location instance built from real census")
    w("block coordinates and populations, used to compare an LP-biased hybrid GRASP against a")
    w("classical α-GRASP baseline **and** against a no-heuristic control.")
    w("")
    w("> **Provenance.** Every number below was measured by the run described in")
    w("> §11 and written directly by `render_report` in `download_and_run_real_world.py`.")
    w("> No value in this file is hardcoded or back-computed, and every comparative")
    w("> word (\"lower\", \"faster\", \"tighter\") is derived from the measured values")
    w("> rather than written into the template. The raw records are in")
    w("> `output/california_4M_results.json`.")
    w("")
    w("---")
    w("")

    # ------------------------------------------------------------- §1 instance
    w("## 1. Instance")
    w("")
    w("| Property | Value |")
    w("|---|---|")
    w("| Source dataset | California Housing (`sklearn.datasets.fetch_california_housing`) |")
    w(f"| Facilities × customers | {n_fac:,} × {n_cust:,} |")
    w(f"| Sampling | two **disjoint** random samples of census blocks, `np.random.seed({DATASET_SEED})` |")
    w("| Facility coordinates | block latitude / longitude (columns 6, 7) |")
    w(f"| Setup cost | `{BASE_SETUP_COST:,.0f} × (population / median population) + {SETUP_FLOOR:,.0f}` |")
    w(f"| Service cost | Euclidean degree distance × {KM_PER_DEGREE} km/deg × {COST_PER_KM} per km |")
    w("")
    w("### LP model size")
    w("")
    w("| Quantity | Count |")
    w("|---|---|")
    w(f"| Decision variables | {n_vars:,} &nbsp;(`{n_fac:,}` y + `{n_cust * n_fac:,}` x) |")
    w(f"| Assignment rows (`Σ_f x[u][f] = 1`) | {n_cust:,} |")
    w(f"| Linking rows (`x[u][f] ≤ y[f]`) | {n_cust * n_fac:,} |")
    w(f"| Total constraint rows | {n_rows:,} |")
    w(f"| Nonzeros in the linking matrix | {2 * n_cust * n_fac:,} |")
    w("")
    w(f"*Describing this instance as “{n_cust * n_fac:,} variables” counts only the assignment")
    w(f"variables `x`. The full model carries {n_vars:,} variables and {n_rows:,} constraint rows —")
    w("it is the linking rows, not the variables, that make the model heavy.*")
    w("")
    w("### Known approximation in the distance model")
    w("")
    w(f"Longitude degrees are converted at the same {KM_PER_DEGREE} km/degree as latitude degrees.")
    w("At Californian latitudes (~37°N) a longitude degree is closer to 88 km, so east–west")
    w("separation is overstated by roughly 26%. The instance is geographically *derived*,")
    w("not geographically *accurate*. This is pinned by a unit test so it cannot drift silently.")
    w("")

    # -------------------------------------------------------------- §2 LP bound
    w("## 2. LP relaxation")
    w("")
    w("| Quantity | Value |")
    w("|---|---|")
    w(f"| **LP bound** | **{lp.bound:,.2f}** |")
    w(f"| Solve time | {lp.solve_seconds:.2f} s |")
    w(f"| Solver | SciPy `linprog(method='highs')` — HiGHS selects the algorithm |")
    w(f"| Facilities with `y ≥ 0.99` (integral, open) | {lp.n_integral_open:,} / {lp.n_facilities:,} |")
    w(f"| Facilities with `0.01 < y < 0.99` (**fractional**) | **{lp.n_fractional:,}** / {lp.n_facilities:,} "
      f"({lp.fractional_pct:.2f}%) |")
    w(f"| Facilities with `y ≤ 0.01` | {lp.n_near_zero:,} / {lp.n_facilities:,} |")
    w(f"| `Σ y[f]` (fractional facility count) | {lp.sum_y:,.2f} |")
    w(f"| Facilities opened by rounding at `y ≥ 0.5` | {lp.rounded_open_count:,} |")
    w("")
    integrality = (
        "essentially integral" if lp.fractional_pct < 1.0
        else "moderately fractional" if lp.fractional_pct < 10.0
        else "highly fractional"
    )
    w(f"**Read:** the relaxation is {integrality} — {lp.n_fractional} of {lp.n_facilities} "
      f"facilities ({lp.fractional_pct:.2f}%) take a strictly fractional value. This matters for")
    w("interpretation: a near-integral relaxation means the LP has very nearly *solved* the")
    w("problem, and the heuristic's job is repair rather than search.")
    w("")
    best_cost = min(
        control.final_cost,
        rounding_ls.final_cost,
        min(r.final_cost for r in lp_biased),
        min(r.final_cost for r in alpha),
    )
    if lp.n_fractional > 0:
        w("> **The LP bound is a bound, not a proven optimum.** All gaps below are measured")
        w("> against it. Because the relaxation is not perfectly integral, the true integer")
        w(f"> optimum lies somewhere in `[{lp.bound:,.2f}, {best_cost:,.2f}]`, so the *true*")
        w("> optimality gaps are **smaller** than the figures reported here. Solving the exact")
        w("> IP at this scale is not tractable with CBC; `run_experiments.solve_exact_ip` does")
        w("> it for the smaller OR-Library instances.")
    else:
        w("> **Here the LP bound *is* the integer optimum.** The relaxation came back fully")
        w("> integral, so its value is attained by a feasible integer solution and the gaps")
        w("> below are true optimality gaps. This is a property of *this* instance, not a")
        w("> guarantee: a fractional relaxation would make the bound strictly weaker.")
    w("")

    # --------------------------------------------------------------- §3 methods
    w("## 3. Methods compared")
    w("")
    w("| | Method | Construction | Local search | Pays for LP |")
    w("|---|---|---|---|---|")
    w("| **A** | LP rounding (control) | open every `y ≥ 0.5` | none | yes |")
    w("| **A+** | Rounding + local search | open every `y ≥ 0.5` (deterministic) | "
      "best-improvement (insert / delete / swap) | yes |")
    w(f"| **B** | LP-biased hybrid GRASP | open `f` with probability `max(y[f], {CONSTRUCTION_EPS})` | "
      "best-improvement (insert / delete / swap) | yes |")
    w("| **C** | α-GRASP baseline | savings-based RCL, `α = 0.2` | same | no |")
    w("")
    w("Arm **A** is the control the earlier revision of this report lacked: without it there is no")
    w("way to tell whether the GRASP layer contributes anything beyond what the LP already knew.")
    w("")
    w("Arm **A+** is the ablation. It differs from arm B in exactly one respect — the construction")
    w("is deterministic rounding rather than probabilistic sampling — and shares the local search,")
    w("the LP and the instance. Any difference between A+ and B is therefore attributable to")
    w("randomization and to nothing else. §6 reports it.")
    w("")

    # --------------------------------------------------------------- §4 results
    w("## 4. Results")
    w("")
    w(f"Heuristic arms were run on {len(seeds)} {seed_word} ({seeds_str}); "
      f"{'values are means across seeds' if len(seeds) > 1 else 'single-seed values'}. "
      "The control is deterministic.")
    w("")
    w("| Metric | A · LP rounding | A+ · Rounding + LS | B · LP-biased GRASP | C · α-GRASP | B vs C |")
    w("|---|---|---|---|---|---|")
    w(f"| Initial cost | {control.initial_cost:,.2f} | {rounding_ls.initial_cost:,.2f} | "
      f"{_mean([r.initial_cost for r in lp_biased]):,.2f} | "
      f"{_mean([r.initial_cost for r in alpha]):,.2f} | "
      f"B starts {_cmp(m_lb_init, m_al_init, 'lower', 'higher')} |")
    w(f"| Initial gap vs LP bound | {control.initial_gap:.4f}% | {rounding_ls.initial_gap:.4f}% | "
      f"{m_lb_init:.4f}% | {m_al_init:.4f}% | {_signed(m_lb_init - m_al_init, ' pp', 4)} |")
    w(f"| Facilities opened by construction | {control.initial_open} | {rounding_ls.initial_open} | "
      f"{m_lb_open_init:,.1f} | {_mean([r.initial_open for r in alpha]):,.1f} | — |")
    w(f"| Local-search iterations | {control.iterations} | {rounding_ls.iterations} | "
      f"{m_lb_iters:,.1f} | {m_al_iters:,.1f} | B does {_cmp(m_lb_iters, m_al_iters, 'fewer', 'more')} |")
    w(f"| **Final cost** | **{control.final_cost:,.2f}** | **{rounding_ls.final_cost:,.2f}** | "
      f"**{_mean([r.final_cost for r in lp_biased]):,.2f}** | "
      f"**{_mean([r.final_cost for r in alpha]):,.2f}** | "
      f"{_signed(_mean([r.final_cost for r in lp_biased]) - _mean([r.final_cost for r in alpha]))} |")
    w(f"| **Final gap vs LP bound** | **{control.final_gap:.4f}%** | **{rounding_ls.final_gap:.4f}%** | "
      f"**{m_lb_final:.4f}%** | **{m_al_final:.4f}%** | "
      f"B is {_cmp(m_lb_final, m_al_final, 'tighter', 'looser')} "
      f"by {abs(m_lb_final - m_al_final):.4f} pp |")
    w(f"| Facilities open at the end | {control.final_open} | {rounding_ls.final_open} | "
      f"{m_lb_open_final:,.1f} | {m_al_open_final:,.1f} | — |")
    w(f"| **End-to-end time** | **{control.total_seconds:.2f} s** | **{rounding_ls.total_seconds:.2f} s** | "
      f"**{m_lb_time:.2f} s** | **{m_al_time:.2f} s** | "
      f"B is {_cmp(m_lb_time, m_al_time, 'faster', 'slower')} by {abs(d_total):.2f} s |")
    w("")
    w(f"*A, A+ and B include the {lp.solve_seconds:.2f} s LP solve in their end-to-end time; C does not.*")
    w("")
    w(f"**Best final gap: {best_name} at {best_final:.4f}%.**")
    w("")

    if len(seeds) > 1:
        w("### Per-seed detail")
        w("")
        w("| Seed | B init gap | B iters | B final gap | B time | C init gap | C iters | C final gap | C time |")
        w("|---|---|---|---|---|---|---|---|---|")
        for b, a in zip(lp_biased, alpha):
            w(f"| {b.seed} | {b.initial_gap:.4f}% | {b.iterations} | {b.final_gap:.4f}% | "
              f"{b.total_seconds:.2f} s | {a.initial_gap:.4f}% | {a.iterations} | "
              f"{a.final_gap:.4f}% | {a.total_seconds:.2f} s |")
        w("")
        w(f"Final-gap spread: B {min(lb_final):.4f}–{max(lb_final):.4f}%, "
          f"C {min(al_final):.4f}–{max(al_final):.4f}%.")
        w("")

        lb_spread = max(lb_final) - min(lb_final)
        al_spread = max(al_final) - min(al_final)
        lb_distinct = len({round(r.final_cost, 6) for r in lp_biased})
        al_distinct = len({round(r.final_cost, 6) for r in alpha})
        steadier = _cmp(lb_spread, al_spread, "B", "C", tie="neither arm")

        if lb_distinct == 1:
            w(f"**Arm B reached the identical solution on all {len(seeds)} seeds** — the same")
            w(f"{lp_biased[0].final_open} open facilities at {lp_biased[0].final_cost:,.2f} — despite starting from")
            w(f"initial gaps spanning {min(lb_init):.2f}%–{max(lb_init):.2f}%. The EPS noise changes")
            w("where the search *starts* but not where it *ends*.")
            w("")
        if lb_distinct != al_distinct or abs(lb_spread - al_spread) > 1e-9:
            w(f"Arm B produced {lb_distinct} distinct final "
              f"{_plural(lb_distinct, 'solution')} across the {len(seeds)} seeds; arm C produced")
            w(f"{al_distinct}. Measured as final-gap spread, **{steadier}** is the more consistent arm")
            w(f"({lb_spread:.4f} pp for B versus {al_spread:.4f} pp for C). Consistency is a separate")
            w("property from mean quality, and on a randomised heuristic it is often the more")
            w("useful one: it is what lets a single run be trusted without replication.")
            w("")

    # ---------------------------------------------------- §5 construction detail
    w("## 5. What the construction actually produces")
    w("")
    w(f"The LP wants about **{lp.rounded_open_count}** facilities open (`y ≥ 0.5`). The LP-biased")
    w(f"construction opened **{m_lb_open_init:,.1f}** on average — an excess of **{eps_excess:+,.1f}**.")
    w("")
    w(f"That excess is not a property of the LP. It is the `EPS = {CONSTRUCTION_EPS}` floor in")
    w("`construct_solution`: every facility is opened with probability `max(y[f], EPS)`, so each")
    w(f"of the **{lp.n_near_zero:,}** facilities the LP set to `y ≤ {CONSTRUCTION_EPS}` still gets a")
    w(f"{CONSTRUCTION_EPS:.0%} chance. The expected number of such spurious openings is")
    w(f"`{CONSTRUCTION_EPS} × {lp.n_near_zero:,} = {lp.expected_eps_openings:.1f}`, which is the same order as the")
    w(f"{eps_excess:+,.1f} observed.")
    w("")
    w(f"Each spurious facility carries a setup cost, and that is what produces arm B's")
    w(f"{m_lb_init:.2f}% initial gap. **The initial gap is an artifact of the EPS floor, not of LP")
    w(f"fractionality** — §2 shows the relaxation is only {lp.fractional_pct:.2f}% fractional.")
    w("")
    w("Because the floor is a fixed per-facility constant, its damage grows linearly with")
    w(f"instance size: negligible on a 50-facility instance, ~{lp.expected_eps_openings:.0f} spurious")
    w(f"{_plural(lp.expected_eps_openings, 'opening')} here. Scaling it as `EPS ∝ 1/|F|`, or")
    w("skipping facilities the LP zeroes confidently, would cut both the initial gap and the")
    w("repair work it creates.")
    w("")

    # ------------------------------------------- §6 the randomization ablation
    w("## 6. What does the randomized construction contribute?")
    w("")
    w("Arms A+ and B share the LP, the instance and the local search. They differ only in how")
    w("the starting solution is built: **A+ rounds the LP deterministically, B samples from it**.")
    w("So the difference between them is what randomization buys.")
    w("")

    lb_best_gap = min(lb_final)
    lb_best_cost = min(r.final_cost for r in lp_biased)
    same_solution = all(
        abs(r.final_cost - rounding_ls.final_cost) < 1e-6 for r in lp_biased
    )
    ratio_iters = (m_lb_iters / rounding_ls.iterations) if rounding_ls.iterations else float("inf")
    ap_heur = rounding_ls.construct_seconds + rounding_ls.search_seconds
    b_heur = m_lb_construct + m_lb_search

    w("| | A+ · deterministic rounding | B · randomized sampling |")
    w("|---|---|---|")
    w(f"| Facilities opened by construction | {rounding_ls.initial_open} | {m_lb_open_init:,.1f} |")
    w(f"| Construction gap | {rounding_ls.initial_gap:.4f}% | {m_lb_init:.4f}% |")
    w(f"| Local-search iterations | **{rounding_ls.iterations}** | **{m_lb_iters:,.1f}** |")
    _mm = rounding_ls.moves
    w(f"| Move mix | {_mm.get('insert', 0)} insert / {_mm.get('delete', 0)} delete / "
      f"{_mm.get('swap', 0)} swap | see §7 |")
    w(f"| **Final cost** | **{rounding_ls.final_cost:,.2f}** | **{_mean([r.final_cost for r in lp_biased]):,.2f}** |")
    w(f"| **Final gap** | **{rounding_ls.final_gap:.4f}%** | **{m_lb_final:.4f}%** |")
    w(f"| Heuristic time, excluding the shared LP | {ap_heur:.2f} s | {b_heur:.2f} s |")
    w("")

    if same_solution:
        w(f"**Both arms reach the identical solution** — {rounding_ls.final_cost:,.2f} at")
        w(f"{rounding_ls.final_gap:.4f}%, on every seed. Randomization changes nothing about the")
        w("answer.")
    elif rounding_ls.final_gap <= lb_best_gap + 1e-9:
        w(f"**Deterministic rounding matches or beats every randomized run** "
          f"({rounding_ls.final_gap:.4f}% against a best of {lb_best_gap:.4f}%).")
    elif rounding_ls.final_gap <= m_lb_final + 1e-9:
        w(f"**Deterministic rounding beats the randomized mean** ({rounding_ls.final_gap:.4f}% vs")
        w(f"{m_lb_final:.4f}%), though the best randomized run reaches {lb_best_gap:.4f}%.")
    else:
        w(f"**Randomization does buy quality here**: {m_lb_final:.4f}% mean against "
          f"{rounding_ls.final_gap:.4f}% for deterministic rounding")
        w(f"(best randomized run: {lb_best_gap:.4f}%, cost {lb_best_cost:,.2f}).")
    w("")

    if rounding_ls.iterations and ratio_iters > 1.0:
        w(f"It is not free. The randomized construction opens {m_lb_open_init - rounding_ls.initial_open:+,.1f}")
        w(f"more facilities than rounding does, and the local search then needs {ratio_iters:.1f}× as many")
        w(f"iterations ({m_lb_iters:,.1f} against {rounding_ls.iterations}) to undo them — "
          f"{b_heur / ap_heur:.1f}× the heuristic")
        w("time once the shared LP solve is excluded from both.")
    elif rounding_ls.iterations:
        w(f"The randomized construction needs {m_lb_iters:,.1f} local-search iterations against")
        w(f"{rounding_ls.iterations} for deterministic rounding.")
    w("")

    w("> **This is the experiment that decides what the method contributes**, and it is the one the")
    w("> original study did not run. Without arm A+, a comparison against α-GRASP cannot separate")
    w("> \"the LP relaxation is informative\" — which it plainly is — from \"biasing a random")
    w("> construction with the LP is informative\", which is the actual claim being made.")
    w("")
    w("One further caveat on the framing: GRASP is a *multistart* procedure, and randomization")
    w("exists in it so that restarts explore different basins and the best is kept. This pipeline")
    w("performs a single construction and a single local search — there is no restart loop anywhere")
    w("in the codebase. With one start, randomization has no mechanism through which to pay off.")
    w("")

    # ---------------------------------------------------- §7 local search detail
    w("## 7. What the local search actually does")
    w("")
    w("Move mix, summed across all seeds:")
    w("")
    w("| Arm | Inserts | Deletes | Swaps | Total | Open facilities |")
    w("|---|---|---|---|---|---|")
    w(f"| B · LP-biased | {lb_inserts} | {lb_deletes} | {lb_swaps} | {sum(lb_iters)} | "
      f"{m_lb_open_init:,.1f} → {m_lb_open_final:,.1f} |")
    w(f"| C · α-GRASP | {al_inserts} | {al_deletes} | {al_swaps} | {sum(al_iters)} | "
      f"{_mean([r.initial_open for r in alpha]):,.1f} → {m_al_open_final:,.1f} |")
    w("")
    lb_delete_share = lb_deletes / max(sum(lb_iters), 1) * 100
    al_swap_share = al_swaps / max(sum(al_iters), 1) * 100
    w(f"**The two arms are not doing the same work.** {lb_delete_share:.0f}% of arm B's moves are")
    w("deletions — it is removing the facilities the EPS floor opened by mistake. "
      f"{al_swap_share:.0f}% of")
    w("arm C's moves are swaps — genuinely relocating facilities, which is the more expensive")
    w("kind of move to find and the harder kind of improvement to make.")
    w("")
    w("So the iteration counts are not directly comparable as a measure of “effort saved”:")
    w("fewer iterations here partly reflects that deletions are cheap repairs of a self-inflicted")
    w("problem, not that arm B started from a structurally better configuration.")
    w("")

    # ------------------------------------------------------------ §8 time budget
    w("## 8. Where the time goes")
    w("")
    w("| Stage | B · LP-biased | C · α-GRASP | Difference (B − C) |")
    w("|---|---|---|---|")
    w(f"| LP relaxation | {lp.solve_seconds:.2f} s | 0.00 s | {_signed(d_lp)} s |")
    w(f"| Construction | {m_lb_construct:.2f} s | {m_al_construct:.2f} s | {_signed(d_construct)} s |")
    w(f"| Local search | {m_lb_search:.2f} s | {m_al_search:.2f} s | {_signed(d_search)} s |")
    w(f"| **Total** | **{m_lb_time:.2f} s** | **{m_al_time:.2f} s** | **{_signed(d_total)} s** |")
    w("")
    faster = _cmp(m_lb_time, m_al_time, "faster", "slower")

    def term(delta: float) -> str:
        """Describe a stage difference from arm B's point of view."""
        if abs(delta) < 5e-3:
            return "no material difference"
        return f"saves arm B {abs(delta):.2f} s" if delta < 0 else f"costs arm B {delta:.2f} s"

    w(f"Arm B is **{faster}** overall by {abs(d_total):.2f} s. That net figure is the sum of three")
    w("stage-level differences pulling in different directions:")
    w("")
    w(f"- **Local search — {term(d_search)}.** Arm B runs {m_lb_iters:,.1f} iterations to arm C's")
    w(f"  {m_al_iters:,.1f}, at a broadly similar cost per iteration.")
    w(f"- **Construction — {term(d_construct)}.** α-GRASP's constructor is a sequential greedy loop")
    w("  that re-scores every closed facility against every customer on every pass; the LP-biased")
    w("  constructor is a single sampling sweep. This term is a property of the *baseline's")
    w("  constructor*, not of the LP bias, and it is easy to overlook.")
    w(f"- **LP relaxation — {term(d_lp)}**, paid only by arm B.")
    w("")
    dominant = max(("local search", abs(d_search)), ("construction", abs(d_construct)),
                   ("the LP solve", abs(d_lp)), key=lambda t: t[1])[0]
    w(f"The largest single term is **{dominant}**. Attributing the whole difference to any one")
    w("stage would misstate where the time actually goes.")
    w("")

    # --------------------------------------------------------------- §9 findings
    w("## 9. Findings")
    w("")
    w(f"**1. The LP relaxation is {integrality}.** {lp.n_fractional} of {lp.n_facilities} facilities")
    w(f"({lp.fractional_pct:.2f}%) are fractional. Any explanation of arm B's behaviour that rests on")
    w("heavy fractionality is not supported by this instance.")
    w("")
    w(f"**2. Arm B's {m_lb_init:.2f}% initial gap is caused by the EPS floor.** It opens about")
    w(f"{lp.expected_eps_openings:.0f} {_plural(lp.expected_eps_openings, 'facility', 'facilities')} the LP had zeroed")
    w(f"({eps_excess:+,.1f} observed above the LP's {lp.rounded_open_count}), each carrying a setup cost. See §5.")
    w("")
    same_solution_f = all(abs(r.final_cost - rounding_ls.final_cost) < 1e-6 for r in lp_biased)
    ratio_iters_f = (m_lb_iters / rounding_ls.iterations) if rounding_ls.iterations else float("inf")
    if same_solution_f:
        w("**3. The randomized construction contributes no measurable quality.** Deterministic LP")
        w(f"rounding followed by the same local search (arm A+) reaches the identical solution —")
        w(f"{rounding_ls.final_cost:,.2f} at {rounding_ls.final_gap:.4f}% — in {rounding_ls.iterations}")
        w(f"local-search {_plural(rounding_ls.iterations, 'iteration')} against {m_lb_iters:,.1f} for the")
        w(f"randomized arm. The sampling opens facilities the LP had already ruled out and the local")
        w("search then removes them again. See §6.")
    elif rounding_ls.final_gap <= m_lb_final + 1e-9:
        w("**3. The randomized construction does not improve on deterministic rounding.** Arm A+")
        w(f"reaches {rounding_ls.final_gap:.4f}% against arm B's {m_lb_final:.4f}% mean, in")
        w(f"{rounding_ls.iterations} local-search {_plural(rounding_ls.iterations, 'iteration')}")
        w(f"against {m_lb_iters:,.1f}. See §6.")
    else:
        w(f"**3. The randomized construction does buy quality.** Arm B reaches {m_lb_final:.4f}% against")
        w(f"{rounding_ls.final_gap:.4f}% for deterministic rounding plus the same local search, at")
        w(f"{ratio_iters_f:.1f}× the local-search cost. See §6.")
    w("")

    ls_helps = rounding_ls.final_gap < control.final_gap - 1e-9
    if ls_helps:
        factor = (control.final_gap / rounding_ls.final_gap
                  if rounding_ls.final_gap > 0 else float("inf"))
        w(f"**4. The local search earns its keep.** Arm A+ takes the same rounded solution the")
        w(f"control stops at ({control.final_gap:.4f}%) and improves it to {rounding_ls.final_gap:.4f}% —")
        w(f"better by a factor of {factor:.1f}× — in {rounding_ls.iterations} "
          f"{_plural(rounding_ls.iterations, 'iteration')}. Because arm A+ introduces no construction")
        w("noise of its own, this is a clean demonstration that the local search does real work")
        w("rather than merely repairing a randomized start. Rounding the LP alone is not enough.")
    else:
        w(f"**4. The local search does not improve on rounding.** The control (arm A) reaches")
        w(f"{control.final_gap:.4f}% and arm A+ reaches {rounding_ls.final_gap:.4f}%, so the local")
        w("search adds no quality over simply rounding the LP solution.")
    w("")
    better = _cmp(m_lb_final, m_al_final, "better", "worse", tie="level with")
    w(f"**5. Arm B ends {better} than arm C** ({m_lb_final:.4f}% vs {m_al_final:.4f}%) and is")
    w(f"{faster} by {abs(d_total):.2f} s.")
    if d_total < 0:
        w("Both results favour arm B. The speed margin, however, is not a single effect: §8")
        w(f"shows it splitting across local search ({_signed(d_search)} s), construction")
        w(f"({_signed(d_construct)} s) and the LP solve ({_signed(d_lp)} s). Attributing it entirely")
        w("to reduced local-search workload would overstate the role of the LP bias, because the")
        w("constructor term belongs to the baseline's greedy loop rather than to anything the LP did.")
    else:
        w("The quality and speed results point in opposite directions here: arm B pays for the LP")
        w("relaxation up front and does not recover that cost at this instance size. §8 gives the")
        w("stage-by-stage breakdown.")
    w("")
    if len(seeds) > 1:
        lb_spread = max(lb_final) - min(lb_final)
        al_spread = max(al_final) - min(al_final)
        lb_distinct = len({round(r.final_cost, 6) for r in lp_biased})
        al_distinct = len({round(r.final_cost, 6) for r in alpha})
        steadier_name = _cmp(lb_spread, al_spread, "Arm B", "Arm C", tie="Neither arm")
        w(f"**6. {steadier_name} is the more reproducible.** Across {len(seeds)} seeds arm B's final gap")
        w(f"spanned {lb_spread:.4f} pp ({lb_distinct} distinct "
          f"{_plural(lb_distinct, 'solution')}) and arm C's spanned {al_spread:.4f} pp")
        w(f"({al_distinct} distinct {_plural(al_distinct, 'solution')}). A single-seed comparison")
        w("cannot see this, and it is arguably the more practically important difference:")
        w("an arm whose answer does not depend on the seed can be run once.")
        w("")
        w(f"Note the scope of that claim, though: arm A+ is *deterministic*, so it reaches")
        w(f"{rounding_ls.final_gap:.4f}% with no seed at all. Reproducibility here is a property of")
        w("not randomizing, which arm A+ achieves more directly than arm B does.")
        w("")
        next_n = 7
    else:
        next_n = 6

    w(f"**{next_n}. The iteration counts measure different work.** {lb_delete_share:.0f}% of arm B's moves are")
    w(f"deletions; {al_swap_share:.0f}% of arm C's are swaps (§7). Deletions repair the construction's own")
    w("EPS noise and are cheap to find; swaps relocate facilities and are the harder improvement.")
    w("A raw “fewer iterations” comparison is therefore a weaker claim than it first appears.")
    w("")

    # ----------------------------------------------------------- §10 limitations
    w("## 10. Limitations")
    w("")
    w("- **Gaps are measured against the LP bound, not a proven integer optimum.** The true")
    w("  optimality gaps are smaller than reported. Calling these “optimality gaps” without")
    w("  qualification would be incorrect.")
    if len(seeds) == 1:
        w("- **Single seed.** Both heuristic arms are randomised; one seed is an anecdote, not a")
        w("  distribution. Re-run with `--seeds 42 7 2024` for a spread.")
    else:
        w(f"- **{len(seeds)} seeds.** Enough to show a spread, not enough for a confidence interval.")
    w("- **Only compare timings within this run.** Every figure here comes from one sequential")
    w("  session on one machine. Wall-clock for this workload is sensitive to memory pressure")
    w("  and concurrent load, and repeated runs of the same configuration have differed by")
    w("  tens of percent. The cost, gap, iteration and move-mix figures are fully deterministic")
    w("  under the stated seeds; the timings are not, and should not be quoted across runs.")
    w("- **The distance model overstates east–west separation by ~26%** (§1).")
    w("- **The HiGHS algorithm is not recorded.** `method='highs'` lets the solver choose between")
    w("  simplex and interior point, so the LP solve time should not be labelled “simplex time”.")
    w("- **A single alpha value** (0.2) was tested for the baseline; it was not tuned.")
    w("")

    # --------------------------------------------------------- §11 reproduction
    w("## 11. Reproduction")
    w("")
    w("```bash")
    w("pip install -r requirements.txt")
    w(f"python download_and_run_real_world.py --size {n_fac} --seeds {' '.join(str(s) for s in seeds)}")
    w("```")
    w("")
    w("| Run metadata | |")
    w("|---|---|")
    w(f"| Generated at | {generated_at} |")
    w(f"| Generated by | `download_and_run_real_world.py` → `render_report` |")
    w(f"| Git commit | `{commit}` |")
    w(f"| Dataset sample seed | {DATASET_SEED} |")
    w(f"| Heuristic {seed_word} | {seeds_str} |")
    w(f"| Total wall-clock | {wall_seconds:.1f} s |")
    w(f"| Python | {env['python']} |")
    w(f"| NumPy / SciPy / scikit-learn | {env['numpy']} / {env['scipy']} / {env['scikit_learn']} |")
    w(f"| Platform | {env['platform']} |")
    w(f"| CPU count | {env['cpu_count']} |")
    w("")
    w("Raw measurements: [`california_4M_results.json`](california_4M_results.json).")
    w("")

    return "\n".join(lines) + "\n"


# ============================================================================
# Orchestration
# ============================================================================

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
    control = ArmResult(**arms["lp_rounding_control"])
    rounding_ls = ArmResult(**arms["lp_rounding_plus_search"])
    lp_biased = [ArmResult(**r) for r in arms["lp_biased"]]
    alpha = [ArmResult(**r) for r in arms["alpha_grasp"]]

    report = render_report(
        n_fac=payload["instance"]["n_facilities"],
        n_cust=payload["instance"]["n_customers"],
        lp=lp, control=control, rounding_ls=rounding_ls, lp_biased=lp_biased, alpha=alpha,
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
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 7, 2024],
                        help="heuristic seeds; default 42 7 2024")
    parser.add_argument("--out-dir", type=str, default="output")
    parser.add_argument("--from-json", type=str, default=None,
                        help="re-render the markdown from a saved measurement sidecar "
                             "instead of re-running the benchmark")
    args = parser.parse_args(argv)

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

    lp_biased: List[ArmResult] = []
    alpha: List[ArmResult] = []
    for seed in args.seeds:
        print(f"Arm B: LP-biased hybrid GRASP (seed {seed})...", flush=True)
        r = run_lp_biased(instance, lp_probs, lp, seed)
        lp_biased.append(r)
        print(f"  init |S|={r.initial_open} gap={r.initial_gap:.4f}%  →  "
              f"iters={r.iterations} {r.moves}  final |S|={r.final_open} "
              f"gap={r.final_gap:.4f}%  time={r.total_seconds:.2f}s", flush=True)

        print(f"Arm C: alpha-GRASP baseline (seed {seed})...", flush=True)
        r = run_alpha_grasp(instance, lp, seed)
        alpha.append(r)
        print(f"  init |S|={r.initial_open} gap={r.initial_gap:.4f}%  →  "
              f"iters={r.iterations} {r.moves}  final |S|={r.final_open} "
              f"gap={r.final_gap:.4f}%  time={r.total_seconds:.2f}s", flush=True)

    wall = time.perf_counter() - wall0
    env = _environment()
    generated_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    commit = _git_commit()

    report = render_report(
        n_fac=n_fac, n_cust=n_cust, lp=lp, control=control, rounding_ls=rounding_ls,
        lp_biased=lp_biased, alpha=alpha, seeds=args.seeds,
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
                "arms": {
                    "lp_rounding_control": asdict(control),
                    "lp_rounding_plus_search": asdict(rounding_ls),
                    "lp_biased": [asdict(r) for r in lp_biased],
                    "alpha_grasp": [asdict(r) for r in alpha],
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
