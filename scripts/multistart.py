"""
Multistart driver — the restart loop that makes a GRASP a GRASP.

WHY THIS EXISTS
    GRASP is a *multistart* procedure: randomization exists so that repeated
    restarts land in different basins and the best result is kept. Earlier
    revisions of this project performed exactly one construction followed by
    one local search, which left randomization with no mechanism through which
    to pay off. Measured against a deterministic ablation, it paid off with
    nothing.

    This module supplies the missing loop, so the randomized arms can be
    evaluated as the procedures they are named after rather than as
    single-shot constructions.

FAIRNESS
    A restart budget is a compute budget. Comparing best-of-N against a single
    deterministic run hands the randomized arm N times the work, so a raw
    best-of-N number on its own is not a fair comparison and this module does
    not report one on its own. Every `MultistartResult` carries:

      * `trajectory`  -- best-so-far after each restart, so quality can be read
                         at any budget rather than only at N;
      * `best_found_at` -- the restart that actually produced the winner, which
                         is the budget the arm really needed;
      * `restarts_to_reach(target)` -- how many restarts (and how much time) the
                         arm needs to match a given quality, which is what makes
                         a compute-matched comparison against a deterministic
                         arm possible.

    Deterministic arms are not run through this module. Restarting a
    deterministic construction reproduces the same solution, so its
    "best-of-N" is its best-of-1 at N times the cost.
"""

from __future__ import annotations

import math
import random
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Sequence

from scripts.run_experiments import run_local_search_iter_count
from scripts.uflp_solver import SolutionState, UFLPInstance


@dataclass
class MultistartResult:
    """Measurements from one multistart run of one randomized method."""

    method: str
    seeds: List[int]
    best_cost: float
    best_gap: float
    best_open: int
    best_found_at: int                      # 1-indexed restart producing the winner
    trajectory: List[float] = field(default_factory=list)      # best-so-far gap
    trajectory_costs: List[float] = field(default_factory=list)
    per_restart_gaps: List[float] = field(default_factory=list)
    total_iterations: int = 0
    moves: Dict[str, int] = field(default_factory=dict)
    construct_seconds: float = 0.0
    search_seconds: float = 0.0
    lp_seconds: float = 0.0

    @property
    def restarts(self) -> int:
        return len(self.seeds)

    @property
    def heuristic_seconds(self) -> float:
        """Time excluding any shared LP solve."""
        return self.construct_seconds + self.search_seconds

    @property
    def total_seconds(self) -> float:
        return self.lp_seconds + self.heuristic_seconds

    @property
    def seconds_per_restart(self) -> float:
        return self.heuristic_seconds / self.restarts if self.restarts else 0.0

    def restarts_to_reach(self, target_gap: float, tol: float = 1e-9) -> int | None:
        """
        The first restart whose best-so-far gap is at or below `target_gap`.

        Returns None if the budget never reached it. This is the metric that
        makes a compute-matched comparison possible: it answers "how many
        restarts does this arm need to match that one?" rather than "who wins
        when one side gets N times the work?".
        """
        for i, gap in enumerate(self.trajectory, start=1):
            if gap <= target_gap + tol:
                return i
        return None

    def seconds_to_reach(self, target_gap: float, tol: float = 1e-9) -> float | None:
        """Heuristic time needed to reach `target_gap`, at the mean cost per restart."""
        n = self.restarts_to_reach(target_gap, tol)
        if n is None:
            return None
        return self.lp_seconds + n * self.seconds_per_restart


def run_multistart(
    method: str,
    construct: Callable[[], SolutionState],
    instance: UFLPInstance,
    lp_bound: float,
    seeds: Sequence[int],
    lp_seconds: float = 0.0,
) -> MultistartResult:
    """
    Run `construct` + local search once per seed and keep the best solution.

    `construct` takes no arguments and reads the global `random` state, which
    this function reseeds before each restart, so restarts are independent and
    the whole run is reproducible from `seeds`.
    """
    if not seeds:
        raise ValueError("multistart needs at least one seed")

    best_cost = math.inf
    best_open = 0
    best_found_at = 0
    trajectory: List[float] = []
    trajectory_costs: List[float] = []
    per_restart_gaps: List[float] = []
    moves_total: Dict[str, int] = {"insert": 0, "delete": 0, "swap": 0}
    total_iterations = 0
    construct_seconds = 0.0
    search_seconds = 0.0

    def gap_of(cost: float) -> float:
        return (cost - lp_bound) / lp_bound * 100.0

    for index, seed in enumerate(seeds, start=1):
        random.seed(seed)

        t0 = time.perf_counter()
        state = construct()
        construct_seconds += time.perf_counter() - t0

        t0 = time.perf_counter()
        state, iterations, moves = run_local_search_iter_count(instance, state)
        search_seconds += time.perf_counter() - t0

        total_iterations += iterations
        for key, value in moves.items():
            moves_total[key] = moves_total.get(key, 0) + value

        per_restart_gaps.append(gap_of(state.total_cost))

        if state.total_cost < best_cost - 1e-9:
            best_cost = state.total_cost
            best_open = len(state.open_facilities)
            best_found_at = index

        trajectory.append(gap_of(best_cost))
        trajectory_costs.append(best_cost)

    return MultistartResult(
        method=method,
        seeds=list(seeds),
        best_cost=best_cost,
        best_gap=gap_of(best_cost),
        best_open=best_open,
        best_found_at=best_found_at,
        trajectory=trajectory,
        trajectory_costs=trajectory_costs,
        per_restart_gaps=per_restart_gaps,
        total_iterations=total_iterations,
        moves=moves_total,
        construct_seconds=construct_seconds,
        search_seconds=search_seconds,
        lp_seconds=lp_seconds,
    )


def multistart_lp_biased(
    instance: UFLPInstance,
    lp_probs: Dict[int, float],
    lp_bound: float,
    seeds: Sequence[int],
    lp_seconds: float = 0.0,
) -> MultistartResult:
    """Multistart LP-biased sampling: p(open f) = max(y_f, EPS)."""
    from scripts.run_experiments import construct_lp_biased_solution

    return run_multistart(
        method="LP-biased multistart",
        construct=lambda: construct_lp_biased_solution(instance, lp_probs),
        instance=instance,
        lp_bound=lp_bound,
        seeds=seeds,
        lp_seconds=lp_seconds,
    )


def multistart_alpha_grasp(
    instance: UFLPInstance,
    lp_bound: float,
    seeds: Sequence[int],
    alpha: float = 0.2,
) -> MultistartResult:
    """
    Multistart alpha-GRASP: savings-based restricted candidate list.

    This arm never reads the relaxation, so it is not charged for the LP solve.
    """
    from scripts.run_experiments import construct_alpha_grasp_solution

    return run_multistart(
        method=f"alpha-GRASP multistart (alpha={alpha})",
        construct=lambda: construct_alpha_grasp_solution(instance, alpha=alpha),
        instance=instance,
        lp_bound=lp_bound,
        seeds=seeds,
        lp_seconds=0.0,
    )


def multistart_random_alpha_grasp(
    instance: UFLPInstance,
    lp_bound: float,
    seeds: Sequence[int],
    choices: Sequence[float] = (0.05, 0.1, 0.2, 0.3, 0.5),
) -> MultistartResult:
    """
    Multistart alpha-GRASP drawing a fresh alpha per restart.

    A standard GRASP variant that sidesteps tuning: rather than committing to
    one alpha, each restart samples one. Included so the baseline is not
    dismissable as an untuned straw man.
    """
    from scripts.run_experiments import construct_alpha_grasp_solution

    def construct() -> SolutionState:
        # Drawn from the same reseeded stream, so the run stays reproducible.
        return construct_alpha_grasp_solution(instance, alpha=random.choice(list(choices)))

    return run_multistart(
        method="alpha-GRASP multistart (random alpha per restart)",
        construct=construct,
        instance=instance,
        lp_bound=lp_bound,
        seeds=seeds,
        lp_seconds=0.0,
    )
