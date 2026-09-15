"""
Unit tests for the multistart driver.

The restart loop is what makes a GRASP a GRASP, and it is also what makes a
best-of-N number easy to misread. These tests pin both halves:

  * the loop itself -- best kept across restarts, trajectory monotone,
    reproducibility from seeds, work accumulated across restarts;
  * the fairness accessors -- `best_found_at`, `restarts_to_reach` and
    `seconds_to_reach`, which exist so a best-of-N result can be compared
    against a deterministic single run at matched compute rather than at
    matched restart count.
"""

import math
import random

import pytest

from scripts.multistart import (
    MultistartResult,
    multistart_alpha_grasp,
    multistart_lp_biased,
    multistart_random_alpha_grasp,
    run_multistart,
)
from scripts.uflp_solver import (
    SolutionState,
    UFLPInstance,
    _compute_auxiliary_data,
    _compute_total_cost,
    _find_closest_two,
)


@pytest.fixture
def instance() -> UFLPInstance:
    """Four co-located customer/facility pairs; the optimum opens all four."""
    n = 4
    facilities = customers = list(range(n))
    setup = {f: 1.0 for f in facilities}
    service = {
        u: {f: (0.0 if u == f else 500.0) for f in facilities} for u in customers
    }
    return UFLPInstance(facilities, customers, setup, service)


def _state_for(instance, open_facilities) -> SolutionState:
    state = SolutionState(open_facilities=set(open_facilities))
    for u in instance.customers:
        c1, c2 = _find_closest_two(u, state.open_facilities, instance.service_costs)
        state.closest_facility[u] = c1
        state.second_closest_facility[u] = c2
    state.total_cost = _compute_total_cost(instance, state)
    _compute_auxiliary_data(instance, state)
    return state


class TestTheLoopKeepsTheBest:
    def test_best_cost_is_the_minimum_over_restarts(self, instance):
        """A scripted sequence of starts: the best must survive to the end."""
        starts = [{0}, {0, 1}, {0, 1, 2, 3}, {1}]
        it = iter(starts)

        result = run_multistart(
            method="scripted",
            construct=lambda: _state_for(instance, next(it)),
            instance=instance,
            lp_bound=4.0,                     # optimum is 4 * 1.0 setup + 0 service
            seeds=[1, 2, 3, 4],
        )
        assert result.restarts == 4
        assert math.isclose(result.best_cost, 4.0, abs_tol=1e-9)
        assert result.best_open == 4
        assert math.isclose(result.best_gap, 0.0, abs_tol=1e-9)

    def test_trajectory_is_monotone_non_increasing(self, instance):
        result = multistart_alpha_grasp(instance, lp_bound=4.0, seeds=list(range(1, 9)))
        assert len(result.trajectory) == 8
        assert all(a >= b - 1e-12 for a, b in zip(result.trajectory, result.trajectory[1:])), (
            "best-so-far must never get worse"
        )
        assert math.isclose(result.trajectory[-1], result.best_gap, abs_tol=1e-12)

    @pytest.mark.parametrize("seeds", [[1], [1, 2], list(range(1, 13))])
    def test_best_found_at_is_consistent_with_the_trajectory(self, instance, seeds):
        """
        `best_found_at` must index the FIRST restart attaining the best gap.

        Asserted as an invariant rather than as a fixed index: on small
        instances the local search frequently reaches the optimum from the very
        first start, so a hardcoded expectation would be testing the fixture
        rather than the loop.
        """
        result = multistart_alpha_grasp(instance, lp_bound=4.0, seeds=seeds)

        assert 1 <= result.best_found_at <= result.restarts
        # The trajectory attains the best gap exactly at that restart...
        assert math.isclose(
            result.trajectory[result.best_found_at - 1], result.best_gap, abs_tol=1e-12
        )
        # ...and strictly improves on whatever preceded it.
        if result.best_found_at > 1:
            assert result.trajectory[result.best_found_at - 2] > result.best_gap + 1e-12
        # No earlier restart matched it.
        for earlier in result.trajectory[: result.best_found_at - 1]:
            assert earlier > result.best_gap + 1e-12

    def test_best_found_at_indexes_the_first_improvement_on_a_scripted_run(self):
        """
        The index semantics, driven directly through the dataclass so the
        result does not depend on how a particular instance happens to solve.
        """
        r = MultistartResult(
            method="m", seeds=[1, 2, 3, 4], best_cost=1.0, best_gap=0.5,
            best_open=1, best_found_at=3, trajectory=[5.0, 5.0, 0.5, 0.5],
        )
        assert r.restarts == 4
        assert r.trajectory[r.best_found_at - 1] == r.best_gap
        assert r.restarts_to_reach(0.5) == 3
        assert r.restarts_to_reach(5.0) == 1

    def test_a_later_equal_solution_does_not_steal_the_credit(self, instance):
        """`best_found_at` is the FIRST restart to reach the best value."""
        it = iter([{0, 1, 2, 3}, {0, 1, 2, 3}])
        result = run_multistart(
            "scripted", lambda: _state_for(instance, next(it)), instance, 4.0, [1, 2]
        )
        assert result.best_found_at == 1


class TestWorkAccounting:
    def test_iterations_and_moves_accumulate_across_restarts(self, instance):
        one = multistart_alpha_grasp(instance, 4.0, seeds=[1])
        four = multistart_alpha_grasp(instance, 4.0, seeds=[1, 2, 3, 4])
        assert four.total_iterations >= one.total_iterations
        assert sum(four.moves.values()) == four.total_iterations

    def test_alpha_arm_is_not_charged_for_the_lp_solve(self, instance):
        """The baseline never reads the relaxation, so it must not pay for it."""
        result = multistart_alpha_grasp(instance, 4.0, seeds=[1, 2])
        assert result.lp_seconds == 0.0
        assert result.total_seconds == result.heuristic_seconds

    def test_lp_arm_carries_the_shared_lp_cost(self, instance):
        probs = {f: 1.0 for f in instance.facilities}
        result = multistart_lp_biased(instance, probs, 4.0, seeds=[1, 2], lp_seconds=12.5)
        assert result.lp_seconds == 12.5
        assert result.total_seconds == pytest.approx(12.5 + result.heuristic_seconds)

    def test_seconds_per_restart_divides_only_the_heuristic_time(self, instance):
        result = multistart_lp_biased(
            instance, {f: 1.0 for f in instance.facilities}, 4.0, seeds=[1, 2, 3, 4],
            lp_seconds=100.0,
        )
        assert result.seconds_per_restart == pytest.approx(result.heuristic_seconds / 4)


class TestComputeMatchedComparison:
    """The accessors that make best-of-N comparable with a single deterministic run."""

    @staticmethod
    def _result(trajectory, lp_seconds=0.0, heuristic=10.0):
        return MultistartResult(
            method="m", seeds=list(range(1, len(trajectory) + 1)),
            best_cost=1.0, best_gap=trajectory[-1], best_open=1,
            best_found_at=1, trajectory=list(trajectory),
            construct_seconds=heuristic, search_seconds=0.0, lp_seconds=lp_seconds,
        )

    def test_restarts_to_reach_finds_the_first_qualifying_restart(self):
        r = self._result([5.0, 3.0, 1.0, 0.5])
        assert r.restarts_to_reach(3.0) == 2
        assert r.restarts_to_reach(1.0) == 3
        assert r.restarts_to_reach(0.5) == 4

    def test_restarts_to_reach_returns_none_when_the_budget_falls_short(self):
        r = self._result([5.0, 3.0, 1.0])
        assert r.restarts_to_reach(0.1) is None
        assert r.seconds_to_reach(0.1) is None

    def test_seconds_to_reach_includes_the_lp_and_scales_with_restarts(self):
        r = self._result([5.0, 3.0, 1.0, 0.5], lp_seconds=100.0, heuristic=40.0)
        # 40s over 4 restarts = 10s each; reaching 1.0 takes 3 restarts.
        assert r.seconds_to_reach(1.0) == pytest.approx(100.0 + 30.0)

    def test_a_target_already_met_at_the_first_restart_costs_one_restart(self):
        r = self._result([0.2, 0.2, 0.2])
        assert r.restarts_to_reach(1.0) == 1


class TestReproducibility:
    def test_the_same_seeds_reproduce_the_same_run(self, instance):
        a = multistart_alpha_grasp(instance, 4.0, seeds=[7, 8, 9])
        b = multistart_alpha_grasp(instance, 4.0, seeds=[7, 8, 9])
        assert a.best_cost == b.best_cost
        assert a.per_restart_gaps == b.per_restart_gaps
        assert a.total_iterations == b.total_iterations

    def test_restarts_are_independent_of_the_ambient_random_state(self, instance):
        random.seed(1)
        a = multistart_alpha_grasp(instance, 4.0, seeds=[7, 8])
        random.seed(999999)
        b = multistart_alpha_grasp(instance, 4.0, seeds=[7, 8])
        assert a.per_restart_gaps == b.per_restart_gaps

    def test_random_alpha_variant_is_also_reproducible(self, instance):
        a = multistart_random_alpha_grasp(instance, 4.0, seeds=[3, 4, 5])
        b = multistart_random_alpha_grasp(instance, 4.0, seeds=[3, 4, 5])
        assert a.per_restart_gaps == b.per_restart_gaps
        assert "random alpha" in a.method


class TestGuards:
    def test_an_empty_seed_list_is_rejected(self, instance):
        with pytest.raises(ValueError, match="at least one seed"):
            multistart_alpha_grasp(instance, 4.0, seeds=[])

    def test_more_restarts_never_produce_a_worse_best(self, instance):
        """
        The defining property of multistart, and the reason a best-of-N number
        must always be read alongside its budget.
        """
        seeds = list(range(1, 13))
        small = multistart_alpha_grasp(instance, 4.0, seeds=seeds[:3])
        large = multistart_alpha_grasp(instance, 4.0, seeds=seeds)
        assert large.best_cost <= small.best_cost + 1e-9
