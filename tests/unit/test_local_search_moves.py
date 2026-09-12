"""
Unit tests for the *move acceptance* path of `local_search`.

The pre-existing suite only ever drove `local_search` down the deletion branch:
coverage showed lines 517-518 (accept an insertion) and 533-534 (accept a swap)
unexecuted, and mutmut confirmed it -- `best_move = ('insert', ...) -> None`,
`best_move = ('swap', ...) -> None`, `f_in -> None`, and every mutation of the
`iteration` counter survived. The iteration count is a headline metric in the
benchmark reports, so it needs assertions of its own.
"""

import math

import pytest

from uflp_solver import (
    SolutionState,
    UFLPInstance,
    _compute_auxiliary_data,
    _compute_total_cost,
    _find_closest_two,
    local_search,
)


def _state_for(instance: UFLPInstance, open_facilities) -> SolutionState:
    """Build a fully-populated SolutionState for a given open set."""
    state = SolutionState(open_facilities=set(open_facilities))
    for u in instance.customers:
        closest, second = _find_closest_two(u, state.open_facilities, instance.service_costs)
        state.closest_facility[u] = closest
        state.second_closest_facility[u] = second
    state.total_cost = _compute_total_cost(instance, state)
    _compute_auxiliary_data(instance, state)
    return state


@pytest.fixture
def insertion_instance() -> UFLPInstance:
    """
    Two far-apart customer clusters, one cheap facility in each.

    Opening only facility 0 costs 10 + 0 + 500 = 510.
    Opening both costs 10 + 10 + 0 + 0 = 20, so a single INSERT of facility 1
    is the unique improving move, worth exactly +490.
    """
    return UFLPInstance(
        facilities=[0, 1],
        customers=[0, 1],
        setup_costs={0: 10.0, 1: 10.0},
        service_costs={
            0: {0: 0.0, 1: 500.0},
            1: {0: 500.0, 1: 0.0},
        },
    )


@pytest.fixture
def swap_instance() -> UFLPInstance:
    """
    Facility 2 dominates facility 1: same setup cost, strictly better service
    for every customer. Starting from {0, 1}, the unique improving move is the
    SWAP (in=2, out=1). Deleting 1 outright is worse than swapping, and
    inserting 2 while keeping 1 pays a redundant setup cost.

    Cost {0, 1} = 50 + 50 + d(0,0)=0 + d(1,1)=40          = 140
    Cost {0, 2} = 50 + 50 + d(0,0)=0 + d(1,2)=10          = 110   <-- best
    Cost {0}    = 50      + d(0,0)=0 + d(1,0)=200         = 250
    Cost {0,1,2}= 50+50+50 + 0 + 10                       = 160
    """
    return UFLPInstance(
        facilities=[0, 1, 2],
        customers=[0, 1],
        setup_costs={0: 50.0, 1: 50.0, 2: 50.0},
        service_costs={
            0: {0: 0.0, 1: 300.0, 2: 300.0},
            1: {0: 200.0, 1: 40.0, 2: 10.0},
        },
    )


class TestInsertionAcceptance:
    def test_local_search_accepts_a_single_insertion(self, insertion_instance):
        state = _state_for(insertion_instance, {0})
        assert state.total_cost == 510.0

        final = local_search(insertion_instance, state, verbose=False)

        assert final.open_facilities == {0, 1}
        assert math.isclose(final.total_cost, 20.0, abs_tol=1e-9)
        # Each customer is served by its own co-located facility.
        assert final.closest_facility == {0: 0, 1: 1}

    def test_insertion_run_reports_exactly_one_iteration(self, insertion_instance, capsys):
        state = _state_for(insertion_instance, {0})
        local_search(insertion_instance, state, verbose=True)
        out = capsys.readouterr().out

        assert "Iter   1: insert (in=1, out=None)" in out
        assert "profit=+490.00" in out
        assert "Local search converged after 1 iterations" in out
        assert "Iter   2" not in out


class TestSwapAcceptance:
    def test_local_search_accepts_a_single_swap(self, swap_instance):
        state = _state_for(swap_instance, {0, 1})
        assert state.total_cost == 140.0

        final = local_search(swap_instance, state, verbose=False)

        assert final.open_facilities == {0, 2}
        assert math.isclose(final.total_cost, 110.0, abs_tol=1e-9)
        assert final.closest_facility == {0: 0, 1: 2}

    def test_swap_run_reports_exactly_one_iteration(self, swap_instance, capsys):
        state = _state_for(swap_instance, {0, 1})
        local_search(swap_instance, state, verbose=True)
        out = capsys.readouterr().out

        assert "Iter   1: swap   (in=2, out=1)" in out
        assert "profit=+30.00" in out
        assert "Local search converged after 1 iterations" in out
        assert "Iter   2" not in out


class TestIterationCounting:
    def test_converged_solution_reports_zero_iterations(self, insertion_instance, capsys):
        """An already-optimal state must apply no move and report 0 iterations."""
        state = _state_for(insertion_instance, {0, 1})
        final = local_search(insertion_instance, state, verbose=True)
        out = capsys.readouterr().out

        assert final.open_facilities == {0, 1}
        assert math.isclose(final.total_cost, 20.0, abs_tol=1e-9)
        assert "Local search converged after 0 iterations" in out
        assert "Iter" not in out

    def test_multi_move_run_counts_each_move_once(self):
        """
        Three co-located customer/facility pairs plus one expensive decoy.
        Starting from {3} (the decoy only), the search must insert 0, 1 and 2 --
        three distinct iterations -- and then delete the decoy: 4 iterations.
        """
        instance = UFLPInstance(
            facilities=[0, 1, 2, 3],
            customers=[0, 1, 2],
            setup_costs={0: 1.0, 1: 1.0, 2: 1.0, 3: 5.0},
            service_costs={
                0: {0: 0.0, 1: 900.0, 2: 900.0, 3: 400.0},
                1: {0: 900.0, 1: 0.0, 2: 900.0, 3: 400.0},
                2: {0: 900.0, 1: 900.0, 2: 0.0, 3: 400.0},
            },
        )
        state = _state_for(instance, {3})
        final = local_search(instance, state, verbose=True)

        assert final.open_facilities == {0, 1, 2}
        assert math.isclose(final.total_cost, 3.0, abs_tol=1e-9)

    def test_every_applied_move_strictly_reduces_cost(self, swap_instance, capsys):
        """
        Move acceptance must be monotone: the printed cost after each iteration
        strictly decreases. A non-exact delta (or a `>` -> `>=` acceptance flip)
        would let the search accept a zero/negative-profit move and cycle.
        """
        state = _state_for(swap_instance, {0, 1, 2})
        local_search(swap_instance, state, verbose=True)
        out = capsys.readouterr().out

        costs = [
            float(line.split("cost=")[1].split()[0].replace(",", ""))
            for line in out.splitlines()
            if "cost=" in line
        ]
        assert costs, "expected at least one applied move"
        assert costs == sorted(costs, reverse=True)
        assert all(a > b for a, b in zip(costs, costs[1:]))
