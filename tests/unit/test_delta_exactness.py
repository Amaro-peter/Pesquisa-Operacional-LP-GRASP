"""
Exhaustive cross-validation of the O(1) delta formulas against brute force.

`_compute_insert_profit`, `_compute_delete_profit` and `_compute_swap_profit`
are the load-bearing part of the local search: every acceptance decision and
every reported cost depends on them being *exact*, not approximate. These tests
enumerate every non-empty open set of small instances and compare each formula
against a from-scratch recomputation of the resulting solution cost.

They also pin the boundary conditions that the incremental formulas rely on:
  * the single-open-facility branch of `_compute_swap_profit`, which cannot use
    `loss` (there is no valid second-closest backup);
  * the sparse-`extra` default, exercised by pairs whose correction is zero;
  * exact ties in the service-cost matrix, where `phi1`/`phi2` are ambiguous.
"""

import itertools
import math
import random

import pytest

from uflp_solver import (
    SolutionState,
    UFLPInstance,
    _compute_auxiliary_data,
    _compute_delete_profit,
    _compute_insert_profit,
    _compute_swap_profit,
    _compute_total_cost,
    _find_closest_two,
)

TOL = 1e-9


def _state_for(instance, open_facilities):
    state = SolutionState(open_facilities=set(open_facilities))
    for u in instance.customers:
        closest, second = _find_closest_two(u, state.open_facilities, instance.service_costs)
        state.closest_facility[u] = closest
        state.second_closest_facility[u] = second
    state.total_cost = _compute_total_cost(instance, state)
    _compute_auxiliary_data(instance, state)
    return state


def _brute_force_cost(instance, open_facilities):
    """Cost of a solution, recomputed from scratch with no incremental state."""
    open_facilities = set(open_facilities)
    cost = sum(instance.setup_costs[f] for f in open_facilities)
    for u in instance.customers:
        cost += min(instance.service_costs[u][f] for f in open_facilities)
    return cost


def _make_instance(rng, n_fac, n_cust, flavour):
    facilities = list(range(n_fac))
    customers = list(range(n_cust))
    if flavour == "euclidean":
        fac_xy = {f: (rng.uniform(0, 100), rng.uniform(0, 100)) for f in facilities}
        cust_xy = {u: (rng.uniform(0, 100), rng.uniform(0, 100)) for u in customers}
        service = {
            u: {
                f: math.dist(cust_xy[u], fac_xy[f])
                for f in facilities
            }
            for u in customers
        }
    elif flavour == "non_euclidean":
        service = {u: {f: rng.uniform(0, 100) for f in facilities} for u in customers}
    elif flavour == "tied":
        # Small integer costs guarantee frequent exact ties in phi1/phi2.
        service = {u: {f: float(rng.randint(0, 3)) for f in facilities} for u in customers}
    else:  # pragma: no cover - guard against typos in the parametrisation
        raise AssertionError(f"unknown flavour {flavour}")
    setup = {f: rng.uniform(10, 300) for f in facilities}
    return UFLPInstance(facilities, customers, setup, service)


@pytest.mark.parametrize("flavour", ["euclidean", "non_euclidean", "tied"])
@pytest.mark.parametrize("n_fac,n_cust", [(2, 1), (3, 4), (5, 3), (6, 5)])
def test_all_move_deltas_are_exact_over_every_open_set(flavour, n_fac, n_cust):
    """
    For every non-empty subset of facilities, every insert/delete/swap profit
    must equal (cost before - cost after) computed from scratch.
    """
    rng = random.Random(hash((flavour, n_fac, n_cust)) & 0xFFFF)
    instance = _make_instance(rng, n_fac, n_cust, flavour)

    checked_insert = checked_delete = checked_swap = 0

    for size in range(1, n_fac + 1):
        for open_set in itertools.combinations(range(n_fac), size):
            state = _state_for(instance, open_set)
            base = _brute_force_cost(instance, open_set)
            assert math.isclose(state.total_cost, base, abs_tol=TOL)

            closed = [f for f in range(n_fac) if f not in open_set]

            for f_i in closed:
                expected = base - _brute_force_cost(instance, set(open_set) | {f_i})
                assert math.isclose(
                    _compute_insert_profit(instance, state, f_i), expected, abs_tol=TOL
                ), f"insert {f_i} into {open_set}"
                checked_insert += 1

            if size > 1:
                for f_r in open_set:
                    expected = base - _brute_force_cost(instance, set(open_set) - {f_r})
                    assert math.isclose(
                        _compute_delete_profit(instance, state, f_r), expected, abs_tol=TOL
                    ), f"delete {f_r} from {open_set}"
                    checked_delete += 1

            for f_i in closed:
                for f_r in open_set:
                    after = (set(open_set) - {f_r}) | {f_i}
                    expected = base - _brute_force_cost(instance, after)
                    assert math.isclose(
                        _compute_swap_profit(instance, state, f_i, f_r), expected, abs_tol=TOL
                    ), f"swap in={f_i} out={f_r} from {open_set}"
                    checked_swap += 1

    # Guard against the loops silently degenerating into no-ops.
    assert checked_insert > 0 and checked_swap > 0
    if n_fac > 1:
        assert checked_delete > 0


def test_swap_profit_uses_exact_branch_when_one_facility_is_open():
    """
    With a single open facility there is no valid second-closest backup, so
    `loss` is meaningless and `_compute_swap_profit` must fall through to the
    direct O(|U|) computation.
    """
    instance = UFLPInstance(
        facilities=[0, 1],
        customers=[0, 1],
        setup_costs={0: 100.0, 1: 20.0},
        service_costs={
            0: {0: 5.0, 1: 30.0},
            1: {0: 7.0, 1: 11.0},
        },
    )
    state = _state_for(instance, {0})
    # before = 100 + 5 + 7 = 112 ; after = 20 + 30 + 11 = 61 ; profit = +51
    assert math.isclose(state.total_cost, 112.0, abs_tol=TOL)
    assert math.isclose(_compute_swap_profit(instance, state, f_i=1, f_r=0), 51.0, abs_tol=TOL)


def test_extra_is_sparse_and_defaults_to_zero_for_absent_pairs():
    """
    `extra` is stored sparsely and read with `.get((f_i, f_r), 0.0)`. A pair
    whose correction is zero must be absent from the dict *and* must still
    produce the exact swap profit -- i.e. the default really is 0.0.
    """
    instance = UFLPInstance(
        facilities=[0, 1, 2],
        customers=[0, 1],
        setup_costs={0: 10.0, 1: 10.0, 2: 10.0},
        service_costs={
            # Facility 2 is strictly worse than both phi1 and phi2 everywhere,
            # so it can never trigger a correction term.
            0: {0: 1.0, 1: 2.0, 2: 900.0},
            1: {0: 3.0, 1: 4.0, 2: 900.0},
        },
    )
    state = _state_for(instance, {0, 1})

    assert (2, 0) not in state.extra
    assert (2, 1) not in state.extra

    for f_r in (0, 1):
        after = ({0, 1} - {f_r}) | {2}
        expected = state.total_cost - _brute_force_cost(instance, after)
        assert math.isclose(
            _compute_swap_profit(instance, state, f_i=2, f_r=f_r), expected, abs_tol=TOL
        )


def test_extra_is_populated_when_the_incoming_facility_beats_phi2():
    """
    The complementary case: when the incoming facility is closer than phi2 for
    a customer currently served by the outgoing facility, `extra` must carry a
    non-zero correction (otherwise save/loss double-count that customer).
    """
    instance = UFLPInstance(
        facilities=[0, 1, 2],
        customers=[0, 1],
        setup_costs={0: 10.0, 1: 10.0, 2: 10.0},
        service_costs={
            0: {0: 1.0, 1: 500.0, 2: 500.0},
            # customer 1: phi1 = 1 (5), phi2 = 0 (400); incoming 2 costs 9 < 400
            1: {0: 400.0, 1: 5.0, 2: 9.0},
        },
    )
    state = _state_for(instance, {0, 1})
    assert state.closest_facility[1] == 1
    assert state.second_closest_facility[1] == 0

    # correction = (d(1,phi2)=400 - d(1,2)=9) - max(0, d(1,f_r)=5 - d(1,2)=9)
    #            = 391 - 0 = 391
    assert (2, 1) in state.extra
    assert math.isclose(state.extra[(2, 1)], 391.0, abs_tol=TOL)

    expected = state.total_cost - _brute_force_cost(instance, {0, 2})
    assert math.isclose(_compute_swap_profit(instance, state, f_i=2, f_r=1), expected, abs_tol=TOL)


def test_save_and_loss_have_the_documented_sign_convention():
    """
    Positive `save` means inserting improves; positive `loss` means removing
    worsens. A sign flip in either would invert every acceptance decision.
    """
    instance = UFLPInstance(
        facilities=[0, 1],
        customers=[0],
        setup_costs={0: 10.0, 1: 1.0},
        service_costs={0: {0: 100.0, 1: 2.0}},
    )
    state = _state_for(instance, {0})

    # Inserting 1: pay setup 1, customer moves from 100 -> 2, so save = -1 + 98 = +97
    assert math.isclose(state.save[1], 97.0, abs_tol=TOL)
    assert _compute_insert_profit(instance, state, 1) > 0

    # With {0, 1} open, removing 1 costs: give back setup 1 but pay 100 - 2 = 98
    state_both = _state_for(instance, {0, 1})
    assert math.isclose(state_both.loss[1], 97.0, abs_tol=TOL)
    assert _compute_delete_profit(instance, state_both, 1) < 0
