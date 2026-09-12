import pytest
import math
from uflp_solver import (
    UFLPInstance,
    SolutionState,
    _compute_total_cost,
    local_search,
    solve_uflp
)
from run_experiments import construct_alpha_grasp_solution

def test_zero_setup_costs():
    """If setup costs are 0, the optimal solution is to open ALL facilities to minimize service costs."""
    inst = UFLPInstance(
        facilities=[0, 1, 2],
        customers=[0, 1, 2],
        setup_costs={0: 0.0, 1: 0.0, 2: 0.0},
        service_costs={
            0: {0: 10, 1: 100, 2: 100},
            1: {0: 100, 1: 10, 2: 100},
            2: {0: 100, 1: 100, 2: 10}
        }
    )
    # The heuristic should figure out that opening all 3 is best.
    # Cost with all 3 is 30. Any missing facility incurs 100.
    final_state = solve_uflp(inst, verbose=False)
    assert final_state.open_facilities == {0, 1, 2}
    assert math.isclose(final_state.total_cost, 30.0, abs_tol=1e-5)

def test_huge_setup_costs_monopoly():
    """If setup costs are massive, the optimal solution is to open exactly ONE facility."""
    inst = UFLPInstance(
        facilities=[0, 1],
        customers=[0, 1, 2],
        setup_costs={0: 10000.0, 1: 10000.0},
        service_costs={
            0: {0: 10, 1: 15},
            1: {0: 10, 1: 15},
            2: {0: 10, 1: 15}
        }
    )
    # Facility 0 service cost total = 30. Total = 10030.
    # Facility 1 service cost total = 45. Total = 10045.
    # Opening both = 20000 + 30 = 20030.
    final_state = solve_uflp(inst, verbose=False)
    assert final_state.open_facilities == {0}
    assert math.isclose(final_state.total_cost, 10030.0, abs_tol=1e-5)

def test_identical_symmetry():
    """If two facilities are identical, the solver should pick one and not cycle."""
    inst = UFLPInstance(
        facilities=[0, 1],
        customers=[0],
        setup_costs={0: 100.0, 1: 100.0},
        service_costs={
            0: {0: 10.0, 1: 10.0}
        }
    )
    final_state = solve_uflp(inst, verbose=False)
    assert len(final_state.open_facilities) == 1
    assert math.isclose(final_state.total_cost, 110.0, abs_tol=1e-5)

def test_single_facility_single_customer():
    """Minimal instance size 1x1."""
    inst = UFLPInstance(
        facilities=[0],
        customers=[0],
        setup_costs={0: 42.0},
        service_costs={0: {0: 8.0}}
    )
    final_state = solve_uflp(inst, verbose=False)
    assert final_state.open_facilities == {0}
    assert math.isclose(final_state.total_cost, 50.0, abs_tol=1e-5)

def test_alpha_grasp_extreme_alphas():
    """Test alpha = 0.0 (pure greedy) and alpha = 1.0 (pure random)."""
    inst = UFLPInstance(
        facilities=[0, 1],
        customers=[0, 1],
        setup_costs={0: 10.0, 1: 100.0},
        service_costs={
            0: {0: 10.0, 1: 10.0},
            1: {0: 10.0, 1: 10.0}
        }
    )
    # alpha = 0.0 (Greedy) should definitely pick facility 0 first
    state_greedy = construct_alpha_grasp_solution(inst, alpha=0.0)
    assert 0 in state_greedy.open_facilities
    
    # alpha = 1.0 (Random) is valid and won't crash
    state_random = construct_alpha_grasp_solution(inst, alpha=1.0)
    assert len(state_random.open_facilities) > 0

def test_local_search_rejects_micro_profits():
    """
    Test that moves with tiny floating point profits (< 1e-9) are rejected 
    to prevent numerical cycling.
    """
    inst = UFLPInstance(
        facilities=[0, 1],
        customers=[0],
        setup_costs={0: 100.0, 1: 100.0 - 1e-10}, # Facility 1 is technically cheaper by 1e-10
        service_costs={
            0: {0: 10.0, 1: 10.0}
        }
    )
    state = SolutionState(
        open_facilities={0},
        closest_facility={0: 0},
        second_closest_facility={0: 0}
    )
    state.total_cost = _compute_total_cost(inst, state)
    from uflp_solver import _compute_auxiliary_data
    _compute_auxiliary_data(inst, state)
    
    # Running local search. Facility 1 is better by 1e-10, but threshold is 1e-9.
    # Therefore, the local search should reject the swap and terminate immediately.
    final_state = local_search(inst, state, verbose=False)
    assert final_state.open_facilities == {0}


def test_empty_instance():
    """An instance with 0 facilities or 0 customers should raise an error early."""
    with pytest.raises(ValueError):
        inst = UFLPInstance(facilities=[], customers=[], setup_costs={}, service_costs={})
        solve_uflp(inst, verbose=False)
