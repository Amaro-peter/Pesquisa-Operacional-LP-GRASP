import os
import sys
import tempfile
import math
from unittest.mock import patch, MagicMock

import pytest
import pulp

from scripts.uflp_solver import (
    UFLPInstance,
    SolutionState,
    parse_orlib_instance,
    generate_random_instance,
    solve_lp_relaxation,
    _find_closest_two,
    _compute_total_cost,
    _compute_auxiliary_data,
    _compute_swap_profit,
    _compute_insert_profit,
    _compute_delete_profit,
    construct_solution,
    _apply_move_and_recompute,
    local_search,
    solve_uflp,
    main,
)


@pytest.fixture
def small_instance() -> UFLPInstance:
    """A small deterministic 3-facility, 3-customer instance for testing."""
    facilities = [0, 1, 2]
    customers = [0, 1, 2]
    setup_costs = {0: 100.0, 1: 150.0, 2: 200.0}
    service_costs = {
        0: {0: 10.0, 1: 50.0, 2: 80.0},
        1: {0: 40.0, 1: 15.0, 2: 60.0},
        2: {0: 70.0, 1: 45.0, 2: 20.0},
    }
    return UFLPInstance(
        facilities=facilities,
        customers=customers,
        setup_costs=setup_costs,
        service_costs=service_costs,
    )


class TestDataStructures:
    def test_uflp_instance_init(self, small_instance):
        assert small_instance.facilities == [0, 1, 2]
        assert small_instance.customers == [0, 1, 2]
        assert small_instance.setup_costs[0] == 100.0
        assert small_instance.service_costs[0][0] == 10.0

    def test_solution_state_defaults(self):
        state = SolutionState()
        assert state.open_facilities == set()
        assert state.total_cost == float('inf')
        assert state.closest_facility == {}
        assert state.second_closest_facility == {}
        assert state.save == {}
        assert state.loss == {}
        assert state.extra == {}


class TestInstanceIO:
    def test_parse_orlib_instance(self):
        orlib_content = """
        2 2
        100 100.0
        200 150.0
        10
        10.5 25.0
        15
        30.0 12.5
        """
        with tempfile.NamedTemporaryFile("w", delete=False) as f:
            f.write(orlib_content)
            temp_path = f.name

        try:
            inst = parse_orlib_instance(temp_path)
            assert inst.facilities == [0, 1]
            assert inst.customers == [0, 1]
            assert inst.setup_costs == {0: 100.0, 1: 150.0}
            assert inst.service_costs[0][0] == 10.5
            assert inst.service_costs[0][1] == 25.0
            assert inst.service_costs[1][0] == 30.0
            assert inst.service_costs[1][1] == 12.5
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)

    def test_generate_random_instance(self):
        inst1 = generate_random_instance(n_facilities=5, n_customers=10, seed=123)
        inst2 = generate_random_instance(n_facilities=5, n_customers=10, seed=123)

        assert inst1.facilities == list(range(5))
        assert inst1.customers == list(range(10))
        assert len(inst1.setup_costs) == 5
        assert len(inst1.service_costs) == 10

        # Verify determinism across same seed
        assert inst1.setup_costs == inst2.setup_costs
        assert inst1.service_costs == inst2.service_costs

        for f in inst1.facilities:
            assert 100.0 <= inst1.setup_costs[f] <= 5000.0
        for u in inst1.customers:
            for f in inst1.facilities:
                assert inst1.service_costs[u][f] >= 0.0


class TestLPRelaxation:
    def test_solve_lp_relaxation_basic(self, small_instance):
        probs = solve_lp_relaxation(small_instance, verbose=False)
        assert len(probs) == len(small_instance.facilities)
        for f in small_instance.facilities:
            assert 0.0 <= probs[f] <= 1.0

    def test_solve_lp_relaxation_verbose(self, small_instance, capsys):
        probs = solve_lp_relaxation(small_instance, verbose=True)
        captured = capsys.readouterr().out
        assert "[Phase 1] LP Relaxation solved" in captured
        assert "LP objective (lower bound):" in captured

    def test_solve_lp_relaxation_non_optimal_raises(self, small_instance):
        class MockResult:
            success = False
            message = "mocked infeasibility"
            status = 2
            
        with patch('scripts.uflp_solver.linprog', return_value=MockResult()):
            with pytest.raises(RuntimeError, match="LP relaxation did not reach optimality"):
                solve_lp_relaxation(small_instance, verbose=False)


class TestStateComputations:
    def test_find_closest_two_single_facility(self, small_instance):
        open_set = {1}
        c1, c2 = _find_closest_two(customer=0, open_set=open_set, service_costs=small_instance.service_costs)
        assert c1 == 1
        assert c2 == 1

    def test_find_closest_two_multiple_facilities(self, small_instance):
        open_set = {0, 1, 2}
        # Customer 0: {0: 10, 1: 50, 2: 80}
        c1, c2 = _find_closest_two(customer=0, open_set=open_set, service_costs=small_instance.service_costs)
        assert c1 == 0
        assert c2 == 1

    def test_compute_total_cost(self, small_instance):
        state = SolutionState(
            open_facilities={0, 1},
            closest_facility={0: 0, 1: 1, 2: 1},
        )
        # Setup: 100 + 150 = 250
        # Service: cust 0 -> fac 0 (10) + cust 1 -> fac 1 (15) + cust 2 -> fac 1 (45) = 70
        # Total = 320
        cost = _compute_total_cost(small_instance, state)
        assert cost == 320.0

    def test_compute_auxiliary_data(self, small_instance):
        state = SolutionState(
            open_facilities={0},
            closest_facility={0: 0, 1: 0, 2: 0},
            second_closest_facility={0: 0, 1: 0, 2: 0},
        )
        _compute_auxiliary_data(small_instance, state)
        # Closed facilities are 1 and 2
        assert 1 in state.save
        assert 2 in state.save
        assert 0 in state.loss


class TestMoveProfits:
    def test_insert_profit(self, small_instance):
        state = SolutionState(
            open_facilities={0},
            closest_facility={0: 0, 1: 0, 2: 0},
            second_closest_facility={0: 0, 1: 0, 2: 0},
        )
        state.total_cost = _compute_total_cost(small_instance, state)
        _compute_auxiliary_data(small_instance, state)

        # Profit of inserting facility 1:
        profit = _compute_insert_profit(small_instance, state, f_i=1)

        # Expected: cost before - cost after
        # Before: setup(0)=100 + d(0,0)=10 + d(1,0)=40 + d(2,0)=70 = 220
        # After inserting 1: open={0, 1}.
        # Closest: cust 0 -> 0 (10), cust 1 -> 1 (15), cust 2 -> 1 (45)
        # Cost after: setup(0)+setup(1) = 250 + 10 + 15 + 45 = 320
        # Profit = 220 - 320 = -100
        assert math.isclose(profit, -100.0, abs_tol=1e-5)

    def test_delete_profit(self, small_instance):
        state = SolutionState(
            open_facilities={0, 1},
            closest_facility={0: 0, 1: 1, 2: 1},
            second_closest_facility={0: 1, 1: 0, 2: 0},
        )
        state.total_cost = _compute_total_cost(small_instance, state)
        _compute_auxiliary_data(small_instance, state)

        profit_del_1 = _compute_delete_profit(small_instance, state, f_r=1)
        # Deleting 1:
        # Before: 320
        # After: open={0}. Cost: setup(0)=100 + d(0,0)=10 + d(1,0)=40 + d(2,0)=70 = 220
        # Profit = 320 - 220 = +100
        assert math.isclose(profit_del_1, 100.0, abs_tol=1e-5)

    def test_swap_profit_single_open_facility(self, small_instance):
        state = SolutionState(
            open_facilities={0},
            closest_facility={0: 0, 1: 0, 2: 0},
            second_closest_facility={0: 0, 1: 0, 2: 0},
        )
        state.total_cost = _compute_total_cost(small_instance, state)
        _compute_auxiliary_data(small_instance, state)

        # Swap 0 out, 1 in
        profit = _compute_swap_profit(small_instance, state, f_i=1, f_r=0)
        # Before: 220
        # After: open={1}. Cost: setup(1)=150 + d(0,1)=50 + d(1,1)=15 + d(2,1)=45 = 260
        # Profit = 220 - 260 = -40
        assert math.isclose(profit, -40.0, abs_tol=1e-5)

    def test_swap_profit_multiple_open_facilities(self, small_instance):
        state = SolutionState(
            open_facilities={0, 1},
            closest_facility={0: 0, 1: 1, 2: 1},
            second_closest_facility={0: 1, 1: 0, 2: 0},
        )
        state.total_cost = _compute_total_cost(small_instance, state)
        _compute_auxiliary_data(small_instance, state)

        # Swap 1 out, 2 in -> open={0, 2}
        profit = _compute_swap_profit(small_instance, state, f_i=2, f_r=1)
        # Cost open={0, 2}:
        # setup = 100 + 200 = 300
        # cust 0: min(10, 80) = 10 (fac 0)
        # cust 1: min(40, 60) = 40 (fac 0)
        # cust 2: min(70, 20) = 20 (fac 2)
        # After = 300 + 70 = 370
        # Before = 320
        # Profit = 320 - 370 = -50
        assert math.isclose(profit, -50.0, abs_tol=1e-5)

    def test_swap_profit_multiple_corrections(self, small_instance):
        # We need a case where TWO customers have the same closest facility,
        # and for BOTH of them, the incoming facility is closer than their second closest.
        # Let's adjust small_instance just for this test
        import copy
        inst = copy.deepcopy(small_instance)
        # f_r = 1, f_i = 2
        # cust 1: phi1=1, phi2=0. d(1, 1)=15, d(1, 0)=100. d(1, 2)=50. (50 < 100) -> correction triggers
        # cust 2: phi1=1, phi2=0. d(2, 1)=10, d(2, 0)=100. d(2, 2)=60. (60 < 100) -> correction triggers
        inst.service_costs[1][0] = 100.0
        inst.service_costs[2][0] = 100.0
        
        state = SolutionState(
            open_facilities={0, 1},
            closest_facility={0: 0, 1: 1, 2: 1},
            second_closest_facility={0: 1, 1: 0, 2: 0},
        )
        state.total_cost = _compute_total_cost(inst, state)
        _compute_auxiliary_data(inst, state)

        # We evaluate the swap
        profit = _compute_swap_profit(inst, state, f_i=2, f_r=1)
        # Verify the exact cost difference
        state_after = SolutionState(open_facilities={0, 2})
        for u in inst.customers:
            c1, c2 = _find_closest_two(u, state_after.open_facilities, inst.service_costs)
            state_after.closest_facility[u] = c1
            state_after.second_closest_facility[u] = c2
        cost_after = _compute_total_cost(inst, state_after)
        
        assert math.isclose(profit, state.total_cost - cost_after, abs_tol=1e-5)

    def test_find_closest_two_exactly_two_facilities(self, small_instance):
        c1, c2 = _find_closest_two(0, {1, 2}, small_instance.service_costs)
        # For customer 0: d(0, 1)=50, d(0, 2)=80
        assert c1 == 1
        assert c2 == 2


class TestConstruction:
    def test_construct_solution_with_probs(self, small_instance):
        lp_probs = {0: 1.0, 1: 0.0, 2: 0.0}
        state = construct_solution(small_instance, lp_probs, verbose=False)
        assert 0 in state.open_facilities
        assert len(state.closest_facility) == 3
        assert state.total_cost < float('inf')

    def test_construct_solution_fallback(self, small_instance):
        # All probabilities zero and random() returns 0.999 > EPS
        lp_probs = {0: 0.0, 1: 0.0, 2: 0.0}
        with patch('random.random', return_value=0.99):
            state = construct_solution(small_instance, lp_probs, verbose=True)
            # Should open at least 1 facility through fallback
            assert len(state.open_facilities) == 1


class TestLocalSearch:
    def test_apply_move_and_recompute(self, small_instance):
        state = SolutionState(
            open_facilities={0},
            closest_facility={0: 0, 1: 0, 2: 0},
            second_closest_facility={0: 0, 1: 0, 2: 0},
            total_cost=220.0,
        )
        # Insert 1
        _apply_move_and_recompute(small_instance, state, 'insert', f_in=1, f_out=None)
        assert state.open_facilities == {0, 1}
        assert state.closest_facility[1] == 1

        # Swap 1 for 2
        _apply_move_and_recompute(small_instance, state, 'swap', f_in=2, f_out=1)
        assert state.open_facilities == {0, 2}
        assert 1 not in state.open_facilities

        # Delete 2
        _apply_move_and_recompute(small_instance, state, 'delete', f_in=None, f_out=2)
        assert state.open_facilities == {0}

    def test_local_search_convergence(self, small_instance):
        # Start from poor solution {0, 1, 2}
        state = SolutionState(
            open_facilities={0, 1, 2},
            closest_facility={0: 0, 1: 1, 2: 2},
            second_closest_facility={0: 1, 1: 0, 2: 1},
        )
        state.total_cost = _compute_total_cost(small_instance, state)
        _compute_auxiliary_data(small_instance, state)

        opt_state = local_search(small_instance, state, verbose=True)
        # Final cost must be <= initial cost
        assert opt_state.total_cost <= 450.0
        assert len(opt_state.open_facilities) >= 1


class TestPipelineAndCLI:
    def test_solve_uflp_quiet_and_verbose(self, small_instance, capsys):
        # Test verbose
        state_v = solve_uflp(small_instance, verbose=True, seed=42)
        captured = capsys.readouterr().out
        assert "Hybrid LP-GRASP Solver for UFLP" in captured
        assert "SOLUTION SUMMARY" in captured
        assert "Customer Assignments:" in captured

        # Test quiet
        state_q = solve_uflp(small_instance, verbose=False, seed=42)
        assert state_q.total_cost == state_v.total_cost

    def test_main_random_instance(self, monkeypatch):
        test_args = ["uflp_solver.py", "--n-facilities", "5", "--n-customers", "10", "--seed", "42", "--quiet"]
        monkeypatch.setattr(sys, "argv", test_args)
        main()

    def test_main_file_instance(self, monkeypatch):
        orlib_content = """
        2 2
        100 100.0
        200 150.0
        10
        10.5 25.0
        15
        30.0 12.5
        """
        with tempfile.NamedTemporaryFile("w", delete=False) as f:
            f.write(orlib_content)
            temp_path = f.name

        try:
            test_args = ["uflp_solver.py", "--file", temp_path, "--quiet"]
            monkeypatch.setattr(sys, "argv", test_args)
            main()
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)
