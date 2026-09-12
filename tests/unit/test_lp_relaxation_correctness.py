"""
Correctness tests for the LP relaxation.

`solve_lp_relaxation` (and its twin `run_experiments.get_lp_bound_and_probs`)
hand-builds the constraint matrix with raw index arithmetic:

    y[f]      lives at column  f_map[f]
    x[u][f]   lives at column  n_F + u_map[u] * n_F + f_map[f]

A single transposed or off-by-one index would still produce a *feasible* LP and
a plausible-looking number, with nothing downstream to catch it -- the LP value
is reported as the "LP bound" that every optimality gap in the benchmark
reports is measured against. These tests pin the formulation against
independently computed ground truth rather than against itself.
"""

import itertools
import math

import pytest

from scripts.run_experiments import get_lp_bound_and_probs
from scripts.uflp_solver import UFLPInstance, generate_random_instance, solve_lp_relaxation


def _brute_force_ip_optimum(instance: UFLPInstance) -> float:
    """Exact UFLP optimum by exhaustive enumeration of open sets."""
    best = math.inf
    facilities = instance.facilities
    for size in range(1, len(facilities) + 1):
        for open_set in itertools.combinations(facilities, size):
            cost = sum(instance.setup_costs[f] for f in open_set)
            for u in instance.customers:
                cost += min(instance.service_costs[u][f] for f in open_set)
            best = min(best, cost)
    return best


@pytest.fixture
def asymmetric_instance() -> UFLPInstance:
    """
    Deliberately asymmetric in both axes: 4 facilities, 3 customers, no two
    rows or columns alike. Any index transposition changes the optimum.
    """
    return UFLPInstance(
        facilities=[0, 1, 2, 3],
        customers=[0, 1, 2],
        setup_costs={0: 90.0, 1: 30.0, 2: 55.0, 3: 210.0},
        service_costs={
            0: {0: 5.0, 1: 61.0, 2: 44.0, 3: 12.0},
            1: {0: 70.0, 1: 9.0, 2: 33.0, 3: 80.0},
            2: {0: 41.0, 1: 52.0, 2: 6.0, 3: 27.0},
        },
    )


class TestLPBoundIsValid:
    def test_lp_bound_never_exceeds_the_true_integer_optimum(self, asymmetric_instance):
        """The LP relaxation is a *relaxation*: its value is a lower bound."""
        lp_value, _probs = get_lp_bound_and_probs(asymmetric_instance)
        ip_optimum = _brute_force_ip_optimum(asymmetric_instance)
        assert lp_value <= ip_optimum + 1e-6
        # ...and a useful one, not a vacuous zero.
        assert lp_value > 0.0

    def test_lp_bound_matches_the_integer_optimum_when_the_relaxation_is_integral(
        self, asymmetric_instance
    ):
        """
        For this instance the relaxation happens to be integral, so the bound
        must equal the enumerated optimum exactly. A misplaced coefficient
        would move the bound off the optimum in either direction.
        """
        lp_value, probs = get_lp_bound_and_probs(asymmetric_instance)
        ip_optimum = _brute_force_ip_optimum(asymmetric_instance)

        integral = all(v < 1e-6 or v > 1 - 1e-6 for v in probs.values())
        if not integral:
            pytest.skip("relaxation is fractional for this instance")
        assert math.isclose(lp_value, ip_optimum, rel_tol=1e-7)

    @pytest.mark.parametrize("seed", [1, 5, 17, 33])
    def test_lp_bound_is_a_valid_lower_bound_on_random_instances(self, seed):
        instance = generate_random_instance(n_facilities=5, n_customers=4, seed=seed)
        lp_value, _probs = get_lp_bound_and_probs(instance)
        assert lp_value <= _brute_force_ip_optimum(instance) + 1e-6


class TestFractionalValuesAreWellFormed:
    def test_probabilities_are_returned_for_every_facility_and_lie_in_the_unit_interval(
        self, asymmetric_instance
    ):
        probs = solve_lp_relaxation(asymmetric_instance, verbose=False)
        assert set(probs) == set(asymmetric_instance.facilities)
        for f, v in probs.items():
            assert -1e-9 <= v <= 1.0 + 1e-9, f"y[{f}] = {v} outside [0, 1]"

    def test_y_values_dominate_the_assignment_they_must_support(self, asymmetric_instance):
        """
        The linking constraint is x[u][f] <= y[f], so every customer must be
        servable: sum of y must be at least 1, and the cheapest facility for
        some customer must carry a positive y.
        """
        probs = solve_lp_relaxation(asymmetric_instance, verbose=False)
        assert sum(probs.values()) >= 1.0 - 1e-9

        opened = {f for f, v in probs.items() if v > 1e-6}
        assert opened, "LP opened no facility at all"

    def test_both_lp_entry_points_agree(self, asymmetric_instance):
        """
        `uflp_solver.solve_lp_relaxation` and
        `run_experiments.get_lp_bound_and_probs` build the same model twice in
        two files. They must not drift apart.
        """
        probs_solver = solve_lp_relaxation(asymmetric_instance, verbose=False)
        _bound, probs_experiments = get_lp_bound_and_probs(asymmetric_instance)
        assert set(probs_solver) == set(probs_experiments)
        for f in probs_solver:
            assert math.isclose(probs_solver[f], probs_experiments[f], abs_tol=1e-7)


class TestObjectiveWiring:
    def test_setup_costs_actually_enter_the_objective(self):
        """
        Two instances identical but for one facility's setup cost must yield
        different LP bounds. If the y-coefficients were dropped or written to
        the wrong columns, the bound would be unchanged.
        """
        base = UFLPInstance(
            facilities=[0, 1],
            customers=[0],
            setup_costs={0: 10.0, 1: 10.0},
            service_costs={0: {0: 5.0, 1: 5.0}},
        )
        pricier = UFLPInstance(
            facilities=[0, 1],
            customers=[0],
            setup_costs={0: 400.0, 1: 400.0},
            service_costs={0: {0: 5.0, 1: 5.0}},
        )
        assert math.isclose(get_lp_bound_and_probs(base)[0], 15.0, abs_tol=1e-6)
        assert math.isclose(get_lp_bound_and_probs(pricier)[0], 405.0, abs_tol=1e-6)

    def test_service_costs_are_indexed_by_customer_then_facility(self):
        """
        A transposed service-cost index is the classic silent defect here. This
        instance is built so that the transpose has a strictly different
        optimum: customer 0 is cheap at facility 1, customer 1 is cheap at
        facility 0, and the setup costs break the symmetry.
        """
        instance = UFLPInstance(
            facilities=[0, 1],
            customers=[0, 1],
            setup_costs={0: 1.0, 1: 100.0},
            service_costs={
                0: {0: 50.0, 1: 1.0},
                1: {0: 2.0, 1: 60.0},
            },
        )
        # Open {0} only: 1 + 50 + 2 = 53   <-- optimum
        # Open {1} only: 100 + 1 + 60 = 161
        # Open both:     101 + 1 + 2 = 104
        lp_value, _ = get_lp_bound_and_probs(instance)
        assert math.isclose(lp_value, 53.0, abs_tol=1e-6)
        assert math.isclose(_brute_force_ip_optimum(instance), 53.0, abs_tol=1e-6)

    def test_every_customer_is_fully_assigned(self):
        """
        The equality constraint sum_f x[u][f] == 1 must hold for *every*
        customer. If a row were dropped, an expensive customer could simply go
        unserved and the bound would fall below the true optimum.
        """
        instance = UFLPInstance(
            facilities=[0],
            customers=[0, 1, 2],
            setup_costs={0: 1.0},
            service_costs={0: {0: 7.0}, 1: {0: 11.0}, 2: {0: 13.0}},
        )
        lp_value, _ = get_lp_bound_and_probs(instance)
        # Every customer must pay: 1 + 7 + 11 + 13 = 32
        assert math.isclose(lp_value, 32.0, abs_tol=1e-6)


class TestVerboseReporting:
    def test_verbose_output_reports_integral_and_fractional_counts(self, capsys):
        """
        The integral/fractional split printed here is the evidence used to
        characterise an instance's LP as "integral" or "fractional" -- a claim
        made in the benchmark write-ups -- so the counts must be right.
        """
        instance = UFLPInstance(
            facilities=[0, 1],
            customers=[0],
            setup_costs={0: 1.0, 1: 500.0},
            service_costs={0: {0: 2.0, 1: 3.0}},
        )
        solve_lp_relaxation(instance, verbose=True)
        out = capsys.readouterr().out

        assert "[Phase 1] LP Relaxation solved" in out
        assert "LP objective (lower bound): 3.00" in out
        # Facility 0 is fully opened, facility 1 fully closed.
        assert "Facilities integral (y>=0.99): 1/2" in out
        assert "Facilities fractional: 0/2" in out

    def test_infeasible_solve_raises_with_the_solver_status(self):
        from unittest.mock import patch

        class MockResult:
            success = False
            message = "mocked infeasibility"
            status = 2

        instance = UFLPInstance(
            facilities=[0], customers=[0], setup_costs={0: 1.0}, service_costs={0: {0: 1.0}}
        )
        with patch("scripts.uflp_solver.linprog", return_value=MockResult()):
            with pytest.raises(RuntimeError, match=r"status=2.*mocked infeasibility"):
                solve_lp_relaxation(instance, verbose=False)
