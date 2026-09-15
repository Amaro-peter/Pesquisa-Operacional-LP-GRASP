"""
Unit tests for reference values.

A `Reference` decides what a reported percentage is allowed to be called. Get
it wrong and every gap in every report is mislabelled, which is the specific
criticism this module exists to answer: earlier revisions measured against the
LP bound and called the result an "optimality gap".

The failure paths matter as much as the happy one. If CBC times out, errors, or
returns something impossible, the module must fall back to the LP bound and say
so — never silently promote a bound to an optimum.
"""

import math
from unittest.mock import patch

import pytest

from scripts.reference import (
    PROVEN_OPTIMAL,
    Reference,
    compute_reference,
    solve_exact_ip,
)
from scripts.uflp_solver import UFLPInstance


@pytest.fixture
def tiny() -> UFLPInstance:
    """Optimum: open facility 0 only, for 1 + 5 + 7 = 13."""
    return UFLPInstance(
        facilities=[0, 1],
        customers=[0, 1],
        setup_costs={0: 1.0, 1: 100.0},
        service_costs={0: {0: 5.0, 1: 6.0}, 1: {0: 7.0, 1: 8.0}},
    )


class TestLabelling:
    def test_a_proven_optimum_may_be_called_an_optimality_gap(self):
        ref = Reference(value=100.0, lp_bound=90.0, proven=True,
                        ip_status="Optimal", ip_seconds=1.0)
        assert ref.kind == "proven_optimum"
        assert ref.gap_label == "optimality gap"
        assert "proven integer optimum" in ref.describe()
        assert "OVERSTATE" not in ref.describe()

    def test_an_unproven_bound_may_not(self):
        """COUNTERWEIGHT: the whole point of the module."""
        ref = Reference(value=90.0, lp_bound=90.0, proven=False,
                        ip_status="Not Solved", ip_seconds=300.0)
        assert ref.kind == "lp_bound"
        assert ref.gap_label == "gap vs LP bound"
        assert "OVERSTATE the true optimality gap" in ref.describe()
        assert "proven integer optimum" not in ref.describe()

    def test_gap_is_measured_against_the_reference_value_not_the_lp_bound(self):
        ref = Reference(value=100.0, lp_bound=90.0, proven=True,
                        ip_status="Optimal", ip_seconds=1.0)
        assert ref.gap(110.0) == pytest.approx(10.0)
        assert ref.gap(100.0) == pytest.approx(0.0)
        # Against the LP bound the same cost would look 22% worse; it must not.
        assert ref.gap(110.0) != pytest.approx((110.0 - 90.0) / 90.0 * 100)

    def test_short_label_tracks_the_long_one(self):
        assert Reference(1, 1, True, "Optimal", 0).gap_label_short == "opt. gap"
        assert Reference(1, 1, False, "x", 0).gap_label_short == "gap vs bound"


class TestExactSolve:
    def test_it_finds_the_known_optimum(self, tiny):
        status, solution_status, objective, seconds = solve_exact_ip(tiny, time_limit=60)
        assert status == "Optimal"
        assert objective == pytest.approx(13.0)
        assert seconds >= 0.0

    def test_it_reports_whether_optimality_was_actually_proven(self, tiny):
        """
        `LpStatus` says "Optimal" for an unproven incumbent too, so the caller
        needs the second flag to tell the two apart.
        """
        _status, solution_status, _obj, _s = solve_exact_ip(tiny, time_limit=60)
        assert solution_status == PROVEN_OPTIMAL

    def test_relaxing_the_assignment_variables_does_not_change_the_optimum(self, tiny):
        """
        The model leaves x continuous because an uncapacitated optimum assigns
        each customer wholly to its cheapest open facility anyway. If that were
        wrong, the objective would come out below the true integer optimum.
        """
        _status, _solution, objective, _ = solve_exact_ip(tiny, time_limit=60)
        brute = min(
            sum(tiny.setup_costs[f] for f in S)
            + sum(min(tiny.service_costs[u][f] for f in S) for u in tiny.customers)
            for S in ([0], [1], [0, 1])
        )
        assert objective == pytest.approx(brute)


class TestComputeReference:
    def test_a_solved_instance_yields_a_proven_optimum(self, tiny):
        ref = compute_reference(tiny, lp_bound=12.0, time_limit=60)
        assert ref.proven
        assert ref.value == pytest.approx(13.0)
        assert ref.lp_bound == 12.0
        assert ref.gap_label == "optimality gap"

    def test_attempt_exact_false_skips_the_solver_entirely(self, tiny):
        """For instances known to be out of reach, do not burn the time limit."""
        ref = compute_reference(tiny, lp_bound=12.0, attempt_exact=False)
        assert not ref.proven
        assert ref.value == 12.0
        assert ref.ip_status == "not attempted"
        assert ref.ip_seconds == 0.0

    def test_a_timeout_falls_back_to_the_lp_bound(self, tiny):
        with patch("scripts.reference.solve_exact_ip",
                   return_value=("Not Solved", "No Solution Found", 99.0, 300.0)):
            ref = compute_reference(tiny, lp_bound=12.0, time_limit=300)
        assert not ref.proven
        assert ref.value == 12.0
        assert ref.incumbent == 99.0        # the incumbent is retained...
        assert ref.gap(12.0) == 0.0         # ...but gaps use the bound
        assert "Not Solved" in ref.describe()

    def test_an_unproven_incumbent_is_labelled_as_such_not_as_not_solved(self, tiny):
        """
        The distinction a reader needs: CBC finding nothing is a different
        failure from CBC finding something it could not prove.
        """
        with patch("scripts.reference.solve_exact_ip",
                   return_value=("Optimal", "Solution Found", 99.0, 300.0)):
            ref = compute_reference(tiny, lp_bound=12.0, time_limit=300)
        assert not ref.proven
        assert ref.ip_status == "not proven (Solution Found)"
        assert ref.incumbent == 99.0
        assert ref.value == 12.0

    def test_a_solver_error_falls_back_to_the_lp_bound(self, tiny):
        with patch("scripts.reference.solve_exact_ip", side_effect=RuntimeError("boom")):
            ref = compute_reference(tiny, lp_bound=12.0)
        assert not ref.proven
        assert ref.value == 12.0
        assert "error: RuntimeError" in ref.ip_status

    def test_an_objective_below_the_lp_bound_is_rejected_not_trusted(self, tiny):
        """
        The LP is a relaxation, so no integer solution can beat its bound. An
        objective that does indicates a malformed model, and must not be
        promoted to "the proven optimum".
        """
        with patch("scripts.reference.solve_exact_ip",
                   return_value=("Optimal", PROVEN_OPTIMAL, 5.0, 1.0)):
            ref = compute_reference(tiny, lp_bound=12.0)
        assert not ref.proven
        assert ref.value == 12.0
        assert "below LP bound" in ref.ip_status
        assert ref.incumbent == 5.0

    def test_an_objective_equal_to_the_lp_bound_is_accepted(self):
        """COUNTERWEIGHT: attaining the bound is normal, not an anomaly."""
        instance = UFLPInstance(
            facilities=[0], customers=[0],
            setup_costs={0: 1.0}, service_costs={0: {0: 2.0}},
        )
        ref = compute_reference(instance, lp_bound=3.0, time_limit=60)
        assert ref.proven
        assert ref.value == pytest.approx(3.0)
        assert ref.gap(3.0) == pytest.approx(0.0)

    def test_a_missing_objective_is_not_treated_as_proven(self, tiny):
        with patch("scripts.reference.solve_exact_ip",
                   return_value=("Optimal", PROVEN_OPTIMAL, None, 1.0)):
            ref = compute_reference(tiny, lp_bound=12.0)
        assert not ref.proven
        assert ref.value == 12.0
