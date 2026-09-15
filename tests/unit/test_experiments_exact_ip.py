"""
Unit tests for the ground-truth optimum used by `run_experiments`.

`run_experiments.solve_exact_ip` supplies the value every gap in the cap134 and
duality-gap-suite reports is measured against. If it ever returns a number the
solver did not prove, those reports silently become fiction -- so the contract
is: a proven optimum, or an exception. Never a quiet approximation.
"""

from unittest.mock import patch

import pytest

from scripts.run_experiments import solve_exact_ip
from scripts.uflp_solver import UFLPInstance


@pytest.fixture
def tiny() -> UFLPInstance:
    """Optimum: open facility 0 only, for 1 + 5 + 7 = 13."""
    return UFLPInstance(
        facilities=[0, 1], customers=[0, 1],
        setup_costs={0: 1.0, 1: 100.0},
        service_costs={0: {0: 5.0, 1: 6.0}, 1: {0: 7.0, 1: 8.0}},
    )


def test_it_returns_the_proven_optimum(tiny):
    assert solve_exact_ip(tiny) == pytest.approx(13.0)


def test_it_matches_exhaustive_enumeration(tiny):
    """The value is checked against every subset, not just against itself."""
    brute = min(
        sum(tiny.setup_costs[f] for f in S)
        + sum(min(tiny.service_costs[u][f] for f in S) for u in tiny.customers)
        for S in ([0], [1], [0, 1])
    )
    assert solve_exact_ip(tiny) == pytest.approx(brute)


def test_an_unproven_incumbent_raises_rather_than_being_returned(tiny):
    """
    The defect this guards: PuLP reports LpStatus "Optimal" for a solver run
    that stopped holding an incumbent it never proved. Returning that value
    would make every downstream gap wrong in an invisible direction.
    """
    with patch("scripts.run_experiments.reference_solve_exact_ip",
               return_value=("Optimal", "Solution Found", 99.0, 1.0)):
        with pytest.raises(RuntimeError, match="did not prove optimality"):
            solve_exact_ip(tiny)


def test_a_missing_objective_raises(tiny):
    with patch("scripts.run_experiments.reference_solve_exact_ip",
               return_value=("Optimal", "Optimal Solution Found", None, 1.0)):
        with pytest.raises(RuntimeError, match="did not prove optimality"):
            solve_exact_ip(tiny)


def test_a_genuinely_proven_result_is_passed_through(tiny):
    """COUNTERWEIGHT: the guard must not reject valid optima."""
    with patch("scripts.run_experiments.reference_solve_exact_ip",
               return_value=("Optimal", "Optimal Solution Found", 42.0, 1.0)):
        assert solve_exact_ip(tiny) == 42.0
