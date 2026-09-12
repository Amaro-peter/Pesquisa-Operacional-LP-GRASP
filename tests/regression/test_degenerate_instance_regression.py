"""
Regression: degenerate instances were accepted and silently mis-solved.

WHAT BROKE
    `solve_uflp` performed no input validation.

    1. A customer-free instance returned a *wrong* answer instead of erroring.
       `construct_solution`'s fallback force-opens the cheapest facility when
       the probabilistic pass opens nothing, and `local_search` refuses to
       delete the last open facility (`if len(open_facilities) > 1`). With zero
       customers the true optimum is to open nothing, at cost 0 -- but the
       solver returned one open facility and its setup cost.

    2. A facility-free instance surfaced as an opaque SciPy message
       ("Invalid input for linprog: c must be a 1-D array ...") raised from
       deep inside `linprog` on a zero-length objective vector.

WHY IT MATTERED
    (1) is a silent wrong answer on a valid, if degenerate, input. (2) made the
    existing `test_empty_instance` a false guard: it asserted `ValueError`, but
    the exception came from SciPy's argument parsing, not from any deliberate
    check in this codebase. Any change to how the LP is built (a different
    backend, a dense path, an early return on an empty model) would have
    silently turned that assertion into a different error or no error at all,
    while the test kept passing for the wrong reason.

FIX
    `uflp_solver.validate_instance`, called at the top of `solve_uflp`, rejects
    empty facility/customer lists and incomplete cost tables with explicit,
    attributable `ValueError`s.

DEFECT CONTEXT
    Found while auditing the provenance of `output/california_4M_results.md`.
"""

import math
import os
import sys

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from scripts.uflp_solver import UFLPInstance, solve_uflp


def test_customer_free_instance_is_rejected_instead_of_mis_solved():
    """
    Pre-fix this returned a SolutionState with one open facility and
    total_cost == 5.0, rather than raising. RED before `validate_instance`.
    """
    instance = UFLPInstance(
        facilities=[0, 1],
        customers=[],
        setup_costs={0: 5.0, 1: 7.0},
        service_costs={},
    )
    with pytest.raises(ValueError, match="at least one customer"):
        solve_uflp(instance, verbose=False)


def test_facility_free_instance_raises_an_attributable_error():
    """
    The error must come from this codebase and name the real problem, not leak
    SciPy's "c must be a 1-D array" message.
    """
    instance = UFLPInstance(
        facilities=[],
        customers=[0, 1],
        setup_costs={},
        service_costs={0: {}, 1: {}},
    )
    with pytest.raises(ValueError, match="at least one facility"):
        solve_uflp(instance, verbose=False)

    from scripts.uflp_solver import validate_instance

    with pytest.raises(ValueError) as excinfo:
        validate_instance(instance)
    assert "linprog" not in str(excinfo.value)


def test_incomplete_setup_costs_are_reported_with_the_offending_facility():
    instance = UFLPInstance(
        facilities=[0, 1],
        customers=[0],
        setup_costs={0: 5.0},  # facility 1 missing
        service_costs={0: {0: 1.0, 1: 2.0}},
    )
    with pytest.raises(ValueError, match=r"Missing setup costs for facilities: \[1\]"):
        solve_uflp(instance, verbose=False)


def test_incomplete_service_costs_are_reported_with_the_offending_pair():
    instance = UFLPInstance(
        facilities=[0, 1],
        customers=[0, 1],
        setup_costs={0: 5.0, 1: 7.0},
        service_costs={0: {0: 1.0, 1: 2.0}, 1: {0: 3.0}},  # (customer 1, facility 1) missing
    )
    with pytest.raises(ValueError, match=r"customer 1, facilities: \[1\]"):
        solve_uflp(instance, verbose=False)

    instance_no_row = UFLPInstance(
        facilities=[0],
        customers=[0, 1],
        setup_costs={0: 5.0},
        service_costs={0: {0: 1.0}},  # customer 1 has no row at all
    )
    with pytest.raises(ValueError, match="Missing service costs for customer 1"):
        solve_uflp(instance_no_row, verbose=False)


def test_counterweight_minimal_valid_instance_still_solves():
    """
    COUNTERWEIGHT: the validation must not overshoot into rejecting legitimate
    minimal instances. The smallest well-formed instance (1 facility,
    1 customer) must still solve, and the single-open-facility deletion
    prohibition it relies on must remain intact.
    """
    instance = UFLPInstance(
        facilities=[0],
        customers=[0],
        setup_costs={0: 42.0},
        service_costs={0: {0: 8.0}},
    )
    from scripts.uflp_solver import validate_instance

    validate_instance(instance)  # must not raise

    state = solve_uflp(instance, verbose=False)
    assert state.open_facilities == {0}
    assert math.isclose(state.total_cost, 50.0, abs_tol=1e-9)


def test_counterweight_zero_valued_costs_are_not_mistaken_for_missing():
    """
    COUNTERWEIGHT: validation keys on *presence*, not truthiness. A perfectly
    valid instance whose setup and service costs are all 0.0 must pass -- a
    `if not instance.setup_costs[f]` style check would wrongly reject it.
    """
    instance = UFLPInstance(
        facilities=[0, 1],
        customers=[0, 1],
        setup_costs={0: 0.0, 1: 0.0},
        service_costs={0: {0: 0.0, 1: 0.0}, 1: {0: 0.0, 1: 0.0}},
    )
    from scripts.uflp_solver import validate_instance

    validate_instance(instance)  # must not raise

    state = solve_uflp(instance, verbose=False)
    assert math.isclose(state.total_cost, 0.0, abs_tol=1e-9)
    assert len(state.open_facilities) >= 1
