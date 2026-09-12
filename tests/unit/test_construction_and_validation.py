"""
Unit tests for `construct_solution` and `validate_instance`.

Both are exercised from `tests/regression/`, but the mutation gauntlet runs
only against `tests/unit` (mutmut executes from a copied tree without `data/`).
These unit-level tests pin the same contracts so the mutation score reflects
them:

  * the EPS floor and the `lp_probs.get(f, 0.0)` default -- a `1.0` default
    would open every facility the LP never mentioned;
  * the strict `<` in `random.random() < prob`, which is what makes y=0.0 mean
    "never open this" rather than "open it whenever random() returns 0.0";
  * the fallback's ranking key, which must be setup **plus** total service
    cost, not either one alone and not their difference;
  * the exact validation messages, which are the only thing telling a caller
    which facility or customer is malformed.
"""

import math
from unittest.mock import patch

import pytest

from scripts.uflp_solver import (
    UFLPInstance,
    construct_solution,
    validate_instance,
)


@pytest.fixture
def four_facility_instance() -> UFLPInstance:
    return UFLPInstance(
        facilities=[0, 1, 2, 3],
        customers=[0, 1],
        setup_costs={0: 10.0, 1: 20.0, 2: 30.0, 3: 40.0},
        service_costs={
            0: {0: 1.0, 1: 2.0, 2: 3.0, 3: 4.0},
            1: {0: 5.0, 1: 6.0, 2: 7.0, 3: 8.0},
        },
    )


class TestProbabilisticSelection:
    def test_facilities_open_exactly_when_random_falls_below_their_probability(
        self, four_facility_instance
    ):
        """
        random() is stubbed to return 0.5 for every draw. Facilities with
        p > 0.5 must open, those with p <= 0.5 must not. This pins the
        comparison direction and the strictness of `<`.
        """
        lp_probs = {0: 0.9, 1: 0.5, 2: 0.51, 3: 0.0}
        with patch("random.random", return_value=0.5):
            state = construct_solution(four_facility_instance, lp_probs, verbose=False)

        assert state.open_facilities == {0, 2}

    def test_probability_exactly_equal_to_the_draw_does_not_open(self, four_facility_instance):
        """`random() < prob` is strict: a draw equal to p must NOT open."""
        lp_probs = {0: 0.25, 1: 0.25, 2: 0.25, 3: 1.0}
        with patch("random.random", return_value=0.25):
            state = construct_solution(four_facility_instance, lp_probs, verbose=False)

        assert state.open_facilities == {3}

    def test_missing_facility_defaults_to_zero_then_the_eps_floor(self, four_facility_instance):
        """
        A facility absent from lp_probs must be treated as y=0.0 and therefore
        get only the EPS=0.01 floor -- never a 1.0 default, which would open
        every unmentioned facility.
        """
        lp_probs = {0: 1.0}  # facilities 1, 2, 3 absent entirely

        # A draw just above the EPS floor must leave the absent facilities shut.
        with patch("random.random", return_value=0.02):
            state = construct_solution(four_facility_instance, lp_probs, verbose=False)
        assert state.open_facilities == {0}

        # A draw just below the floor opens them -- proving the floor is 0.01,
        # not 0.0 and not 1.0.
        with patch("random.random", return_value=0.005):
            state = construct_solution(four_facility_instance, lp_probs, verbose=False)
        assert state.open_facilities == {0, 1, 2, 3}

    def test_eps_floor_lifts_zero_probability_facilities(self, four_facility_instance):
        """An explicit y=0.0 is floored to EPS, not left at zero."""
        lp_probs = {f: 0.0 for f in four_facility_instance.facilities}
        with patch("random.random", return_value=0.009):
            state = construct_solution(four_facility_instance, lp_probs, verbose=False)
        assert state.open_facilities == {0, 1, 2, 3}


class TestFallback:
    def test_fallback_ranks_by_setup_plus_total_service_cost(self):
        """
        When nothing opens, the fallback must pick the facility minimising
        setup + sum of service costs.

        Facility 0: 100 + (1 + 1)   = 102
        Facility 1:   5 + (60 + 60) = 125   <- cheapest setup, but worst total
        Facility 2:  40 + (30 + 30) = 100   <- true minimum
        A key of setup alone would pick 1; setup - services would pick 1 as
        well; services alone would pick 0.
        """
        instance = UFLPInstance(
            facilities=[0, 1, 2],
            customers=[0, 1],
            setup_costs={0: 100.0, 1: 5.0, 2: 40.0},
            service_costs={
                0: {0: 1.0, 1: 60.0, 2: 30.0},
                1: {0: 1.0, 1: 60.0, 2: 30.0},
            },
        )
        lp_probs = {f: 0.0 for f in instance.facilities}
        with patch("random.random", return_value=0.99):
            state = construct_solution(instance, lp_probs, verbose=False)

        assert state.open_facilities == {2}
        assert math.isclose(state.total_cost, 100.0, abs_tol=1e-9)

    def test_fallback_populates_assignments_and_cost(self, four_facility_instance):
        lp_probs = {f: 0.0 for f in four_facility_instance.facilities}
        with patch("random.random", return_value=0.99):
            state = construct_solution(four_facility_instance, lp_probs, verbose=False)

        assert len(state.open_facilities) == 1
        only = next(iter(state.open_facilities))
        assert state.closest_facility == {0: only, 1: only}
        assert state.second_closest_facility == {0: only, 1: only}
        assert state.total_cost < float("inf")


class TestConstructionReporting:
    def test_verbose_output_reports_the_open_count_and_cost(self, four_facility_instance):
        lp_probs = {0: 1.0, 1: 0.0, 2: 0.0, 3: 0.0}
        with patch("random.random", return_value=0.5):
            state = construct_solution(four_facility_instance, lp_probs, verbose=True)
        assert state.open_facilities == {0}

    def test_verbose_text_names_the_phase_counts_and_cost(self, four_facility_instance, capsys):
        lp_probs = {0: 1.0, 1: 0.0, 2: 0.0, 3: 0.0}
        with patch("random.random", return_value=0.5):
            construct_solution(four_facility_instance, lp_probs, verbose=True)
        out = capsys.readouterr().out

        assert "[Phase 2] Constructed initial solution" in out
        assert "Open facilities: 1/4" in out
        # setup 10 + service 1 + 5 = 16
        assert "Initial cost: 16.00" in out

    def test_quiet_construction_prints_nothing(self, four_facility_instance, capsys):
        lp_probs = {0: 1.0, 1: 0.0, 2: 0.0, 3: 0.0}
        with patch("random.random", return_value=0.5):
            construct_solution(four_facility_instance, lp_probs, verbose=False)
        assert capsys.readouterr().out == ""


class TestValidation:
    def test_empty_facility_list_message(self):
        instance = UFLPInstance(facilities=[], customers=[0], setup_costs={}, service_costs={0: {}})
        with pytest.raises(ValueError) as excinfo:
            validate_instance(instance)
        assert str(excinfo.value) == "UFLP instance must contain at least one facility"

    def test_empty_customer_list_message(self):
        instance = UFLPInstance(
            facilities=[0], customers=[], setup_costs={0: 1.0}, service_costs={}
        )
        with pytest.raises(ValueError) as excinfo:
            validate_instance(instance)
        assert str(excinfo.value) == "UFLP instance must contain at least one customer"

    def test_facility_check_precedes_the_customer_check(self):
        """Both empty: the facility message must win, deterministically."""
        instance = UFLPInstance(facilities=[], customers=[], setup_costs={}, service_costs={})
        with pytest.raises(ValueError) as excinfo:
            validate_instance(instance)
        assert "facility" in str(excinfo.value)

    def test_missing_setup_costs_list_every_offending_facility_in_order(self):
        instance = UFLPInstance(
            facilities=[0, 1, 2, 3],
            customers=[0],
            setup_costs={1: 1.0},
            service_costs={0: {0: 1.0, 1: 1.0, 2: 1.0, 3: 1.0}},
        )
        with pytest.raises(ValueError) as excinfo:
            validate_instance(instance)
        assert str(excinfo.value) == "Missing setup costs for facilities: [0, 2, 3]"

    def test_missing_service_row_names_the_customer(self):
        instance = UFLPInstance(
            facilities=[0],
            customers=[0, 7],
            setup_costs={0: 1.0},
            service_costs={0: {0: 1.0}},
        )
        with pytest.raises(ValueError) as excinfo:
            validate_instance(instance)
        assert str(excinfo.value) == "Missing service costs for customer 7"

    def test_missing_service_entries_name_the_customer_and_facilities(self):
        instance = UFLPInstance(
            facilities=[0, 1, 2],
            customers=[0],
            setup_costs={0: 1.0, 1: 1.0, 2: 1.0},
            service_costs={0: {1: 5.0}},
        )
        with pytest.raises(ValueError) as excinfo:
            validate_instance(instance)
        assert str(excinfo.value) == "Missing service costs for customer 0, facilities: [0, 2]"

    def test_a_complete_instance_passes_silently(self):
        instance = UFLPInstance(
            facilities=[0, 1],
            customers=[0, 1],
            setup_costs={0: 1.0, 1: 2.0},
            service_costs={0: {0: 1.0, 1: 2.0}, 1: {0: 3.0, 1: 4.0}},
        )
        assert validate_instance(instance) is None

    def test_extra_unused_cost_entries_are_tolerated(self):
        """
        Validation checks that every declared facility/customer has costs, not
        that the dicts contain nothing else. A stricter check would reject
        legitimately over-specified instances.
        """
        instance = UFLPInstance(
            facilities=[0],
            customers=[0],
            setup_costs={0: 1.0, 99: 5.0},
            service_costs={0: {0: 1.0, 99: 7.0}, 42: {0: 1.0}},
        )
        assert validate_instance(instance) is None
