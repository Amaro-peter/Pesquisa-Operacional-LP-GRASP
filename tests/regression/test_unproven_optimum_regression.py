"""
Regression: a time-limited CBC run was being reported as a PROVEN OPTIMUM.

WHAT BROKE
    `compute_reference` decided a reference was a proven integer optimum from
    `pulp.LpStatus[model.status] == "Optimal"` alone. PuLP sets that status to
    "Optimal" whenever CBC terminates holding an integer-feasible solution --
    *including when CBC stopped because it ran out of time*. The flag that
    actually distinguishes the two lives in `model.sol_status`:

        sol_status == 1  "Optimal Solution Found"   <- proven
        sol_status == 2  "Solution Found"           <- incumbent, NOT proven

WHY IT MATTERED
    Every gap in every report is labelled from this flag. Under the defect the
    reports said "true optimality gaps -- not gaps against a bound" over numbers
    measured against an unproven incumbent, and the README's headline claim
    ("Every integer optimum proven by CBC") rested on it.

    It was not a theoretical risk. On Koerkel-Ghosh `ga250a-1` the run reported
    a "proven optimum" of 257,553 while three of its own arms returned 257,549,
    257,540 and 257,538. The 257,538 solution was rebuilt and its cost
    recomputed from first principles -- setup costs plus each customer served by
    its cheapest open facility -- and came to exactly 257,538.000000, comfortably
    above the LP bound of 257,202.22. The heuristic was right; the "optimum" was
    an incumbent CBC had not finished proving.

THE TWO GUARDS
    1. `compute_reference` now requires `sol_status` to say the solution was
       proven optimal, not merely found.
    2. `Reference.validated_against` demotes any claimed optimum that a feasible
       solution actually beats. This one is solver-agnostic: whatever a solver
       claims, a value something beats was not a minimum.
"""

import pytest

from scripts.koerkel_ghosh import generate_koerkel_ghosh_instance
from scripts.reference import Reference, compute_reference
from scripts.uflp_solver import UFLPInstance


class TestTimeLimitIsNotProof:
    """Guard 1 -- the status flag must mean what the report says it means."""

    def test_a_cbc_run_that_stopped_without_proving_is_not_proven(self):
        """
        RED before the fix.

        This exact call is what exposed the defect. On Koerkel-Ghosh 150x150
        class a, CBC under a three-second limit returns LpStatus "Optimal" with
        objective 156,287 -- and the same instance under a thirty-second limit
        returns LpStatus "Optimal" with 156,260. Two "optimal" answers 27 apart
        cannot both be optimal; `sol_status` is 2 ("Solution Found") in both
        cases and says so. The reference must refuse the claim.
        """
        instance = generate_koerkel_ghosh_instance(150, "a", symmetric=False, seed=1)
        ref = compute_reference(instance, lp_bound=155_000.0, time_limit=3.0)

        assert not ref.proven, (
            f"an unproven CBC incumbent was reported as a proven optimum "
            f"(status {ref.ip_status!r}, {ref.ip_seconds:.1f}s)"
        )
        assert ref.kind == "lp_bound"
        assert ref.gap_label == "gap vs LP bound"
        assert "OVERSTATE the true optimality gap" in ref.describe()
        # The incumbent is kept, but clearly labelled as not proven.
        assert ref.incumbent is not None
        assert "not proven" in ref.ip_status

    def test_an_instance_cbc_really_does_close_is_still_proven(self):
        """
        COUNTERWEIGHT: the fix must not demote every reference to a bound. A
        two-facility instance is closed instantly and must still be reported as
        a proven optimum, or the guard has simply disabled the feature.
        """
        instance = UFLPInstance(
            facilities=[0, 1], customers=[0, 1],
            setup_costs={0: 1.0, 1: 100.0},
            service_costs={0: {0: 5.0, 1: 6.0}, 1: {0: 7.0, 1: 8.0}},
        )
        ref = compute_reference(instance, lp_bound=12.0, time_limit=60.0)

        assert ref.proven
        assert ref.value == pytest.approx(13.0)
        assert ref.gap_label == "optimality gap"
        assert "proven integer optimum" in ref.describe()


class TestAFeasibleSolutionBeatingTheOptimumDemotesIt:
    """Guard 2 -- solver-agnostic. A value something beats was not a minimum."""

    def _claimed(self, value=257_553.0, bound=257_202.22):
        return Reference(value=value, lp_bound=bound, proven=True,
                         ip_status="Optimal", ip_seconds=840.0, incumbent=value)

    def test_the_exact_case_observed_in_the_run(self):
        """RED before the fix: `validated_against` did not exist."""
        ref = self._claimed().validated_against(257_538.0)

        assert not ref.proven
        assert ref.value == pytest.approx(257_202.22)      # falls back to the bound
        assert ref.incumbent == pytest.approx(257_553.0)   # what CBC had claimed
        assert "257,538" in ref.ip_status and "257,553" in ref.ip_status
        assert ref.gap_label == "gap vs LP bound"

    def test_gaps_are_never_negative_after_validation(self):
        """
        The user-visible symptom: a report printing a negative optimality gap,
        which is not a thing that can exist.
        """
        raw = self._claimed()
        assert raw.gap(257_538.0) < 0.0                    # the nonsense it printed
        assert raw.validated_against(257_538.0).gap(257_538.0) > 0.0

    def test_a_solution_that_merely_attains_the_optimum_does_not_demote_it(self):
        """COUNTERWEIGHT: attaining the optimum is the expected outcome."""
        ref = self._claimed().validated_against(257_553.0)
        assert ref.proven
        assert ref.value == pytest.approx(257_553.0)
        assert ref.gap(257_553.0) == pytest.approx(0.0)

    def test_a_worse_solution_does_not_demote_it(self):
        """COUNTERWEIGHT: the usual case must be left completely alone."""
        ref = self._claimed().validated_against(300_000.0)
        assert ref.proven
        assert ref.ip_status == "Optimal"

    def test_floating_point_noise_does_not_demote_a_valid_optimum(self):
        """
        COUNTERWEIGHT: costs are summed in a different order by the solver and
        by the heuristic, so a few ulps below must not be read as a refutation.
        """
        ref = self._claimed().validated_against(257_553.0 - 1e-9)
        assert ref.proven

    def test_an_unproven_reference_is_returned_untouched(self):
        """There is nothing to demote when nothing was claimed."""
        ref = Reference(value=100.0, lp_bound=100.0, proven=False,
                        ip_status="Not Solved", ip_seconds=1.0)
        assert ref.validated_against(50.0) is ref
