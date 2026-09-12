"""
Unit tests for the Körkel-Ghosh instance generator.

The generator's whole value is that it matches a published specification, so
these tests pin the specification rather than the implementation: the exact
fixed-cost range per class, the allocation-cost range, the symmetry property,
integrality of the costs, and reproducibility. If any of those drift, the
instances stop being Körkel-Ghosh instances and the benchmark built on them
stops meaning what it claims.
"""

import random

import pytest

from koerkel_ghosh import (
    ALLOCATION_COST_RANGE,
    FIXED_COST_RANGES,
    LIBRARY_SIZES,
    generate_koerkel_ghosh_instance,
    instance_name,
)
from uflp_solver import validate_instance


class TestSpecificationCompliance:
    @pytest.mark.parametrize("klass,lo,hi", [("a", 100, 200), ("b", 1000, 2000), ("c", 10000, 20000)])
    def test_fixed_costs_match_the_published_class_ranges(self, klass, lo, hi):
        """Class A: [100,200]; B: [1000,2000]; C: [10000,20000]."""
        assert FIXED_COST_RANGES[klass] == (lo, hi)
        instance = generate_koerkel_ghosh_instance(size=60, klass=klass, seed=3)
        values = list(instance.setup_costs.values())
        assert len(values) == 60
        assert all(lo <= v <= hi for v in values), (min(values), max(values))
        # With 60 draws the sample should span a good part of the range; a
        # collapsed or mis-scaled range would show up here.
        assert max(values) - min(values) > (hi - lo) * 0.5

    def test_allocation_costs_are_always_drawn_from_the_same_range(self):
        """The transportation costs are [1000, 2000] for every class."""
        assert ALLOCATION_COST_RANGE == (1000, 2000)
        for klass in ("a", "b", "c"):
            instance = generate_koerkel_ghosh_instance(size=40, klass=klass, seed=5)
            for u in instance.customers:
                for f in instance.facilities:
                    assert 1000 <= instance.service_costs[u][f] <= 2000

    def test_all_costs_are_integer_valued(self):
        instance = generate_koerkel_ghosh_instance(size=30, klass="b", seed=7)
        for v in instance.setup_costs.values():
            assert v == int(v)
        for u in instance.customers:
            for f in instance.facilities:
                assert instance.service_costs[u][f] == int(instance.service_costs[u][f])

    def test_symmetric_instances_satisfy_c_ij_equals_c_ji(self):
        instance = generate_koerkel_ghosh_instance(size=50, klass="a", symmetric=True, seed=11)
        for i in instance.customers:
            for j in instance.facilities:
                assert instance.service_costs[i][j] == instance.service_costs[j][i]

    def test_asymmetric_instances_are_genuinely_asymmetric(self):
        """
        COUNTERWEIGHT to the test above: the asymmetric variant must NOT be
        accidentally symmetric. With 50x50 independent draws from a 1001-value
        range, near-total symmetry is effectively impossible.
        """
        instance = generate_koerkel_ghosh_instance(size=50, klass="a", symmetric=False, seed=11)
        mismatches = sum(
            1
            for i in instance.customers
            for j in instance.facilities
            if instance.service_costs[i][j] != instance.service_costs[j][i]
        )
        assert mismatches > 0.9 * 50 * 49, mismatches

    def test_instances_are_square(self):
        instance = generate_koerkel_ghosh_instance(size=35, klass="c", seed=2)
        assert len(instance.facilities) == 35
        assert len(instance.customers) == 35

    def test_generated_instances_are_well_formed(self):
        """Every generated instance must satisfy the solver's own validation."""
        for klass in ("a", "b", "c"):
            for symmetric in (True, False):
                instance = generate_koerkel_ghosh_instance(
                    size=20, klass=klass, symmetric=symmetric, seed=1
                )
                validate_instance(instance)  # must not raise


class TestReproducibility:
    def test_same_seed_gives_identical_instances(self):
        a = generate_koerkel_ghosh_instance(size=40, klass="b", symmetric=False, seed=99)
        b = generate_koerkel_ghosh_instance(size=40, klass="b", symmetric=False, seed=99)
        assert a.setup_costs == b.setup_costs
        assert a.service_costs == b.service_costs

    def test_different_seeds_give_different_instances(self):
        a = generate_koerkel_ghosh_instance(size=40, klass="b", seed=1)
        b = generate_koerkel_ghosh_instance(size=40, klass="b", seed=2)
        assert a.setup_costs != b.setup_costs

    def test_generation_does_not_disturb_the_global_random_state(self):
        """
        The heuristics seed `random` globally. If the generator drew from the
        same stream, generating an instance would silently shift every
        subsequent heuristic run and make seeded comparisons meaningless.
        """
        random.seed(1234)
        expected = [random.random() for _ in range(5)]

        random.seed(1234)
        generate_koerkel_ghosh_instance(size=30, klass="a", seed=777)
        actual = [random.random() for _ in range(5)]

        assert actual == expected


class TestNamingAndValidation:
    def test_names_follow_the_library_convention(self):
        assert instance_name(250, "a", True, 1) == "gs250a-1"
        assert instance_name(750, "c", False, 5) == "ga750c-5"
        assert instance_name(500, "b", True, 3) == "gs500b-3"

    def test_library_sizes_are_the_published_ones(self):
        assert LIBRARY_SIZES == (250, 500, 750)

    def test_unknown_class_is_rejected(self):
        with pytest.raises(ValueError, match="unknown Körkel-Ghosh class"):
            generate_koerkel_ghosh_instance(size=10, klass="d")
        with pytest.raises(ValueError, match="unknown Körkel-Ghosh class"):
            instance_name(250, "z", True, 1)

    def test_non_positive_size_is_rejected(self):
        with pytest.raises(ValueError, match="size must be at least 1"):
            generate_koerkel_ghosh_instance(size=0, klass="a")


class TestTheFamilyIsActuallyHard:
    """
    The reason this family exists in the repo. If a change ever made these
    instances easy, they would stop serving their purpose and these tests should
    fail loudly rather than let the benchmark quietly become another soft case.
    """

    def test_the_relaxation_is_strongly_fractional(self):
        """
        Fractionality is a property of the family, not of any one draw: an
        individual small instance can come back integral by luck (size 60,
        seed 1 does). So this averages over several instances at a size where
        the effect is established, and states the family-level property.
        """
        from run_experiments import get_lp_bound_and_probs

        shares = []
        for seed in (1, 2, 3):
            instance = generate_koerkel_ghosh_instance(
                size=125, klass="a", symmetric=True, seed=seed
            )
            _bound, probs = get_lp_bound_and_probs(instance)
            fractional = sum(1 for v in probs.values() if 0.01 < v < 0.99)
            shares.append(fractional / len(instance.facilities))

        mean_share = sum(shares) / len(shares)
        assert mean_share > 0.10, (
            f"expected a strongly fractional relaxation across the family, got "
            f"per-instance fractional shares {[f'{s:.1%}' for s in shares]}"
        )
        assert sum(1 for s in shares if s > 0.05) >= 2, (
            "fractionality should be typical, not a single outlier: "
            f"{[f'{s:.1%}' for s in shares]}"
        )

    def test_rounding_the_relaxation_is_not_enough_on_the_expensive_classes(self):
        """
        The sharpest statement of why this family is hard for an LP-guided
        method: on classes B and C the relaxation spreads so thinly that
        thresholding at y >= 0.5 selects (almost) nothing, so rounding alone
        cannot produce a usable solution.
        """
        from run_experiments import get_lp_bound_and_probs

        instance = generate_koerkel_ghosh_instance(size=125, klass="c", symmetric=True, seed=1)
        _bound, probs = get_lp_bound_and_probs(instance)
        above_half = sum(1 for v in probs.values() if v >= 0.5)
        assert above_half <= 0.02 * len(instance.facilities), (
            f"expected the class-C relaxation to put almost no facility above 0.5, "
            f"got {above_half}"
        )
        assert sum(probs.values()) > 1.0, "the relaxation must still serve every customer"

    def test_allocation_costs_carry_no_geometric_structure(self):
        """
        These costs are drawn independently rather than from an embedding.

        Note what this does NOT claim: with every cost in [1000, 2000] the
        quadrangle inequality c[u1][f1] <= c[u1][f2] + c[u2][f2] + c[u2][f1]
        can never be violated, since its left side is at most 2000 and its right
        side at least 3000. These instances are not "non-metric" in that sense.

        What makes them hard is the absence of structure: two customers'
        preference orderings over facilities are uncorrelated, whereas in a
        Euclidean instance nearby customers rank facilities almost identically.
        """
        from scipy.stats import spearmanr

        instance = generate_koerkel_ghosh_instance(size=120, klass="a", symmetric=False, seed=4)
        facilities = instance.facilities
        rows = [[instance.service_costs[u][f] for f in facilities] for u in instance.customers[:12]]

        correlations = [
            abs(spearmanr(rows[i], rows[j]).statistic)
            for i in range(len(rows))
            for j in range(i + 1, len(rows))
        ]
        mean_abs_rho = sum(correlations) / len(correlations)
        assert mean_abs_rho < 0.2, (
            f"customer preference orderings should be essentially uncorrelated; "
            f"mean |rho| was {mean_abs_rho:.3f}"
        )

        # And the whole cost range spans less than a factor of two, so no
        # facility is ever dramatically better than another for a customer.
        all_costs = [c for row in rows for c in row]
        assert max(all_costs) / min(all_costs) <= 2.0
