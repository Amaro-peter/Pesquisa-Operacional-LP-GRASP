"""
Unit tests for the California-housing instance builder.

`create_california_uflp_instance` is the generator behind
`output/california_4M_results.md`, so the numbers in that report are only as
trustworthy as this function. These tests pin the properties the report's
narrative depends on: that facilities and customers are disjoint samples, that
the geographic columns really are latitude/longitude, that setup costs are
driven by block population, and that service costs are a symmetric,
non-negative, metric-consistent function of distance.

They also pin the known *approximation*: longitude degrees are converted with
the same 111 km/degree factor as latitude degrees, which overstates east-west
separation at Californian latitudes. That is a deliberate simplification of the
model, not a bug in the solver -- but it must not drift silently.
"""

import math

import numpy as np
import pytest

from download_and_run_real_world import create_california_uflp_instance

pytest.importorskip("sklearn", reason="scikit-learn is required for the California dataset")


@pytest.fixture(scope="module")
def small_instance():
    return create_california_uflp_instance(n_fac=12, n_cust=9)


class TestShapeAndSampling:
    def test_instance_has_the_requested_dimensions(self, small_instance):
        assert small_instance.facilities == list(range(12))
        assert small_instance.customers == list(range(9))
        assert len(small_instance.setup_costs) == 12
        assert len(small_instance.service_costs) == 9
        for u in small_instance.customers:
            assert len(small_instance.service_costs[u]) == 12

    def test_generation_is_deterministic(self):
        a = create_california_uflp_instance(n_fac=6, n_cust=5)
        b = create_california_uflp_instance(n_fac=6, n_cust=5)
        assert a.setup_costs == b.setup_costs
        assert a.service_costs == b.service_costs

    def test_facilities_and_customers_are_disjoint_census_blocks(self):
        """
        Facilities take indices[:n_fac] and customers indices[n_fac:2*n_fac].
        If those windows overlapped, some customers would sit exactly on top of
        a facility at zero distance and flatter the reported service costs.
        """
        from sklearn.datasets import fetch_california_housing

        data = fetch_california_housing()
        np.random.seed(42)
        indices = np.random.permutation(len(data.data))
        n_fac = n_cust = 40
        fac_indices = indices[:n_fac]
        cust_indices = indices[n_fac : n_fac + n_cust]

        assert len(set(fac_indices.tolist()) & set(cust_indices.tolist())) == 0

        instance = create_california_uflp_instance(n_fac=n_fac, n_cust=n_cust)
        zero_cost_pairs = sum(
            1
            for u in instance.customers
            for f in instance.facilities
            if instance.service_costs[u][f] == 0.0
        )
        assert zero_cost_pairs == 0


class TestCostSemantics:
    def test_setup_costs_track_block_population(self, small_instance):
        """
        setup = 5000 * (population / median_population) + 1000, so the cheapest
        facility must be the least populous one and costs must exceed 1000.
        """
        assert min(small_instance.setup_costs.values()) > 1000.0

        from sklearn.datasets import fetch_california_housing

        data = fetch_california_housing()
        np.random.seed(42)
        indices = np.random.permutation(len(data.data))
        fac_indices = indices[:12]
        populations = data.data[fac_indices, 4]
        normalised = populations / np.median(populations)

        for f in small_instance.facilities:
            expected = 5000.0 * normalised[f] + 1000.0
            assert math.isclose(small_instance.setup_costs[f], expected, rel_tol=1e-9)

        # Ordering by population must equal ordering by setup cost.
        by_pop = sorted(small_instance.facilities, key=lambda f: populations[f])
        by_cost = sorted(small_instance.facilities, key=lambda f: small_instance.setup_costs[f])
        assert by_pop == by_cost

    def test_service_costs_are_non_negative_and_finite(self, small_instance):
        for u in small_instance.customers:
            for f in small_instance.facilities:
                cost = small_instance.service_costs[u][f]
                assert math.isfinite(cost)
                assert cost > 0.0

    def test_service_cost_equals_scaled_euclidean_degree_distance(self, small_instance):
        """
        service = ||(lat,lon)_c - (lat,lon)_f||_2 * 111.0 * 10.0.

        This pins the documented approximation: a degree of longitude is
        charged the same 111 km as a degree of latitude. At ~37 deg N a
        longitude degree is really ~88 km, so east-west separation is
        overstated by roughly 26%. The instance is therefore "geographically
        derived", not "real-world kilometres".
        """
        from sklearn.datasets import fetch_california_housing

        data = fetch_california_housing()
        coords = data.data[:, [6, 7]]
        np.random.seed(42)
        indices = np.random.permutation(len(coords))
        fac_indices = indices[:12]
        cust_indices = indices[12:21]

        for u in small_instance.customers:
            for f in small_instance.facilities:
                expected = (
                    float(np.linalg.norm(coords[cust_indices[u]] - coords[fac_indices[f]]))
                    * 111.0
                    * 10.0
                )
                assert math.isclose(
                    small_instance.service_costs[u][f], expected, rel_tol=1e-9
                )

        # Latitude column really is latitude (California spans ~32.5-42 N)
        # and longitude really is longitude (~-124.5 to -114).
        assert 32.0 <= coords[:, 0].min() and coords[:, 0].max() <= 42.5
        assert -125.0 <= coords[:, 1].min() and coords[:, 1].max() <= -114.0

    def test_service_costs_obey_the_triangle_inequality_via_shared_facility(self, small_instance):
        """
        Costs derive from a Euclidean embedding, so for any two customers and
        any two facilities the four pairwise costs must satisfy the quadrangle
        (triangle-derived) inequality. A transposed index or a scrambled row
        would break this.
        """
        sc = small_instance.service_costs
        customers = small_instance.customers[:4]
        facilities = small_instance.facilities[:4]
        for u1 in customers:
            for u2 in customers:
                for f1 in facilities:
                    for f2 in facilities:
                        # d(u1,f1) <= d(u1,f2) + d(u2,f2) + d(u2,f1)
                        assert sc[u1][f1] <= sc[u1][f2] + sc[u2][f2] + sc[u2][f1] + 1e-6


class TestLPRelaxationCharacter:
    def test_lp_relaxation_of_this_family_is_essentially_integral(self):
        """
        The published report attributes a 34% initial gap to a "highly
        fractional" LP relaxation. On this instance family the LP is in fact
        almost perfectly integral -- this test pins that, so the claim cannot
        be restated without the test going red.
        """
        from run_experiments import get_lp_bound_and_probs

        instance = create_california_uflp_instance(n_fac=60, n_cust=60)
        _bound, lp_probs = get_lp_bound_and_probs(instance)

        fractional = [f for f, v in lp_probs.items() if 0.01 < v < 0.99]
        assert len(fractional) <= 0.05 * len(instance.facilities), (
            f"expected a near-integral LP, got {len(fractional)} fractional "
            f"facilities out of {len(instance.facilities)}"
        )
