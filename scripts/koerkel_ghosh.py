"""
Körkel-Ghosh (KG) benchmark instances for the UFLP.

WHY THIS FAMILY
    Every instance family already in this repository has a near-integral LP
    relaxation: 7 fractional facilities out of 2000 on the California instance,
    an exactly integral bound on OR-Library's cap134, and a fully integral
    relaxation on six of the ten synthetic duality-gap instances. That makes
    them a poor test bed for a method whose whole premise is that the *fractional*
    LP solution carries useful information — there is barely any fractionality to
    exploit.

    The KG family is the standard hard case. Its allocation costs are drawn at
    random rather than from a metric embedding, which destroys the geometric
    structure that makes the UFLP relaxation tight, and its relaxations are
    strongly fractional. Fischetti et al. report that 50 of the 90 KG instances
    remained out of reach of exact methods.

GENERATION SPECIFICATION
    Taken from Karapetyan & Goldengorin, "Conditional Markov Chain Search for
    the Simple Plant Location Problem improves upper bounds on twelve
    Körkel-Ghosh instances" (arXiv:1711.06347), which states the library's
    construction directly:

        "The KG instance library includes three classes of instances, namely,
         A, B and C. In class A, the fixed costs f_i are drawn uniformly from
         [100, 200], in class B -- from [1000, 2000], and in class C -- from
         [10000, 20000]. The transportation costs c_ij are always drawn
         uniformly from [1000, 2000]. Symmetric and asymmetric instances are
         included, where symmetric instances satisfy c_ij = c_ji. The KG library
         includes instances of size m x n = 250 x 250, m x n = 500 x 500 and
         m x n = 750 x 750."

    Costs are integers.

PROVENANCE -- READ THIS BEFORE COMPARING TO PUBLISHED RESULTS
    These instances are GENERATED TO THE PUBLISHED SPECIFICATION. They are NOT
    the official UflLib instance files (gs250a-1 ... ga750c-5): those are served
    from hosts that refused every request from this environment (HTTP 403 from
    resources.mpi-inf.mpg.de, a redirect loop from the Frankfurt mirror).

    Consequently the objective values here are NOT comparable with published KG
    results, and best-known upper bounds from the literature do not apply to
    them. What they are valid for is the comparison this module exists to
    support: several solvers run on identical instances, with the LP bound
    computed exactly for each. That comparison is unaffected by whether the
    random draws match the official files.
"""

from __future__ import annotations

import random
from typing import Dict, List, Tuple

from scripts.uflp_solver import UFLPInstance

# Fixed-cost range per class, per the specification quoted above.
FIXED_COST_RANGES: Dict[str, Tuple[int, int]] = {
    "a": (100, 200),
    "b": (1000, 2000),
    "c": (10000, 20000),
}

# Allocation costs are drawn from this range for every class.
ALLOCATION_COST_RANGE: Tuple[int, int] = (1000, 2000)

# The sizes the official library ships.
LIBRARY_SIZES: Tuple[int, ...] = (250, 500, 750)


def instance_name(size: int, klass: str, symmetric: bool, index: int) -> str:
    """
    Name an instance the way the library does: gs250a-1, ga750c-5, ...

    'gs' = symmetric allocation matrix, 'ga' = asymmetric.
    """
    _validate_class(klass)
    return f"{'gs' if symmetric else 'ga'}{size}{klass}-{index}"


def _validate_class(klass: str) -> None:
    if klass not in FIXED_COST_RANGES:
        raise ValueError(
            f"unknown Körkel-Ghosh class {klass!r}; expected one of "
            f"{sorted(FIXED_COST_RANGES)}"
        )


def generate_koerkel_ghosh_instance(
    size: int = 250,
    klass: str = "a",
    symmetric: bool = True,
    seed: int = 1,
) -> UFLPInstance:
    """
    Build a Körkel-Ghosh instance to the published specification.

    Args:
        size: number of facilities, which equals the number of customers.
        klass: fixed-cost class, 'a' (small), 'b' (medium) or 'c' (large).
        symmetric: whether the allocation matrix satisfies c_ij = c_ji.
        seed: seeds a private RNG, so generation is reproducible and does not
            disturb the global `random` state the heuristics rely on.

    Returns:
        A `UFLPInstance` with integer-valued costs stored as floats.
    """
    _validate_class(klass)
    if size < 1:
        raise ValueError(f"size must be at least 1, got {size}")

    rng = random.Random(seed)
    facilities: List[int] = list(range(size))
    customers: List[int] = list(range(size))

    lo, hi = FIXED_COST_RANGES[klass]
    setup_costs = {f: float(rng.randint(lo, hi)) for f in facilities}

    c_lo, c_hi = ALLOCATION_COST_RANGE
    service_costs: Dict[int, Dict[int, float]] = {u: {} for u in customers}

    if symmetric:
        # Draw the upper triangle (including the diagonal) and mirror it, so
        # that c_ij == c_ji exactly.
        for i in range(size):
            for j in range(i, size):
                value = float(rng.randint(c_lo, c_hi))
                service_costs[i][j] = value
                service_costs[j][i] = value
    else:
        for u in customers:
            row = service_costs[u]
            for f in facilities:
                row[f] = float(rng.randint(c_lo, c_hi))

    return UFLPInstance(
        facilities=facilities,
        customers=customers,
        setup_costs=setup_costs,
        service_costs=service_costs,
    )
