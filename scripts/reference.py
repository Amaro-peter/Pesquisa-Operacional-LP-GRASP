"""
Reference values — what a reported gap is actually measured against.

WHY THIS EXISTS
    Earlier revisions of this project reported every gap against the LP bound
    and called it an "optimality gap". That is only correct when the bound is
    attained by an integer solution. On families with a non-zero duality gap
    the LP bound sits strictly below the optimum, so a gap measured against it
    OVERSTATES the true optimality gap by an unknown amount -- and the word
    "optimality" is doing work the number cannot support.

    This module makes the distinction explicit and machine-checkable. A
    `Reference` knows whether its value is a proven optimum or merely a lower
    bound, and it names the gap accordingly. Reports ask the reference for the
    label rather than hardcoding one.

HOW A REFERENCE IS OBTAINED
    `compute_reference` tries to solve the integer program exactly with CBC
    under a time limit.

      * CBC proves optimality  -> `kind="proven_optimum"`, gaps are true
                                  optimality gaps.
      * CBC times out or fails -> the LP bound is used, `proven=False`, and
                                  every gap is labelled as measured against a
                                  lower bound.

    The LP bound is always retained alongside, so a report can state the
    bracket the true optimum lies in even when it is not known exactly.

READING THE SOLVER CORRECTLY
    PuLP's `LpStatus` is NOT sufficient to establish optimality. It reports
    "Optimal" whenever CBC terminates holding an integer-feasible solution --
    including when CBC stopped on its time limit with an incumbent it had not
    finished proving. Measured on a Koerkel-Ghosh 150x150 instance:

        time limit  LpStatus     sol_status              objective
        3 s         'Optimal'    2 'Solution Found'       156,287
        30 s        'Optimal'    2 'Solution Found'       156,260   <- better!
        (none)      'Optimal'    1 'Optimal Solution Found'    ...

    Two "Optimal" answers disagreeing by 27 units is proof enough that the flag
    does not mean what its name suggests. `sol_status == 1` is the one that
    does, and it is what this module requires.

    A second, solver-agnostic guard backs it up: `validated_against` demotes any
    claimed optimum that a feasible solution actually beats. Whatever a solver
    asserts, a value something beats was never a minimum.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, replace
from typing import Optional

import pulp

from scripts.uflp_solver import UFLPInstance

# PuLP's LpSolution[1]. Anything else means the solver stopped without proving
# optimality, however its LpStatus is spelled.
PROVEN_OPTIMAL = "Optimal Solution Found"


@dataclass
class Reference:
    """The value gaps are measured against, and what kind of value it is."""

    value: float
    lp_bound: float
    proven: bool
    ip_status: str
    ip_seconds: float
    incumbent: Optional[float] = None      # best integer solution CBC found, if any

    @property
    def kind(self) -> str:
        return "proven_optimum" if self.proven else "lp_bound"

    @property
    def gap_label(self) -> str:
        """How a gap against this reference may honestly be described."""
        return "optimality gap" if self.proven else "gap vs LP bound"

    @property
    def gap_label_short(self) -> str:
        return "opt. gap" if self.proven else "gap vs bound"

    def gap(self, cost: float) -> float:
        """Percentage of `cost` above the reference value."""
        return (cost - self.value) / self.value * 100.0

    def validated_against(self, best_known_cost: float,
                          tol: float = 1e-6) -> "Reference":
        """
        Demote a claimed optimum that a feasible solution beats.

        Solvers can and do report an unproven incumbent as optimal. A feasible
        solution cheaper than the claimed optimum refutes the claim outright,
        so the reference falls back to the LP bound -- the only value still
        known to be a valid lower bound -- and says why.

        `tol` absorbs the float noise of summing the same costs in a different
        order; it is far below any difference that could indicate a real
        refutation.
        """
        if not self.proven or best_known_cost >= self.value - tol:
            return self
        return replace(
            self,
            value=self.lp_bound,
            proven=False,
            ip_status=(f"rejected: feasible solution {best_known_cost:,.2f} beats "
                       f"claimed optimum {self.value:,.2f}"),
            incumbent=self.value,
        )

    def describe(self) -> str:
        """One line a report can print verbatim."""
        if self.proven:
            return (
                f"proven integer optimum {self.value:,.2f} "
                f"(CBC, {self.ip_seconds:.1f}s); gaps below are true optimality gaps"
            )
        return (
            f"LP bound {self.lp_bound:,.2f} (integer optimum not proven: CBC "
            f"{self.ip_status} after {self.ip_seconds:.1f}s); gaps below are measured "
            f"against a lower bound and therefore OVERSTATE the true optimality gap"
        )


def solve_exact_ip(
    instance: UFLPInstance,
    time_limit: Optional[float] = None,
) -> tuple[str, str, Optional[float], float]:
    """
    Solve the UFLP integer program with CBC.

    Returns `(status, solution_status, objective, seconds)`. Both statuses are
    returned because only the second one establishes optimality -- see the
    module docstring. The assignment variables are left
    continuous in [0, 1]: for the uncapacitated problem an optimal solution
    assigns each customer wholly to its cheapest open facility, so only the
    facility-opening variables need to be binary. This shrinks the model
    substantially without changing the optimum.
    """
    F = instance.facilities
    U = instance.customers
    c = instance.setup_costs
    d = instance.service_costs

    model = pulp.LpProblem("uflp_exact", pulp.LpMinimize)
    y = {f: pulp.LpVariable(f"y_{f}", cat="Binary") for f in F}
    x = {
        u: {f: pulp.LpVariable(f"x_{u}_{f}", lowBound=0, upBound=1) for f in F}
        for u in U
    }

    model += (
        pulp.lpSum(c[f] * y[f] for f in F)
        + pulp.lpSum(d[u][f] * x[u][f] for u in U for f in F)
    )
    for u in U:
        model += pulp.lpSum(x[u][f] for f in F) == 1
        for f in F:
            model += x[u][f] <= y[f]

    solver = pulp.PULP_CBC_CMD(msg=0, timeLimit=time_limit)
    t0 = time.perf_counter()
    model.solve(solver)
    elapsed = time.perf_counter() - t0

    status = pulp.LpStatus[model.status]
    solution_status = pulp.LpSolution[model.sol_status]
    objective = pulp.value(model.objective)
    return status, solution_status, objective, elapsed


def compute_reference(
    instance: UFLPInstance,
    lp_bound: float,
    time_limit: Optional[float] = 300.0,
    attempt_exact: bool = True,
) -> Reference:
    """
    Determine what gaps on this instance should be measured against.

    Set `attempt_exact=False` for instances where the integer program is known
    to be out of reach (the 2000x2000 California instance has four million
    binary-linked assignment variables); the LP bound is then used directly and
    labelled as such, without burning the time limit first.
    """
    if not attempt_exact:
        return Reference(
            value=lp_bound,
            lp_bound=lp_bound,
            proven=False,
            ip_status="not attempted",
            ip_seconds=0.0,
        )

    try:
        status, solution_status, objective, elapsed = solve_exact_ip(
            instance, time_limit=time_limit)
    except Exception as exc:  # pragma: no cover - solver-environment failures
        return Reference(
            value=lp_bound,
            lp_bound=lp_bound,
            proven=False,
            ip_status=f"error: {type(exc).__name__}",
            ip_seconds=0.0,
        )

    # "Optimal" alone is not proof: PuLP reports it for a time-limited stop
    # that merely holds an incumbent. Only PROVEN_OPTIMAL says CBC closed it.
    proven = (status == "Optimal" and solution_status == PROVEN_OPTIMAL
              and objective is not None)
    if proven:
        # Guard against a solver returning something below the relaxation's own
        # bound, which would indicate a malformed model rather than a better
        # solution.
        if objective < lp_bound - 1e-6:
            return Reference(
                value=lp_bound,
                lp_bound=lp_bound,
                proven=False,
                ip_status=f"rejected: IP objective {objective:,.2f} below LP bound",
                ip_seconds=elapsed,
                incumbent=objective,
            )
        return Reference(
            value=objective,
            lp_bound=lp_bound,
            proven=True,
            ip_status=status,
            ip_seconds=elapsed,
            incumbent=objective,
        )

    # Say which of the two flags refused the claim, so a reader can tell a
    # genuine failure ("Not Solved") from an unproven incumbent.
    unproven_status = (f"not proven ({solution_status})"
                       if status == "Optimal" else status)
    return Reference(
        value=lp_bound,
        lp_bound=lp_bound,
        proven=False,
        ip_status=unproven_status,
        ip_seconds=elapsed,
        incumbent=objective,
    )
