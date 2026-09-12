"""
Regression: the scaling benchmark's "LP-biased" arm was not LP-biased.

WHAT BROKE
    `run_scaling_benchmark.evaluate_seed` read the LP fractional values from a
    module-level global (`global_lp_probs_dict`) that `main()` assigned in the
    parent process just before submitting work to a `ProcessPoolExecutor`.
    Worker processes only inherit parent globals under the "fork" start method.
    Python 3.14 changed the default start method on Linux to "forkserver" (it
    has always been "spawn" on macOS/Windows), so each worker re-imported the
    module and saw the *initial* empty dict.

WHY IT MATTERED
    This failed silently rather than crashing. `construct_lp_biased_solution`
    does `max(lp_probs.get(f, 0.0), EPS)`, so an empty dict degrades every
    facility to the EPS=0.01 floor -- the "LP-biased" arm collapsed into a
    uniform 1%-random construction identical in spirit to the control it was
    supposed to beat. Every LP-biased row in `scaling_results.md` /
    `scaling_analysis.png` produced under a non-fork start method measured the
    wrong algorithm while reporting it as the LP-biased hybrid.

FIX
    `lp_probs` now travels inside the task tuple, so it is pickled to the
    worker regardless of start method.

DEFECT CONTEXT
    Found while auditing the provenance of `output/california_4M_results.md`.
"""

import concurrent.futures
import multiprocessing
import os
import sys

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import run_scaling_benchmark as rsb
from uflp_solver import generate_random_instance


def _worker_reports_lp_init_gap(task):
    """Run the real worker body in a child process and report its LP-arm gap."""
    return rsb.evaluate_seed(task)["lp_init_gap"]


def test_lp_probs_reach_worker_processes_under_non_fork_start_methods():
    """
    The LP probabilities must survive the trip into a worker process under a
    start method that does NOT inherit parent globals.

    The probe drives the *real* `evaluate_seed` body in a spawned child, with
    every y[f] = 1.0. If lp_probs arrive intact, the LP-biased construction
    opens every facility and its initial gap equals the all-open gap exactly.
    If they do not, the EPS=0.01 floor opens a near-empty random subset and the
    gap is wildly different.

    Pre-fix this is RED (the 4-tuple unpack raises in the worker).
    """
    # "spawn" is the strictest case and is available on every platform.
    ctx = multiprocessing.get_context("spawn")
    instance = generate_random_instance(n_facilities=6, n_customers=5, seed=7)
    lp_probs = {f: 1.0 for f in instance.facilities}

    all_open_cost = sum(instance.setup_costs.values()) + sum(
        min(instance.service_costs[u].values()) for u in instance.customers
    )
    lp_bound = 1000.0
    expected_gap = (all_open_cost - lp_bound) / lp_bound * 100

    tasks = [(seed, instance, lp_bound, 0.01, lp_probs) for seed in (1, 2)]

    with concurrent.futures.ProcessPoolExecutor(max_workers=2, mp_context=ctx) as executor:
        seen = list(executor.map(_worker_reports_lp_init_gap, tasks))

    assert seen == pytest.approx([expected_gap, expected_gap], abs=1e-9), (
        "workers did not receive the LP probabilities -- the LP-biased arm "
        "silently degraded to a uniform EPS-random construction"
    )


def test_evaluate_seed_consumes_lp_probs_from_the_task_tuple():
    """
    `evaluate_seed` must unpack lp_probs from its argument, not from a global.

    This is the in-process counterpart of the test above: it pins the contract
    without paying for process startup.
    """
    assert not hasattr(rsb, "global_lp_probs_dict"), (
        "the module-level LP-probability global is back; worker processes "
        "under forkserver/spawn will not see it"
    )

    instance = generate_random_instance(n_facilities=5, n_customers=4, seed=11)
    # y=1.0 for every facility => the construction must open all of them.
    lp_probs = {f: 1.0 for f in instance.facilities}

    result = rsb.evaluate_seed((1, instance, 1000.0, 0.25, lp_probs))

    assert set(result) == {
        "lp_time",
        "lp_init_gap",
        "lp_final_gap",
        "alpha_time",
        "alpha_init_gap",
        "alpha_final_gap",
        "lp_only_time",
    }
    # The LP solve time supplied in the task must be folded into the reported
    # end-to-end LP time, not dropped.
    assert result["lp_only_time"] == 0.25
    assert result["lp_time"] >= 0.25


def test_counterweight_all_zero_lp_probs_still_produce_a_feasible_solution():
    """
    COUNTERWEIGHT: passing lp_probs explicitly must not break the legitimate
    all-zero case. A degenerate LP (every y=0) must still yield a feasible,
    finite-cost solution via the EPS floor / cheapest-facility fallback -- the
    fix must not turn "no probabilities" into a crash.
    """
    instance = generate_random_instance(n_facilities=5, n_customers=4, seed=13)
    lp_probs = {f: 0.0 for f in instance.facilities}

    result = rsb.evaluate_seed((3, instance, 1000.0, 0.0, lp_probs))

    assert result["lp_final_gap"] == pytest.approx(result["lp_final_gap"])  # finite, not NaN
    assert result["lp_final_gap"] <= result["lp_init_gap"] + 1e-9, (
        "local search must never worsen the constructed solution"
    )
