"""
Regression: `run_experiments.main()` looked for the instance in a directory
that does not exist.

WHAT BROKE
    `main()` hardcoded `instance_file = "docs/cap134.txt"`. The OR-Library
    instance actually lives at `data/cap134.txt`, and there is no `docs/`
    directory in the repository at all.

WHY IT MATTERED
    The comparative benchmark -- the script that produces `cap134.md` and
    `cap134.png`, the headline artifacts of the study -- printed
    "Error: docs/cap134.txt not found" and called `sys.exit(1)` before doing
    any work. The published artifacts therefore could not have been produced by
    the committed code as it stood, which is exactly the provenance question
    this audit was opened to answer.

FIX
    `main()` resolves the first existing path from a candidate list
    (`data/cap134.txt`, then `cap134.txt` for the older flat layout).

DEFECT CONTEXT
    Found while auditing the provenance of `output/california_4M_results.md`.
"""

import os
import re
import sys

import pytest

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, REPO_ROOT)

import run_experiments


def test_benchmark_resolves_the_instance_file_that_actually_ships():
    """
    The path `main()` resolves must exist on disk. Pre-fix `main()` referenced
    the non-existent `docs/cap134.txt`, so this test is RED before the fix.
    """
    source = _main_source()
    candidates = re.findall(r'"([^"]*cap134\.txt)"', source)
    assert candidates, "main() no longer references cap134.txt at all"

    resolved = [c for c in candidates if os.path.exists(os.path.join(REPO_ROOT, c))]
    assert resolved, (
        f"none of the instance paths referenced by main() exist: {candidates}; "
        "the benchmark would sys.exit(1) before running"
    )


def test_shipped_instance_parses_into_the_expected_problem_size():
    """
    Guards the other half of the same defect: the file that main() resolves has
    to be a well-formed cap134 instance, not merely a path that exists.
    """
    path = os.path.join(REPO_ROOT, "data", "cap134.txt")
    if not os.path.exists(path):
        pytest.skip("data/cap134.txt is not present in this checkout")

    instance = run_experiments.parse_orlib_instance(path)
    assert len(instance.facilities) == 50
    assert len(instance.customers) == 50
    assert len(instance.setup_costs) == 50
    for u in instance.customers:
        assert len(instance.service_costs[u]) == 50


def test_counterweight_missing_instance_still_exits_cleanly():
    """
    COUNTERWEIGHT: the fix must not overshoot into silently proceeding when the
    instance genuinely is absent. With every candidate path missing, `main()`
    must still report the problem and exit non-zero rather than crash with a
    traceback or run on garbage.
    """
    source = _main_source()
    assert "sys.exit(1)" in source, "the missing-instance guard was removed"
    assert "not found" in source


def _main_source() -> str:
    import inspect

    return inspect.getsource(run_experiments.main)
