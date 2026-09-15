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

import scripts.run_experiments as run_experiments


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


class TestGeneratedArtifactsLandInOutputDir:
    """
    Every generated artifact belongs in `output/`, and the path a script prints
    must be the path it actually wrote.

    `run_scaling_benchmark` used to write `scaling_results.md` and
    `scaling_analysis.png` into the repository root while `run_experiments`
    wrote into `output/`, and `run_experiments` printed "Results written to:
    cap134.md" while actually writing `output/cap134.md`. Both are the same
    defect in different directions: a reader cannot tell where an artifact came
    from, which is the failure mode this whole audit exists to prevent.
    """

    @staticmethod
    def _source(module_name):
        import pathlib

        return (
            pathlib.Path(__file__).resolve().parents[2] / "scripts" / f"{module_name}.py"
        ).read_text()

    @pytest.mark.parametrize(
        "module", ["run_experiments", "run_scaling_benchmark", "run_cap134"]
    )
    def test_scripts_declare_an_output_dir_and_use_it(self, module):
        src = self._source(module)
        assert 'OUTPUT_DIR = "output"' in src, f"{module} has no OUTPUT_DIR constant"
        # Either the constant directly, or an --out-dir argument defaulting to it.
        assert ("os.makedirs(OUTPUT_DIR" in src
                or "os.makedirs(args.out_dir" in src), f"{module} never creates its output dir"

    @pytest.mark.parametrize(
        "module,artifacts",
        [
            # run_experiments' cap134 comparison is superseded by run_cap134.py,
            # so its outputs carry a distinct name and the two cannot be confused.
            ("run_experiments", ["duality_gap_suite.md", "duality_gap_suite.png"]),
            ("run_cap134", ["cap134_results.md", "cap134_results.json"]),
            ("run_scaling_benchmark", ["scaling_results.md", "scaling_analysis.png"]),
        ],
    )
    def test_no_artifact_is_written_to_a_bare_relative_path(self, module, artifacts):
        """
        A bare `open("scaling_results.md", "w")` or `savefig("x.png")` writes to
        the working directory, not to output/. The filename may only appear as
        an argument to os.path.join.
        """
        src = self._source(module)
        for name in artifacts:
            bare_write = f'open("{name}"'
            bare_savefig = f'savefig("{name}"'
            assert bare_write not in src, f"{module} writes {name} to a bare path"
            assert bare_savefig not in src, f"{module} saves {name} to a bare path"
            assert f'"{name}"' in src, f"{module} no longer produces {name} at all"

    def test_counterweight_the_instance_path_is_still_an_input_not_an_output(self):
        """
        COUNTERWEIGHT: the OUTPUT_DIR change must not have swept up the *input*
        instance path. `data/cap134.txt` is read, not written, and must stay
        outside output/.
        """
        src = self._source("run_experiments")
        assert '"data/cap134.txt"' in src
        assert 'os.path.join(OUTPUT_DIR, "cap134.txt")' not in src


class TestSupersededArtifactsCannotBeConfused:
    """
    `run_cap134.py` reports cap134 as a multistart benchmark against the proven
    optimum. `run_experiments.py` still contains an older single-start cap134
    comparison, and for a while both wrote `output/cap134.md` -- so whichever
    script ran last silently decided what "the cap134 result" was.

    Two artifacts describing the same instance with different methodologies
    must not share a filename.
    """

    @staticmethod
    def _source(name):
        import pathlib

        return (pathlib.Path(__file__).resolve().parents[2] / "scripts" / f"{name}.py").read_text()

    def test_the_two_cap134_producers_write_different_files(self):
        legacy = self._source("run_experiments")
        current = self._source("run_cap134")

        assert "duality_gap_suite.md" in legacy
        assert "cap134.md" not in legacy, "the legacy script reclaimed the superseded name"
        assert "cap134_results.md" in current
        assert "duality_gap_suite" not in current

    def test_the_supersession_is_documented_where_the_name_is_chosen(self):
        """COUNTERWEIGHT: a rename with no explanation invites a later revert."""
        legacy = self._source("run_experiments")
        i = legacy.index("duality_gap_suite.md")
        context = legacy[max(0, i - 500):i]
        assert "SUPERSEDED" in context or "superseded" in context
        assert "run_cap134" in context
