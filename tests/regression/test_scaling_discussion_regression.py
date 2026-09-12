"""
Regression: the scaling report's discussion contradicted its own table.

WHAT BROKE
    `run_scaling_benchmark` wrote a "Discussion of Scaling Behavior" section as
    three hardcoded sentences. They claimed the LP-biased construction starts
    "under 1% across all sizes" and that alpha-GRASP starts "over 30% to 140%".
    The table printed directly above them, from the same run, showed LP-biased
    initial gaps of 0.00-10.51% and alpha-GRASP initial gaps of 1.86-19.56%.

WHY IT MATTERED
    This is the same defect that produced `fix_markdown.py`: prose asserted
    independently of the measurements, in a file whose only purpose is to
    report measurements. It is worse than no discussion, because a reader who
    trusts the prose is misled by a document that contains its own refutation
    two paragraphs earlier.

FIX
    `describe_scaling(rows)` derives every figure and every comparative word
    from the measured rows.

DEFECT CONTEXT
    Found while regenerating the scaling artifacts after the forkserver fix,
    during the same audit that produced the California and Körkel-Ghosh reports.
"""

import os
import pathlib
import re
import sys

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from scripts.run_scaling_benchmark import describe_scaling

SOURCE = pathlib.Path(__file__).resolve().parents[2] / "scripts" / "run_scaling_benchmark.py"


def _rows(lp_inits, alpha_inits, lp_finals, alpha_finals, lp_times, alpha_times):
    return [
        {
            "size": f"{i}x{i}", "vars": (i + 1) * 1000,
            "lp_time": lt, "lp_init_gap": li, "lp_final_gap": lf,
            "alpha_time": at, "alpha_init_gap": ai, "alpha_final_gap": af,
            "lp_only_time": lt / 2,
        }
        for i, (li, ai, lf, af, lt, at) in enumerate(
            zip(lp_inits, alpha_inits, lp_finals, alpha_finals, lp_times, alpha_times)
        )
    ]


class TestFiguresComeFromTheRows:
    def test_reported_ranges_are_the_measured_ranges(self):
        rows = _rows([0.0, 3.09, 10.51], [1.86, 16.41, 19.56],
                     [0.0, 0.0, 0.0], [0.0, 0.1736, 0.0],
                     [0.006, 0.032, 0.123], [0.001, 0.018, 0.105])
        text = " ".join(describe_scaling(rows))

        assert "**0.00%**" in text and "**10.51%**" in text      # LP-biased span
        assert "**1.86%**" in text and "**19.56%**" in text      # alpha span
        # The discredited hardcoded claims must be gone.
        assert "under 1%" not in text
        assert "over 30% to 140%" not in text

    def test_a_different_dataset_produces_different_numbers(self):
        """The sentences must track the data, not be fixed strings."""
        a = " ".join(describe_scaling(
            _rows([1.0], [2.0], [0.0], [0.0], [1.0], [2.0])))
        b = " ".join(describe_scaling(
            _rows([40.0], [50.0], [0.5], [0.9], [1.0], [2.0])))
        assert a != b
        assert "**40.00%**" in b and "**40.00%**" not in a


class TestComparativeWordsFollowTheData:
    def test_faster_and_slower_flip_with_the_measured_times(self):
        faster = " ".join(describe_scaling(
            _rows([1.0], [2.0], [0.0], [0.0], lp_times=[1.0], alpha_times=[4.0])))
        slower = " ".join(describe_scaling(
            _rows([1.0], [2.0], [0.0], [0.0], lp_times=[4.0], alpha_times=[1.0])))

        assert "4.00x faster" in faster and "slower" not in faster
        assert "0.25x slower" in slower and "faster" not in slower

    def test_win_loss_tie_counts_are_counted_not_asserted(self):
        rows = _rows([1, 1, 1], [2, 2, 2],
                     lp_finals=[0.1, 0.9, 0.5], alpha_finals=[0.9, 0.1, 0.5],
                     lp_times=[1, 1, 1], alpha_times=[2, 2, 2])
        text = " ".join(describe_scaling(rows))
        assert "better on 1 size, worse on 1, and tied on 1" in text

    def test_counterweight_the_baseline_winning_is_reported_plainly(self):
        """
        COUNTERWEIGHT: the section must be able to report that alpha-GRASP came
        out ahead. A discussion that can only flatter one arm is not a
        discussion.
        """
        rows = _rows([5, 5], [1, 1],
                     lp_finals=[0.9, 0.8], alpha_finals=[0.1, 0.2],
                     lp_times=[4, 4], alpha_times=[1, 1])
        text = " ".join(describe_scaling(rows))
        assert "better on 0 sizes, worse on 2" in text
        assert "slower" in text


def _source_without_docstrings() -> str:
    """
    Return the script's source with every docstring blanked out.

    `describe_scaling` quotes the discredited sentences in its own docstring in
    order to explain the defect it fixed. Documenting a defect is not
    committing it, so the scan below must see executable code and emitted
    strings only.
    """
    import ast

    text = SOURCE.read_text()
    lines = text.splitlines()
    tree = ast.parse(text)
    for node in ast.walk(tree):
        if not isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef,
                                 ast.AsyncFunctionDef)):
            continue
        body = getattr(node, "body", None)
        if not body:
            continue
        first = body[0]
        if (isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant)
                and isinstance(first.value.value, str)):
            for i in range(first.lineno - 1, first.end_lineno):
                lines[i] = ""
    return "\n".join(lines)


class TestNoHardcodedVerdictSurvivesInTheSource:
    def test_the_discredited_sentences_are_not_in_the_script(self):
        """RED against the pre-fix script, which embedded them verbatim."""
        src = _source_without_docstrings()
        for phrase in ["under 1%", "over 30% to 140%",
                       "converges almost instantly",
                       "making the LP-biased method faster at scale"]:
            assert phrase not in src, f"hardcoded claim still present: {phrase!r}"

    def test_the_discussion_is_produced_by_a_pure_function(self):
        """COUNTERWEIGHT: removing the prose is not enough; it must be computed."""
        import inspect

        params = inspect.signature(describe_scaling).parameters
        assert "rows" in params
        assert "for line in describe_scaling(detailed_rows):" in SOURCE.read_text()

    def test_counterweight_documenting_the_defect_is_still_allowed(self):
        """
        COUNTERWEIGHT to the scan above: it must ignore docstrings, so the fix
        can keep explaining what it fixed.
        """
        import ast

        doc = ast.get_docstring(ast.parse(SOURCE.read_text())) or ""
        fn_doc = describe_scaling.__doc__ or ""
        assert "under 1%" in fn_doc, "the defect explanation was lost"
        assert "under 1%" not in _source_without_docstrings()


class TestNegativeZeroGapFormatting:
    """
    A gap below the LP bound is impossible -- the bound is a valid lower bound.
    Sub-tolerance negatives are float noise from `(cost - bound) / bound` when
    the arm attains the bound exactly, and rendering them as "-0.0000%" reads
    like a defect. A genuinely negative gap, however, IS a defect and must stay
    visible rather than be quietly absorbed.
    """

    def test_float_noise_is_rendered_as_zero(self):
        from scripts.run_scaling_benchmark import fmt_gap

        assert fmt_gap(-1e-16) == "0.0000"
        assert fmt_gap(-0.0) == "0.0000"
        assert fmt_gap(0.0) == "0.0000"

    def test_a_real_negative_gap_is_not_hidden(self):
        """COUNTERWEIGHT: the clamp must not swallow an actual anomaly."""
        from scripts.run_scaling_benchmark import fmt_gap

        assert fmt_gap(-0.5) == "-0.5000"
        assert fmt_gap(-1e-3) == "-0.0010"

    def test_positive_gaps_are_unchanged(self):
        from scripts.run_scaling_benchmark import fmt_gap

        assert fmt_gap(0.1736) == "0.1736"
        assert fmt_gap(10.51, 2) == "10.51"

    def test_the_rendered_report_contains_no_negative_zero(self):
        import pathlib

        report = pathlib.Path(__file__).resolve().parents[2] / "output" / "scaling_results.md"
        if not report.exists():
            pytest.skip("scaling_results.md not generated in this checkout")
        assert "-0.0000%" not in report.read_text()
