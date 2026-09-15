"""
Regression: the results report asserted conclusions its own table contradicted.

WHAT BROKE
    `download_and_run_real_world.py` built `output/california_4M_results.md`
    from an f-string template with the comparative wording written *into* it:

        | **Initial Constructive Cost** | {lp_cost} | {alpha_cost} | LP-biased starts lower |
        | **Initial Optimality Gap** | ... | LP-biased gap is {...:.4f}% tighter |

    The run it described produced the opposite: the LP-biased arm started at a
    34.20% gap against alpha-GRASP's 18.07%. The template said "lower" and
    "tighter" regardless of what was measured.

WHY IT MATTERED
    Someone noticed the contradiction and, instead of fixing the template,
    wrote `fix_markdown.py` -- a script that hardcodes the numbers as literals
    and overwrites the results file with a hand-authored narrative. The
    published artifact was therefore no longer the output of the program that
    computed it, two of its figures were back-computed from rounded percentages
    rather than measured, and re-running the generator would have produced a
    file whose prose contradicted its own table again.

    A report that can state a conclusion its data does not support is not a
    report; it is a template with numbers pasted in.

FIX
    `render_report` derives every comparative phrase from the measured values,
    so the wording always follows the data. These tests drive the renderer with
    mirror-image datasets and require the wording to flip.

DEFECT CONTEXT
    Found while auditing the provenance of `output/california_4M_results.md`.
"""

import ast
import os
import pathlib
import sys

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

GENERATOR = (
    pathlib.Path(__file__).resolve().parents[2] / "scripts" / "download_and_run_real_world.py"
)

ENV = {
    "python": "3.14.6", "numpy": "2.5.3", "scipy": "1.18.1",
    "scikit_learn": "1.9.1", "platform": "test", "cpu_count": "12",
}

REF = 100_000.0


def _lp(n_fractional=7, bound=REF, solve_seconds=60.0):
    from scripts.download_and_run_real_world import LPProfile

    return LPProfile(
        bound=bound, solve_seconds=solve_seconds, n_facilities=2000,
        n_integral_open=54, n_fractional=n_fractional,
        n_near_zero=2000 - 54 - n_fractional, sum_y=57.5, rounded_open_count=61,
    )


def _reference(proven=False, bound=REF, value=None):
    from scripts.reference import Reference

    return Reference(
        value=value if value is not None else bound,
        lp_bound=bound, proven=proven,
        ip_status="Optimal" if proven else "not attempted",
        ip_seconds=0.0,
    )


def _det(gap_pct, iterations=5, lp_seconds=60.0, construct=9.0, search=50.0):
    from scripts.download_and_run_real_world import ArmResult

    return ArmResult(
        method="det", seed=None,
        initial_cost=REF * 1.3, initial_gap=30.0, initial_open=75,
        final_cost=REF * (1 + gap_pct / 100), final_gap=gap_pct, final_open=58,
        iterations=iterations, moves={"insert": 0, "delete": 3, "swap": 2},
        construct_seconds=construct, search_seconds=search, lp_seconds=lp_seconds,
    )


def _ms(trajectory_gaps, lp_seconds=60.0, search=200.0, per_restart=None):
    from scripts.multistart import MultistartResult

    costs = [REF * (1 + g / 100) for g in trajectory_gaps]
    best = min(trajectory_gaps)
    return MultistartResult(
        method="ms", seeds=list(range(1, len(trajectory_gaps) + 1)),
        best_cost=REF * (1 + best / 100), best_gap=best, best_open=58,
        best_found_at=trajectory_gaps.index(best) + 1,
        trajectory=list(trajectory_gaps), trajectory_costs=costs,
        per_restart_gaps=list(per_restart if per_restart else trajectory_gaps),
        total_iterations=200, moves={"insert": 0, "delete": 17, "swap": 3},
        construct_seconds=9.0, search_seconds=search, lp_seconds=lp_seconds,
    )


def _render(*, ap=0.10, a=1.00, d=0.30, b=(1.0, 0.20), c=(2.0, 0.40),
            n_fractional=7, proven=False, alpha_value=0.2):
    from scripts.download_and_run_real_world import render_report

    return render_report(
        n_fac=2000, n_cust=2000, lp=_lp(n_fractional=n_fractional),
        reference=_reference(proven=proven),
        control=_det(a, iterations=0), rounding_ls=_det(ap), no_lp=_det(d),
        lp_biased=_ms(list(b)), alpha=_ms(list(c)),
        seeds=list(range(1, len(b) + 1)), env=ENV,
        generated_at="2026-09-12 00:00 UTC", commit="testcommit",
        wall_seconds=1.0, alpha_value=alpha_value,
    )


def _prose(md):
    stripped = [ln.lstrip().removeprefix("> ").removeprefix(">") for ln in md.splitlines()]
    return " ".join(" ".join(stripped).split())


# ==========================================================================
# The head-to-head verdict must follow the data
# ==========================================================================

class TestHeadToHeadWordingFollowsTheData:
    def test_lp_biased_winning_is_said_plainly(self):
        md = _prose(_render(b=(1.0, 0.10), c=(2.0, 0.50)))
        assert "The LP-biased construction wins the head-to-head" in md
        assert "classical baseline wins" not in md

    def test_baseline_winning_is_said_plainly(self):
        """COUNTERWEIGHT: the baseline must be able to win, and be said to."""
        md = _prose(_render(b=(1.0, 0.50), c=(2.0, 0.10)))
        assert "The classical baseline wins the head-to-head" in md
        assert "LP-biased construction wins" not in md

    def test_a_tie_is_said_plainly(self):
        md = _prose(_render(b=(1.0, 0.25), c=(2.0, 0.25)))
        assert "The two constructions tie at this restart budget" in md


# ==========================================================================
# The multistart verdict must follow the data
# ==========================================================================

class TestMultistartVerdictFollowsTheData:
    def test_multistart_overturning_the_single_start_finding_is_said_plainly(self):
        """
        The branch that would reverse this project's earlier conclusion. It
        must be reachable and must name the earlier result as an artifact.
        """
        md = _prose(_render(ap=0.50, b=(1.0, 0.05)))
        assert "Multistart overturns the single-start result" in md
        assert "an artifact of running it once" in md
        assert "still holds up" not in md

    def test_deterministic_arm_holding_up_is_said_plainly(self):
        md = _prose(_render(ap=0.05, b=(1.0, 0.30)))
        assert "The deterministic arm still holds up" in md
        assert "without" in md and "overturning it at this budget" in md
        assert "overturns the single-start result" not in md

    def test_an_exact_tie_is_reported_as_level_at_n_times_the_work(self):
        md = _prose(_render(ap=0.20, b=(1.0, 0.20)))
        assert "level with the deterministic one" in md
        assert "× the work" in md

    def test_restarts_needed_to_match_are_measured_not_asserted(self):
        md = _prose(_render(ap=0.50, b=(2.0, 1.0, 0.40)))
        assert "after **3** restarts" in md

    def test_never_matching_within_budget_is_said_plainly(self):
        md = _prose(_render(ap=0.01, b=(2.0, 1.0, 0.40)))
        assert "never matches A+" in md
        assert "restart budget" in md


# ==========================================================================
# What the LP contributes must follow the data
# ==========================================================================

class TestLpContributionFollowsTheData:
    def test_the_lp_contributing_is_said_plainly(self):
        md = _prose(_render(ap=0.10, d=0.90))
        assert "The LP contributes" in md
        assert "not contributing here" not in md

    def test_the_lp_not_contributing_is_said_plainly(self):
        """COUNTERWEIGHT: the unflattering direction must be reachable."""
        md = _prose(_render(ap=0.90, d=0.10))
        assert "The LP is not contributing here" in md

    def test_no_measurable_difference_is_said_plainly(self):
        md = _prose(_render(ap=0.30, d=0.30))
        assert "The LP makes no measurable difference here" in md


# ==========================================================================
# Integrality wording, and the reference label
# ==========================================================================

class TestIntegralityAndReference:
    def test_a_near_integral_relaxation_is_not_called_fractional(self):
        md = _prose(_render(n_fractional=7))
        assert "essentially integral" in md
        assert "highly fractional" not in md
        assert "0.35%" in md

    def test_a_genuinely_fractional_relaxation_is_called_fractional(self):
        """COUNTERWEIGHT: the fix must not hardcode 'integral' either."""
        md = _prose(_render(n_fractional=1200))
        assert "highly fractional" in md
        assert "essentially integral" not in md

    def test_an_unproven_reference_forbids_calling_gaps_optimality_gaps(self):
        md = _prose(_render(proven=False))
        assert "gap vs LP bound" in md
        assert "measured against the LP bound, not a proven optimum" in md
        assert "OVERSTATE" in md

    def test_a_proven_reference_changes_the_label(self):
        md = _prose(_render(proven=True))
        assert "proven integer optimum" in md
        assert "optimality gap" in md


# ==========================================================================
# No hardcoded verdicts survive in the generator source
# ==========================================================================

def _source_without_docstrings() -> str:
    """
    The generator quotes the discredited sentences in its docstrings to explain
    the defect it fixed. Documenting a defect is not committing it.
    """
    text = GENERATOR.read_text()
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


class TestTemplateCarriesNoHardcodedVerdict:
    def test_generator_source_contains_no_hardcoded_comparative_verdicts(self):
        """RED against the pre-fix generator, which embedded these verbatim."""
        src = _source_without_docstrings()
        forbidden = [
            "LP-biased starts lower",
            "LP-biased gap is {",
            "LP-biased is faster by",
            "LP-biased finishes closer",
            "LP-biased reduces workload",
        ]
        found = [p for p in forbidden if p in src]
        assert not found, f"generator hardcodes comparative verdicts: {found}"

    def test_generator_still_documents_the_defect_it_fixed(self):
        """COUNTERWEIGHT: the scan must ignore docstrings."""
        doc = ast.get_docstring(ast.parse(GENERATOR.read_text())) or ""
        assert "PROVENANCE CONTRACT" in doc

    def test_generator_exposes_a_pure_renderer(self):
        import inspect

        from scripts.download_and_run_real_world import render_report

        params = inspect.signature(render_report).parameters
        for required in ("lp", "reference", "control", "rounding_ls", "no_lp",
                         "lp_biased", "alpha", "seeds"):
            assert required in params, f"render_report is missing {required!r}"

    def test_every_writer_of_the_results_file_goes_through_the_renderer(self):
        """
        The results file may have several write sites, but every one must get
        its text from `render_report`. Writing it from anything else is the
        `fix_markdown.py` pattern.
        """
        src = GENERATOR.read_text()
        assert "fix_markdown" not in src, "a post-processing step was re-introduced"

        tree = ast.parse(src)
        writers = [
            (node.name, ast.get_source_segment(src, node) or "")
            for node in ast.walk(tree)
            if isinstance(node, ast.FunctionDef)
            and "california_4M_results.md" in (ast.get_source_segment(src, node) or "")
        ]
        assert writers, "nothing writes the results file any more"
        for name, body in writers:
            assert "render_report" in body, (
                f"{name}() writes the results file without going through render_report"
            )


class TestNoBackComputedFigures:
    def test_reported_costs_are_the_measured_costs(self):
        """
        The superseded artifact derived its initial-cost cells as
        `bound * (1 + rounded_gap)`, losing precision. The renderer must print
        the cost it was handed.
        """
        from scripts.download_and_run_real_world import render_report

        arm = _det(0.0421)
        arm.final_cost = 397_767.97
        md = render_report(
            n_fac=2000, n_cust=2000, lp=_lp(), reference=_reference(),
            control=_det(1.0328, iterations=0), rounding_ls=arm, no_lp=_det(0.30),
            lp_biased=_ms([0.5, 0.0421]), alpha=_ms([1.0, 0.3444]),
            seeds=[1, 2], env=ENV, generated_at="x", commit="testcommit",
            wall_seconds=1.0,
        )
        assert "397,767.97" in md
        assert "533,580.24" not in md      # the back-computed value it used to print

    def test_report_records_its_own_provenance(self):
        md = _prose(_render())
        assert "render_report" in md
        assert "testcommit" in md
        assert "california_4M_results.json" in md
        assert "--from-json" in md
