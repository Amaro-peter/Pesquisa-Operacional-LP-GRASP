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
    `render_report` derives every comparative phrase from the measured values
    via `_cmp` / `_signed` / direction-branching, so the wording always follows
    the data. This test drives the renderer with two mirror-image datasets and
    asserts the wording flips.

DEFECT CONTEXT
    Found while auditing the provenance of `output/california_4M_results.md`.
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import pathlib

GENERATOR = (
    pathlib.Path(__file__).resolve().parents[2] / "scripts" / "download_and_run_real_world.py"
)

ENV = {
    "python": "3.14.6",
    "numpy": "2.5.3",
    "scipy": "1.18.1",
    "scikit_learn": "1.9.1",
    "platform": "test",
    "cpu_count": "12",
}


def _lp(bound=100_000.0, n_fractional=7):
    from scripts.download_and_run_real_world import LPProfile

    return LPProfile(
        bound=bound,
        solve_seconds=50.0,
        n_facilities=2000,
        n_integral_open=50,
        n_fractional=n_fractional,
        n_near_zero=1943,
        sum_y=57.5,
        rounded_open_count=61,
    )


def _arm(method, seed, init_gap, final_gap, iterations, moves,
         construct=10.0, search=100.0, lp_seconds=0.0, bound=100_000.0):
    from scripts.download_and_run_real_world import ArmResult

    return ArmResult(
        method=method,
        seed=seed,
        initial_cost=bound * (1 + init_gap / 100),
        initial_gap=init_gap,
        initial_open=75,
        final_cost=bound * (1 + final_gap / 100),
        final_gap=final_gap,
        final_open=58,
        iterations=iterations,
        moves=moves,
        construct_seconds=construct,
        search_seconds=search,
        lp_seconds=lp_seconds,
    )


def _render(lp_init_gap, al_init_gap, lp_final_gap, al_final_gap,
            lp_search, al_search, lp_construct, al_construct,
            ap_final_gap=0.55):
    from scripts.download_and_run_real_world import render_report

    lp = _lp()
    control = _arm("LP rounding (control)", None, 1.0, 1.0, 0,
                   {"insert": 0, "delete": 0, "swap": 0}, construct=0.1,
                   search=0.0, lp_seconds=lp.solve_seconds)
    rounding_ls = _arm("LP rounding + local search (deterministic ablation)", None, 1.0,
                       ap_final_gap, 5,
                       {"insert": 0, "delete": 3, "swap": 2}, construct=0.1,
                       search=40.0, lp_seconds=lp.solve_seconds)
    biased = [_arm("LP-biased hybrid GRASP", 42, lp_init_gap, lp_final_gap, 20,
                   {"insert": 0, "delete": 17, "swap": 3},
                   construct=lp_construct, search=lp_search,
                   lp_seconds=lp.solve_seconds)]
    alpha = [_arm("alpha-GRASP baseline (alpha=0.2)", 42, al_init_gap, al_final_gap, 36,
                  {"insert": 0, "delete": 9, "swap": 27},
                  construct=al_construct, search=al_search, lp_seconds=0.0)]
    return render_report(
        n_fac=2000, n_cust=2000, lp=lp, control=control, rounding_ls=rounding_ls,
        lp_biased=biased, alpha=alpha, seeds=[42], env=ENV,
        generated_at="2026-09-11 00:00 UTC", commit="testcommit", wall_seconds=1.0,
    )


class TestWordingFollowsTheData:
    def test_worse_initial_gap_is_not_described_as_lower(self):
        """
        The exact shape of the original defect: the LP-biased arm starts at a
        WORSE initial gap than the baseline. The report must say so.

        Pre-fix this is RED -- the template printed "LP-biased starts lower"
        unconditionally.
        """
        md = _render(lp_init_gap=34.2001, al_init_gap=18.0699,
                     lp_final_gap=0.0421, al_final_gap=0.3444,
                     lp_search=164.0, al_search=292.0,
                     lp_construct=9.3, al_construct=106.8)

        assert "B starts higher" in md
        assert "B starts lower" not in md

    def test_better_initial_gap_is_described_as_lower(self):
        """The mirror image: same renderer, flipped data, flipped wording."""
        md = _render(lp_init_gap=12.0, al_init_gap=25.0,
                     lp_final_gap=0.0421, al_final_gap=0.3444,
                     lp_search=164.0, al_search=292.0,
                     lp_construct=9.3, al_construct=106.8)

        assert "B starts lower" in md
        assert "B starts higher" not in md

    def test_tighter_and_looser_final_gaps_flip(self):
        tighter = _render(34.2, 18.1, lp_final_gap=0.0421, al_final_gap=0.3444,
                          lp_search=164.0, al_search=292.0,
                          lp_construct=9.3, al_construct=106.8)
        looser = _render(34.2, 18.1, lp_final_gap=0.9000, al_final_gap=0.3444,
                         lp_search=164.0, al_search=292.0,
                         lp_construct=9.3, al_construct=106.8)

        assert "B is tighter" in tighter and "B is looser" not in tighter
        assert "B is looser" in looser and "B is tighter" not in looser

    def test_faster_and_slower_totals_flip(self):
        faster = _render(34.2, 18.1, 0.04, 0.34,
                         lp_search=164.0, al_search=292.0,
                         lp_construct=9.3, al_construct=106.8)
        slower = _render(34.2, 18.1, 0.04, 0.34,
                         lp_search=400.0, al_search=100.0,
                         lp_construct=9.3, al_construct=10.0)

        assert "is **faster** overall" in faster
        assert "is **slower** overall" in slower

    def test_stage_terms_name_the_direction_they_actually_pull(self):
        """
        A stage where arm B spends MORE time must be reported as a cost, not
        folded into a blanket "the saving comes from ..." sentence.
        """
        md = _render(34.2, 18.1, 0.04, 0.34,
                     lp_search=400.0, al_search=100.0,   # B's search is slower
                     lp_construct=9.3, al_construct=106.8)

        assert "Local search — costs arm B" in md
        assert "Construction — saves arm B" in md
        # The LP is only ever paid by arm B, so it is always a cost.
        assert "LP relaxation — costs arm B" in md


class TestControlArmIsReported:
    def test_local_search_uplift_claim_follows_the_control(self):
        """
        The "local search earns its keep" claim must be conditional on the
        measured improvement of arm A+ over the control, not asserted.

        Arm A+ is the right arm to key on: it introduces no construction noise,
        so any gain it makes over the control is genuine search, not self-repair.
        The control's gap is fixed at 1.0% by the fixture.
        """
        helps = _render(34.2, 18.1, lp_final_gap=0.0421, al_final_gap=0.3444,
                        lp_search=164.0, al_search=292.0,
                        lp_construct=9.3, al_construct=106.8,
                        ap_final_gap=0.0421)
        assert "The local search earns its keep" in helps
        assert "in 5 iterations" in helps

        # A+ no better than the control it started from.
        no_help = _render(34.2, 18.1, lp_final_gap=1.5, al_final_gap=2.0,
                          lp_search=164.0, al_search=292.0,
                          lp_construct=9.3, al_construct=106.8,
                          ap_final_gap=1.0)
        assert "The local search does not improve on rounding" in no_help
        assert "The local search earns its keep" not in no_help


class TestIntegralityNarrative:
    def test_near_integral_relaxation_is_not_called_fractional(self):
        """
        7/2000 fractional is 0.35%. The report must not describe that as
        "highly fractional" -- the claim this whole audit started from.
        """
        md = _render(34.2, 18.1, 0.04, 0.34, 164.0, 292.0, 9.3, 106.8)
        assert "essentially integral" in md
        assert "highly fractional" not in md
        assert "0.35%" in md

    def test_genuinely_fractional_relaxation_is_called_fractional(self):
        """COUNTERWEIGHT: the fix must not hardcode 'integral' either."""
        lp = _lp(n_fractional=1200)  # 60% fractional
        control = _arm("LP rounding (control)", None, 1.0, 1.0, 0,
                       {"insert": 0, "delete": 0, "swap": 0},
                       construct=0.1, search=0.0, lp_seconds=lp.solve_seconds)
        rounding_ls = _arm("LP rounding + local search (deterministic ablation)", None, 1.0, 0.55, 5,
                           {"insert": 0, "delete": 3, "swap": 2},
                           construct=0.1, search=40.0, lp_seconds=lp.solve_seconds)
        biased = [_arm("LP-biased hybrid GRASP", 42, 34.2, 0.04, 20,
                       {"insert": 0, "delete": 17, "swap": 3},
                       construct=9.3, search=164.0, lp_seconds=lp.solve_seconds)]
        alpha = [_arm("alpha-GRASP baseline (alpha=0.2)", 42, 18.1, 0.34, 36,
                      {"insert": 0, "delete": 9, "swap": 27},
                      construct=106.8, search=292.0)]
        from scripts.download_and_run_real_world import render_report

        md = render_report(
            n_fac=2000, n_cust=2000, lp=lp, control=control, rounding_ls=rounding_ls,
            lp_biased=biased, alpha=alpha, seeds=[42], env=ENV,
            generated_at="2026-09-11 00:00 UTC", commit="testcommit", wall_seconds=1.0,
        )
        assert "highly fractional" in md
        assert "essentially integral" not in md


class TestNoBackComputedFigures:
    def test_reported_costs_are_the_measured_costs(self):
        """
        The previous artifact derived its two initial-cost cells as
        `bound * (1 + rounded_gap)`, losing precision in the second decimal.
        The renderer must print the cost it was handed.
        """
        lp = _lp()
        measured_cost = 533_580.06
        control = _arm("LP rounding (control)", None, 1.0, 1.0, 0,
                       {"insert": 0, "delete": 0, "swap": 0},
                       construct=0.1, search=0.0, lp_seconds=lp.solve_seconds)
        rounding_ls = _arm("LP rounding + local search (deterministic ablation)", None, 1.0, 0.55, 5,
                           {"insert": 0, "delete": 3, "swap": 2},
                           construct=0.1, search=40.0, lp_seconds=lp.solve_seconds)
        from scripts.download_and_run_real_world import ArmResult, render_report

        biased = [ArmResult(
            method="LP-biased hybrid GRASP", seed=42,
            initial_cost=measured_cost, initial_gap=34.2001, initial_open=75,
            final_cost=397_767.97, final_gap=0.0421, final_open=58,
            iterations=20, moves={"insert": 0, "delete": 17, "swap": 3},
            construct_seconds=9.3, search_seconds=164.0, lp_seconds=lp.solve_seconds,
        )]
        alpha = [_arm("alpha-GRASP baseline (alpha=0.2)", 42, 18.0699, 0.3444, 36,
                      {"insert": 0, "delete": 9, "swap": 27},
                      construct=106.8, search=292.0)]
        md = render_report(
            n_fac=2000, n_cust=2000, lp=lp, control=control, rounding_ls=rounding_ls,
            lp_biased=biased, alpha=alpha, seeds=[42], env=ENV,
            generated_at="2026-09-11 00:00 UTC", commit="testcommit", wall_seconds=1.0,
        )
        assert "533,580.06" in md
        # The back-computed value that the superseded artifact printed.
        assert "533,580.24" not in md

    def test_report_records_its_own_provenance(self):
        """
        COUNTERWEIGHT: honest wording is not enough on its own -- a reader must
        be able to tell what produced the file and re-run it.
        """
        md = _render(34.2, 18.1, 0.04, 0.34, 164.0, 292.0, 9.3, 106.8)
        assert "render_report" in md
        assert "testcommit" in md
        assert "2026-09-11 00:00 UTC" in md
        assert "python download_and_run_real_world.py" in md
        assert "california_4M_results.json" in md


def _generator_source_without_module_docstring() -> str:
    """
    The generator's own docstring quotes the defective phrases to explain the
    defect. Strip it so the scan sees only executable code and templates.
    """
    import ast

    text = GENERATOR.read_text()
    tree = ast.parse(text)
    docstring_node = (
        tree.body[0]
        if tree.body
        and isinstance(tree.body[0], ast.Expr)
        and isinstance(tree.body[0].value, ast.Constant)
        and isinstance(tree.body[0].value.value, str)
        else None
    )
    if docstring_node is None:
        return text
    lines = text.splitlines(keepends=True)
    return "".join(lines[docstring_node.end_lineno:])


class TestTemplateCarriesNoHardcodedVerdict:
    def test_generator_source_contains_no_hardcoded_comparative_verdicts(self):
        """
        The defect in its rawest form. The pre-fix template embedded these exact
        strings in the markdown it emitted, so the report asserted them no
        matter what the run measured. RED against the pre-fix generator.
        """
        src = _generator_source_without_module_docstring()
        forbidden = [
            "LP-biased starts lower",
            "LP-biased gap is {",
            "LP-biased is faster by",
            "LP-biased finishes closer",
            "LP-biased reduces workload",
        ]
        found = [phrase for phrase in forbidden if phrase in src]
        assert not found, (
            f"generator hardcodes comparative verdicts into its report template: {found}; "
            "the report can then contradict its own table"
        )

    def test_generator_exposes_a_pure_renderer(self):
        """
        COUNTERWEIGHT: removing the hardcoded phrases is not enough -- the
        report must be produced by a function that takes measured records, so
        it can be tested without a 30-minute benchmark run.
        """
        import inspect

        from scripts.download_and_run_real_world import render_report

        params = inspect.signature(render_report).parameters
        for required in ("lp", "control", "lp_biased", "alpha", "seeds"):
            assert required in params, f"render_report is missing the {required!r} input"

    def test_generator_still_documents_the_defect_it_fixed(self):
        """
        COUNTERWEIGHT to the check above: the scan must skip the module
        docstring, so explaining the historical defect in prose stays allowed.
        Documentation is not a regression.
        """
        import ast

        tree = ast.parse(GENERATOR.read_text())
        docstring = ast.get_docstring(tree) or ""
        assert "PROVENANCE CONTRACT" in docstring

    def test_every_writer_of_the_results_file_goes_through_the_renderer(self):
        """
        COUNTERWEIGHT: the results file may have more than one write site (the
        benchmark run and the `--from-json` re-render), but every one of them
        must obtain its text from `render_report`. A post-processing step that
        writes the file from anything else -- the `fix_markdown.py` pattern --
        is exactly what this forbids.
        """
        import ast

        src = GENERATOR.read_text()
        assert "fix_markdown" not in src, "a post-processing step was re-introduced"

        tree = ast.parse(src)
        writers = []
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                body = ast.get_source_segment(src, node) or ""
                if "california_4M_results.md" in body:
                    writers.append((node.name, body))

        assert writers, "nothing writes the results file any more"
        for name, body in writers:
            assert "render_report" in body, (
                f"{name}() writes california_4M_results.md without going through "
                "render_report; its numbers would not be traceable to a measurement"
            )


class TestAblationVerdictFollowsTheData:
    """
    Section 6 is the experiment that decides what the method contributes, so
    its conclusion must be derived from the measured arms in every direction --
    including the direction that favours the method being audited. A verdict
    that can only come out one way is not a verdict.
    """

    @staticmethod
    def _render_ablation(ap_final_gap, ap_iters, b_final_gaps, ap_final_cost=None):
        from scripts.download_and_run_real_world import ArmResult, LPProfile, render_report

        lp = LPProfile(
            bound=100_000.0, solve_seconds=67.0, n_facilities=2000,
            n_integral_open=54, n_fractional=7, n_near_zero=1939,
            sum_y=57.5, rounded_open_count=61,
        )
        control = ArmResult(
            method="LP rounding (control)", seed=None,
            initial_cost=101_033.0, initial_gap=1.033, initial_open=61,
            final_cost=101_033.0, final_gap=1.033, final_open=61,
            iterations=0, moves={"insert": 0, "delete": 0, "swap": 0},
            construct_seconds=9.0, search_seconds=0.0, lp_seconds=67.0,
        )
        rounding_ls = ArmResult(
            method="LP rounding + local search (deterministic ablation)", seed=None,
            initial_cost=101_033.0, initial_gap=1.033, initial_open=61,
            final_cost=(ap_final_cost if ap_final_cost is not None
                        else 100_000.0 * (1 + ap_final_gap / 100)),
            final_gap=ap_final_gap, final_open=58,
            iterations=ap_iters, moves={"insert": 0, "delete": 3, "swap": 2},
            construct_seconds=9.0, search_seconds=40.0, lp_seconds=67.0,
        )
        biased = [
            ArmResult(
                method="LP-biased hybrid GRASP", seed=seed,
                initial_cost=134_200.0, initial_gap=34.2, initial_open=75,
                final_cost=100_000.0 * (1 + g / 100), final_gap=g, final_open=58,
                iterations=20, moves={"insert": 0, "delete": 17, "swap": 3},
                construct_seconds=9.3, search_seconds=164.0, lp_seconds=67.0,
            )
            for seed, g in zip([42, 7, 2024], b_final_gaps)
        ]
        alpha = [
            ArmResult(
                method="alpha-GRASP baseline (alpha=0.2)", seed=seed,
                initial_cost=118_070.0, initial_gap=18.07, initial_open=66,
                final_cost=100_344.0, final_gap=0.344, final_open=57,
                iterations=36, moves={"insert": 0, "delete": 9, "swap": 27},
                construct_seconds=106.8, search_seconds=292.0, lp_seconds=0.0,
            )
            for seed in [42, 7, 2024]
        ]
        return render_report(
            n_fac=2000, n_cust=2000, lp=lp, control=control, rounding_ls=rounding_ls,
            lp_biased=biased, alpha=alpha, seeds=[42, 7, 2024], env=ENV,
            generated_at="2026-09-12 00:00 UTC", commit="testcommit", wall_seconds=2272.0,
        )

    def test_identical_final_solution_is_reported_as_no_contribution(self):
        """The measured case: both arms land on the same solution."""
        md = self._render_ablation(
            ap_final_gap=0.0421, ap_iters=5,
            b_final_gaps=[0.0421, 0.0421, 0.0421],
            ap_final_cost=100_000.0 * 1.000421,
        )
        assert "Both arms reach the identical solution" in md
        assert "randomized construction contributes no measurable quality" in md
        assert "Randomization does buy quality here" not in md

    def test_randomization_winning_is_reported_as_a_contribution(self):
        """
        COUNTERWEIGHT: if the randomized arm is genuinely better, the report
        must say so plainly. This is the branch that keeps the audit honest.
        """
        md = self._render_ablation(
            ap_final_gap=0.9000, ap_iters=5,
            b_final_gaps=[0.0421, 0.0500, 0.0600],
        )
        assert "Randomization does buy quality here" in md
        assert "The randomized construction does buy quality." in md
        assert "Both arms reach the identical solution" not in md
        assert "contributes no measurable quality" not in md

    def test_deterministic_beating_only_the_mean_is_distinguished(self):
        """
        A middle case: rounding beats the randomized mean but not its best run.
        The report must not overstate that as beating every run.
        """
        md = self._render_ablation(
            ap_final_gap=0.0500, ap_iters=5,
            b_final_gaps=[0.0421, 0.0700, 0.0700],
        )
        assert "Deterministic rounding beats the randomized mean" in md
        assert "matches or beats every randomized run" not in md

    def test_deterministic_beating_every_run_is_distinguished(self):
        md = self._render_ablation(
            ap_final_gap=0.0100, ap_iters=5,
            b_final_gaps=[0.0421, 0.0700, 0.0700],
        )
        assert "matches or beats every randomized run" in md
        assert "beats the randomized mean" not in md

    def test_iteration_cost_is_only_claimed_when_it_is_real(self):
        md_costly = self._render_ablation(
            ap_final_gap=0.0421, ap_iters=5,
            b_final_gaps=[0.0421, 0.0421, 0.0421],
            ap_final_cost=100_000.0 * 1.000421,
        )
        assert "It is not free." in md_costly
        assert "4.0× as many" in md_costly     # 20 iterations against 5

        md_cheap = self._render_ablation(
            ap_final_gap=0.0421, ap_iters=40,
            b_final_gaps=[0.0421, 0.0421, 0.0421],
            ap_final_cost=100_000.0 * 1.000421,
        )
        assert "It is not free." not in md_cheap
        assert "needs 20.0 local-search iterations against" in md_cheap

    def test_the_multistart_caveat_is_stated(self):
        md = self._render_ablation(
            ap_final_gap=0.0421, ap_iters=5,
            b_final_gaps=[0.0421, 0.0421, 0.0421],
            ap_final_cost=100_000.0 * 1.000421,
        )
        assert "GRASP is a *multistart* procedure" in md
        assert "there is no restart loop anywhere" in md
