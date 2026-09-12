"""
Unit tests for the California benchmark pipeline.

Covers the measurement records (`LPProfile`, `ArmResult`), the three method
runners, the small text helpers that decide the report's comparative wording,
and an end-to-end `main()` run on a tiny instance that asserts both output
files are written and agree with each other.

The wording helpers look trivial, but they are what stops the report from
asserting a conclusion its own table contradicts, so they are pinned at the
boundaries: equal values, values either side, and the tie tolerance.
"""

import json
import math
import os
import sys

import pytest

from scripts.download_and_run_real_world import (
    ArmResult,
    CONSTRUCTION_EPS,
    LPProfile,
    _cmp,
    _mean,
    _plural,
    _signed,
    _spread,
    create_california_uflp_instance,
    main,
    profile_lp,
    round_lp_solution,
    run_alpha_grasp,
    run_lp_biased,
    run_lp_rounding,
)
from scripts.run_experiments import get_lp_bound_and_probs
from scripts.uflp_solver import UFLPInstance


@pytest.fixture(scope="module")
def tiny():
    """A small California instance plus its solved LP, shared across tests."""
    instance = create_california_uflp_instance(n_fac=25, n_cust=20)
    bound, probs = get_lp_bound_and_probs(instance)
    lp = profile_lp(bound, probs, solve_seconds=0.25)
    return instance, probs, lp


# ---------------------------------------------------------------- records ---

class TestLPProfile:
    def test_counts_partition_the_facilities(self, tiny):
        _instance, probs, lp = tiny
        assert lp.n_facilities == len(probs)
        # The three bands are mutually exclusive and cover every facility.
        assert lp.n_integral_open + lp.n_fractional + lp.n_near_zero == lp.n_facilities

    def test_fractional_percentage_and_eps_expectation(self):
        lp = LPProfile(
            bound=1000.0, solve_seconds=1.0, n_facilities=2000,
            n_integral_open=50, n_fractional=10, n_near_zero=1940,
            sum_y=57.5, rounded_open_count=60,
        )
        assert math.isclose(lp.fractional_pct, 0.5)
        assert math.isclose(lp.expected_eps_openings, CONSTRUCTION_EPS * 1940)

    def test_profile_classifies_a_known_probability_vector(self):
        probs = {0: 1.0, 1: 0.995, 2: 0.5, 3: 0.4, 4: 0.005, 5: 0.0}
        lp = profile_lp(123.0, probs, solve_seconds=2.0)
        assert lp.bound == 123.0
        assert lp.solve_seconds == 2.0
        assert lp.n_facilities == 6
        assert lp.n_integral_open == 2          # 1.0, 0.995
        assert lp.n_fractional == 2             # 0.5, 0.4
        assert lp.n_near_zero == 2              # 0.005, 0.0
        assert lp.rounded_open_count == 3       # 1.0, 0.995, 0.5
        assert math.isclose(lp.sum_y, 2.9)


class TestArmResult:
    def test_total_seconds_sums_all_three_stages(self):
        r = ArmResult(
            method="m", seed=1, initial_cost=10.0, initial_gap=1.0, initial_open=2,
            final_cost=9.0, final_gap=0.5, final_open=1, iterations=3,
            construct_seconds=2.0, search_seconds=5.0, lp_seconds=11.0,
        )
        assert math.isclose(r.total_seconds, 18.0)

    def test_stage_times_default_to_zero(self):
        r = ArmResult(
            method="m", seed=None, initial_cost=1.0, initial_gap=0.0, initial_open=1,
            final_cost=1.0, final_gap=0.0, final_open=1, iterations=0,
        )
        assert r.total_seconds == 0.0


# ---------------------------------------------------------------- methods ---

class TestLPRounding:
    def test_rounds_at_the_threshold_inclusively(self):
        instance = UFLPInstance(
            facilities=[0, 1, 2],
            customers=[0],
            setup_costs={0: 1.0, 1: 1.0, 2: 1.0},
            service_costs={0: {0: 5.0, 1: 6.0, 2: 7.0}},
        )
        probs = {0: 0.9, 1: 0.5, 2: 0.49}
        state = round_lp_solution(instance, probs, threshold=0.5)
        assert state.open_facilities == {0, 1}

    def test_falls_back_to_the_highest_y_when_nothing_reaches_the_threshold(self):
        instance = UFLPInstance(
            facilities=[0, 1, 2],
            customers=[0],
            setup_costs={0: 1.0, 1: 1.0, 2: 1.0},
            service_costs={0: {0: 5.0, 1: 6.0, 2: 7.0}},
        )
        probs = {0: 0.1, 1: 0.3, 2: 0.2}
        state = round_lp_solution(instance, probs)
        assert state.open_facilities == {1}
        # A feasible solution must still be fully costed.
        assert math.isclose(state.total_cost, 1.0 + 6.0)
        assert state.closest_facility == {0: 1}

    def test_control_arm_reports_no_search_and_carries_the_lp_time(self, tiny):
        instance, probs, lp = tiny
        result = run_lp_rounding(instance, probs, lp)

        assert result.iterations == 0
        assert result.moves == {"insert": 0, "delete": 0, "swap": 0}
        assert result.search_seconds == 0.0
        assert result.lp_seconds == lp.solve_seconds
        # A control does no search, so initial and final must be identical.
        assert result.initial_cost == result.final_cost
        assert result.initial_gap == result.final_gap
        assert result.initial_open == result.final_open


class TestHeuristicArms:
    def test_lp_biased_arm_is_deterministic_for_a_fixed_seed(self, tiny):
        instance, probs, lp = tiny
        a = run_lp_biased(instance, probs, lp, seed=42)
        b = run_lp_biased(instance, probs, lp, seed=42)
        assert a.initial_cost == b.initial_cost
        assert a.final_cost == b.final_cost
        assert a.iterations == b.iterations
        assert a.moves == b.moves

    def test_lp_biased_arm_never_worsens_its_construction(self, tiny):
        instance, probs, lp = tiny
        r = run_lp_biased(instance, probs, lp, seed=7)
        assert r.final_cost <= r.initial_cost + 1e-9
        assert r.final_gap <= r.initial_gap + 1e-9
        assert sum(r.moves.values()) == r.iterations

    def test_lp_biased_arm_pays_for_the_lp_but_alpha_does_not(self, tiny):
        instance, probs, lp = tiny
        biased = run_lp_biased(instance, probs, lp, seed=42)
        alpha = run_alpha_grasp(instance, lp, seed=42)

        assert biased.lp_seconds == lp.solve_seconds
        assert alpha.lp_seconds == 0.0, "the baseline must not be charged for the LP solve"

    def test_alpha_arm_records_its_alpha_in_the_method_name(self, tiny):
        instance, _probs, lp = tiny
        r = run_alpha_grasp(instance, lp, seed=1, alpha=0.35)
        assert "0.35" in r.method

    def test_gaps_are_consistent_with_the_costs_and_the_bound(self, tiny):
        instance, probs, lp = tiny
        for r in (run_lp_biased(instance, probs, lp, seed=3),
                  run_alpha_grasp(instance, lp, seed=3),
                  run_lp_rounding(instance, probs, lp)):
            expected_init = (r.initial_cost - lp.bound) / lp.bound * 100
            expected_final = (r.final_cost - lp.bound) / lp.bound * 100
            assert math.isclose(r.initial_gap, expected_init, rel_tol=1e-12)
            assert math.isclose(r.final_gap, expected_final, rel_tol=1e-12)

    def test_no_arm_can_beat_the_lp_bound(self, tiny):
        """The LP is a relaxation; no feasible integer solution may be cheaper."""
        instance, probs, lp = tiny
        for r in (run_lp_biased(instance, probs, lp, seed=5),
                  run_alpha_grasp(instance, lp, seed=5),
                  run_lp_rounding(instance, probs, lp)):
            assert r.final_cost >= lp.bound - 1e-6, r.method
            assert r.final_gap >= -1e-9, r.method


# ---------------------------------------------------------------- helpers ---

class TestComparativeHelpers:
    def test_cmp_picks_the_word_matching_the_direction(self):
        assert _cmp(1.0, 2.0, "lower", "higher") == "lower"
        assert _cmp(2.0, 1.0, "lower", "higher") == "higher"

    def test_cmp_reports_a_tie_within_tolerance(self):
        assert _cmp(1.0, 1.0, "lower", "higher") == "the same as"
        assert _cmp(1.0, 1.0 + 1e-12, "lower", "higher") == "the same as"
        assert _cmp(1.0, 1.0 + 1e-6, "lower", "higher") == "lower"

    def test_cmp_accepts_a_custom_tie_word(self):
        assert _cmp(3.0, 3.0, "better", "worse", tie="level with") == "level with"

    def test_signed_always_shows_the_direction(self):
        assert _signed(1.5) == "+1.50"
        assert _signed(-1.5) == "-1.50"
        assert _signed(0.0) == "+0.00"
        assert _signed(-12.8702, " pp", 4) == "-12.8702 pp"
        assert _signed(1234.5, " s") == "+1,234.50 s"

    def test_plural_agrees_with_a_rounded_count(self):
        assert _plural(1, "facility", "facilities") == "facility"
        assert _plural(1.4, "facility", "facilities") == "facility"
        assert _plural(2, "facility", "facilities") == "facilities"
        assert _plural(0, "opening") == "openings"
        assert _plural(1, "opening") == "opening"

    def test_mean_and_spread(self):
        assert math.isclose(_mean([1.0, 2.0, 3.0]), 2.0)
        assert _spread([1.0]) == ""
        assert "1.0000" in _spread([1.0, 3.0]) and "3.0000" in _spread([1.0, 3.0])


# ------------------------------------------------------------ end-to-end ---

class TestMainWritesBothArtifacts:
    def test_main_writes_agreeing_markdown_and_json(self, tmp_path, monkeypatch):
        """
        Drive the whole pipeline on a tiny instance and check that the JSON
        sidecar and the markdown describe the same run. This is what makes the
        markdown auditable without re-running the benchmark.
        """
        out = tmp_path / "out"
        monkeypatch.setattr(
            sys, "argv",
            ["download_and_run_real_world.py", "--size", "20", "--seeds", "42",
             "--out-dir", str(out)],
        )
        main()

        md_path = out / "california_4M_results.md"
        json_path = out / "california_4M_results.json"
        assert md_path.exists() and json_path.exists()

        payload = json.loads(json_path.read_text())
        md = md_path.read_text()

        assert payload["seeds"] == [42]
        assert payload["instance"]["n_facilities"] == 20
        assert payload["construction_eps"] == CONSTRUCTION_EPS
        assert payload["generated_by"].endswith("render_report")

        # The LP bound printed in the markdown must be the measured one.
        bound = payload["lp_profile"]["bound"]
        assert f"{bound:,.2f}" in md

        # Every arm's final gap must appear in the report.
        for arm in payload["arms"]["lp_biased"] + payload["arms"]["alpha_grasp"]:
            assert f"{arm['final_gap']:.4f}%" in md
        assert f"{payload['arms']['lp_rounding_control']['final_gap']:.4f}%" in md

        # Section skeleton, so a silently truncated report fails loudly.
        for heading in ("## 1. Instance", "## 2. LP relaxation", "## 3. Methods compared",
                        "## 4. Results", "## 5. What the construction actually produces",
                        "## 6. What does the randomized construction contribute?",
                        "## 7. What the local search actually does",
                        "## 8. Where the time goes", "## 9. Findings",
                        "## 10. Limitations", "## 11. Reproduction"):
            assert heading in md, f"missing {heading}"

    def test_main_defaults_to_the_output_directory(self, monkeypatch, tmp_path):
        """`--out-dir` defaults to `output`; check the default is wired through."""
        monkeypatch.chdir(tmp_path)
        monkeypatch.setattr(
            sys, "argv",
            ["download_and_run_real_world.py", "--size", "15", "--seeds", "1"],
        )
        main()
        assert (tmp_path / "output" / "california_4M_results.md").exists()
        assert (tmp_path / "output" / "california_4M_results.json").exists()


class TestMultiSeedReporting:
    """
    The per-seed table and the multi-seed limitation wording only render when
    more than one seed is supplied. The published run uses three, so these
    paths ship to readers and need covering.
    """

    @staticmethod
    def _render_with_seeds(seeds):
        from scripts.download_and_run_real_world import render_report

        lp = LPProfile(
            bound=100_000.0, solve_seconds=57.0, n_facilities=2000,
            n_integral_open=50, n_fractional=7, n_near_zero=1943,
            sum_y=57.5, rounded_open_count=61,
        )
        control = ArmResult(
            method="LP rounding (control)", seed=None,
            initial_cost=101_033.0, initial_gap=1.033, initial_open=61,
            final_cost=101_033.0, final_gap=1.033, final_open=61,
            iterations=0, moves={"insert": 0, "delete": 0, "swap": 0},
            construct_seconds=9.0, search_seconds=0.0, lp_seconds=57.0,
        )
        rounding_ls = ArmResult(
            method="LP rounding + local search (deterministic ablation)", seed=None,
            initial_cost=101_033.0, initial_gap=1.033, initial_open=61,
            final_cost=100_055.0, final_gap=0.055, final_open=58,
            iterations=5, moves={"insert": 0, "delete": 3, "swap": 2},
            construct_seconds=9.0, search_seconds=40.0, lp_seconds=67.0,
        )
        biased, alpha = [], []
        for i, seed in enumerate(seeds):
            biased.append(ArmResult(
                method="LP-biased hybrid GRASP", seed=seed,
                initial_cost=134_200.0 + i, initial_gap=34.2 + i, initial_open=75,
                final_cost=100_042.0 + i, final_gap=0.042 + i * 0.01, final_open=58,
                iterations=20 + i, moves={"insert": 0, "delete": 17, "swap": 3},
                construct_seconds=9.3, search_seconds=164.0, lp_seconds=57.0,
            ))
            alpha.append(ArmResult(
                method="alpha-GRASP baseline (alpha=0.2)", seed=seed,
                initial_cost=118_070.0 + i, initial_gap=18.07 + i, initial_open=66,
                final_cost=100_344.0 + i, final_gap=0.344 + i * 0.01, final_open=57,
                iterations=36 + i, moves={"insert": 0, "delete": 9, "swap": 27},
                construct_seconds=106.8, search_seconds=292.0, lp_seconds=0.0,
            ))
        return render_report(
            n_fac=2000, n_cust=2000, lp=lp, control=control, rounding_ls=rounding_ls,
            lp_biased=biased, alpha=alpha, seeds=seeds,
            env={"python": "3.14.6", "numpy": "2.5.3", "scipy": "1.18.1",
                 "scikit_learn": "1.9.1", "platform": "test", "cpu_count": "12"},
            generated_at="2026-09-11 00:00 UTC", commit="abc1234", wall_seconds=1800.0,
        )

    def test_multiple_seeds_render_a_per_seed_table_and_a_spread(self):
        md = self._render_with_seeds([42, 7, 2024])

        assert "### Per-seed detail" in md
        for seed in (42, 7, 2024):
            assert f"| {seed} |" in md
        assert "Final-gap spread:" in md
        assert "3 seeds." in md
        assert "Single seed." not in md
        assert "values are means across seeds" in md

    def test_single_seed_omits_the_table_and_flags_the_limitation(self):
        md = self._render_with_seeds([42])

        assert "### Per-seed detail" not in md
        assert "Final-gap spread:" not in md
        assert "Single seed." in md
        assert "single-seed values" in md


class TestRobustnessNarrative:
    """
    The multi-seed run's strongest result is that one arm converges to the same
    solution from every start while the other does not. That claim must follow
    the measured spread, not be asserted.
    """

    @staticmethod
    def _render(biased_finals, alpha_finals):
        from scripts.download_and_run_real_world import render_report

        seeds = [42, 7, 2024][: len(biased_finals)]
        lp = LPProfile(
            bound=100_000.0, solve_seconds=67.0, n_facilities=2000,
            n_integral_open=50, n_fractional=7, n_near_zero=1939,
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
            final_cost=100_055.0, final_gap=0.055, final_open=58,
            iterations=5, moves={"insert": 0, "delete": 3, "swap": 2},
            construct_seconds=9.0, search_seconds=40.0, lp_seconds=67.0,
        )

        def arm(method, seed, final_gap, construct, search, lp_seconds):
            return ArmResult(
                method=method, seed=seed,
                initial_cost=134_200.0, initial_gap=34.2, initial_open=75,
                final_cost=100_000.0 * (1 + final_gap / 100), final_gap=final_gap,
                final_open=58, iterations=20,
                moves={"insert": 0, "delete": 17, "swap": 3},
                construct_seconds=construct, search_seconds=search, lp_seconds=lp_seconds,
            )

        biased = [arm("LP-biased hybrid GRASP", s, g, 9.3, 164.0, 67.0)
                  for s, g in zip(seeds, biased_finals)]
        alpha = [arm("alpha-GRASP baseline (alpha=0.2)", s, g, 106.8, 292.0, 0.0)
                 for s, g in zip(seeds, alpha_finals)]
        return render_report(
            n_fac=2000, n_cust=2000, lp=lp, control=control, rounding_ls=rounding_ls,
            lp_biased=biased, alpha=alpha, seeds=seeds,
            env={"python": "3.14.6", "numpy": "2.5.3", "scipy": "1.18.1",
                 "scikit_learn": "1.9.1", "platform": "test", "cpu_count": "12"},
            generated_at="2026-09-12 00:00 UTC", commit="abc1234", wall_seconds=2272.0,
        )

    def test_identical_solutions_across_seeds_are_reported(self):
        md = self._render([0.0421, 0.0421, 0.0421], [0.3444, 0.1834, 0.0421])
        assert "reached the identical solution on all 3 seeds" in md
        assert "**B** is the more consistent arm" in md       # section 4
        assert "Arm B is the more reproducible." in md         # section 8
        assert "1 distinct solution" in md
        assert "3 distinct solutions" in md

    def test_a_varying_arm_is_not_credited_with_identical_solutions(self):
        """COUNTERWEIGHT: the claim must disappear when the data stops supporting it."""
        md = self._render([0.04, 0.09, 0.21], [0.3444, 0.1834, 0.0421])
        assert "reached the identical solution" not in md

    def test_reproducibility_verdict_follows_the_measured_spread(self):
        """If the baseline is the steadier arm, the report must say so."""
        md = self._render([0.10, 0.50, 0.90], [0.20, 0.20, 0.20])
        assert "Arm C is the more reproducible." in md
        assert "Arm B is the more reproducible." not in md
        assert "**C** is the more consistent arm" in md

    def test_single_seed_makes_no_reproducibility_claim(self):
        md = self._render([0.0421], [0.3444])
        assert "more reproducible" not in md
        assert "distinct solution" not in md
        # Finding numbering must close up rather than skip.
        assert "**6. The iteration counts measure different work.**" in md
        assert "**7. The iteration counts" not in md

    def test_timing_caveat_makes_no_unfounded_precision_claim(self):
        md = self._render([0.0421, 0.0421, 0.0421], [0.3444, 0.1834, 0.0421])
        assert "Only compare timings within this run." in md
        assert "±15%" not in md


class TestRerenderFromJson:
    def test_rerender_reproduces_the_report_without_rerunning(self, tmp_path, monkeypatch):
        """
        The sidecar exists so prose can be corrected without a 38-minute
        re-run. Render once, re-render from the JSON, and require byte
        equality: the measurements must survive the round trip exactly.
        """
        from scripts.download_and_run_real_world import rerender_from_json

        out = tmp_path / "out"
        monkeypatch.setattr(
            sys, "argv",
            ["download_and_run_real_world.py", "--size", "18", "--seeds", "42", "7",
             "--out-dir", str(out)],
        )
        main()
        original = (out / "california_4M_results.md").read_text()

        (out / "california_4M_results.md").unlink()
        rerender_from_json(str(out / "california_4M_results.json"), str(out))

        assert (out / "california_4M_results.md").read_text() == original

    def test_rerender_via_cli_flag(self, tmp_path, monkeypatch, capsys):
        out = tmp_path / "out"
        monkeypatch.setattr(
            sys, "argv",
            ["download_and_run_real_world.py", "--size", "15", "--seeds", "1",
             "--out-dir", str(out)],
        )
        main()

        target = tmp_path / "redo"
        monkeypatch.setattr(
            sys, "argv",
            ["download_and_run_real_world.py",
             "--from-json", str(out / "california_4M_results.json"),
             "--out-dir", str(target)],
        )
        main()

        assert (target / "california_4M_results.md").exists()
        assert "no benchmark re-run" in capsys.readouterr().out
        # Re-rendering must not fabricate a sidecar it did not measure.
        assert not (target / "california_4M_results.json").exists()
