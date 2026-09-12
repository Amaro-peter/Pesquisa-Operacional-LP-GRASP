"""
Unit tests for the Körkel-Ghosh benchmark harness.

The harness exists to answer one question — does the randomized construction
help where the LP is weak? — so the tests concentrate on the part that states
the answer. The verdict must follow the measured excesses in every direction,
including the direction that would overturn the finding from the California
instance. A verdict that can only come out one way is not a verdict.
"""

import json
import math
import sys

import pytest

from run_koerkel_ghosh import _excess, main, render_report, run_one_instance

ENV = {"python": "3.14.6", "numpy": "2.5.3", "scipy": "1.18.1",
       "scikit_learn": "1.9.1", "platform": "test", "cpu_count": "12"}


def _record(name, ap_excess, b_excesses, c_excesses, a_excess=5.0,
            n_fractional=40, rounded_open=3, sum_y=9.0, bound=100_000.0,
            d_excess=None):
    """Build a record with prescribed excesses over a best-found cost."""
    best = bound * 1.02

    def arm(excess, iterations=5):
        return {
            "method": "m", "seed": None,
            "initial_cost": best * 1.3, "initial_gap": 30.0, "initial_open": 20,
            "final_cost": best * (1 + excess / 100), "final_gap": 1.0, "final_open": 10,
            "iterations": iterations, "moves": {"insert": 0, "delete": 3, "swap": 2},
            "construct_seconds": 1.0, "search_seconds": 2.0, "lp_seconds": 3.0,
        }

    return {
        "name": name, "size": 250, "klass": "a", "symmetric": True, "index": 1,
        "lp": {
            "bound": bound, "solve_seconds": 3.0, "n_facilities": 250,
            "n_integral_open": 5, "n_fractional": n_fractional,
            "n_near_zero": 250 - n_fractional - 5, "sum_y": sum_y,
            "rounded_open_count": rounded_open,
        },
        "best_cost_found": best,
        "arms": {
            "lp_rounding_control": arm(a_excess, iterations=0),
            "lp_rounding_plus_search": arm(ap_excess),
            "local_search_only": arm(ap_excess if d_excess is None else d_excess),
            "lp_biased": [arm(e) for e in b_excesses],
            "alpha_grasp": [arm(e) for e in c_excesses],
        },
    }


def _render(records, seeds=(42, 7)):
    return render_report(records, list(seeds), 250, ENV,
                         "2026-09-12 00:00 UTC", "testcommit", 120.0)


def _prose(md: str) -> str:
    """
    Flatten the markdown to a single line of prose.

    Sentences in the report wrap across lines and, inside block quotes, are
    additionally broken by "> " markers. Assertions here are about the claims
    the report makes, not about where the lines happen to break.
    """
    stripped = [line.lstrip().removeprefix("> ").removeprefix(">") for line in md.splitlines()]
    return " ".join(" ".join(stripped).split())


class TestExcess:
    def test_excess_is_percent_above_the_best_found(self):
        assert math.isclose(_excess(110.0, 100.0), 10.0)
        assert math.isclose(_excess(100.0, 100.0), 0.0)

    def test_the_best_arm_has_zero_excess(self):
        assert _excess(100.0, 100.0) == 0.0


class TestVerdictFollowsTheData:
    def test_randomization_winning_is_reported_as_vindication(self):
        """
        The branch that would overturn the California finding. It must be
        reachable and must say so plainly.
        """
        records = [
            _record("gs250a-1", ap_excess=1.00, b_excesses=[0.20, 0.30], c_excesses=[0.9, 1.1]),
            _record("gs250b-1", ap_excess=0.80, b_excesses=[0.10, 0.20], c_excesses=[0.7, 0.8]),
        ]
        md = _render(records)
        assert "Randomization earns its place on this family" in md
        assert "The LP bias is vindicated on this family" in md
        assert "does not earn its place" not in md

    def test_randomization_losing_is_reported_as_the_finding_surviving(self):
        records = [
            _record("gs250a-1", ap_excess=0.10, b_excesses=[0.50, 0.60], c_excesses=[0.9, 1.1]),
            _record("gs250b-1", ap_excess=0.05, b_excesses=[0.40, 0.50], c_excesses=[0.7, 0.8]),
        ]
        md = _render(records)
        assert "Randomization does not earn its place even here" in md
        assert "The LP bias is not vindicated even here" in md
        assert "earns its place on this family" not in md

    def test_an_exact_tie_is_reported_as_indistinguishable(self):
        records = [
            _record("gs250a-1", ap_excess=0.30, b_excesses=[0.30, 0.30], c_excesses=[0.9, 1.1]),
        ]
        md = _render(records)
        assert "indistinguishable on this family" in md
        assert "earns its place" not in md
        assert "does not earn its place" not in md

    def test_win_loss_tie_counts_are_computed_from_the_records(self):
        records = [
            _record("i1", ap_excess=1.0, b_excesses=[0.5], c_excesses=[2.0]),   # B wins
            _record("i2", ap_excess=0.2, b_excesses=[0.9], c_excesses=[2.0]),   # A+ wins
            _record("i3", ap_excess=0.4, b_excesses=[0.4], c_excesses=[2.0]),   # tie
        ]
        md = _render(records)
        assert "beat arm A+" in md
        assert "on **1**" in md and "lost on **1**" in md and "tied on **1**" in md

    def test_best_arm_is_selected_by_measured_mean_excess(self):
        records = [_record("i1", ap_excess=0.10, b_excesses=[0.90], c_excesses=[0.50], a_excess=5.0)]
        md = _render(records)
        assert "Best arm overall: A+ · rounding + local search" in md

        records = [_record("i1", ap_excess=0.90, b_excesses=[0.10], c_excesses=[0.50], a_excess=5.0)]
        md = _render(records)
        assert "Best arm overall: B · LP-biased GRASP" in md


class TestRoundingCollapseSection:
    def test_zero_threshold_instances_are_called_out(self):
        records = [
            _record("gs250c-1", 0.3, [0.3], [0.5], rounded_open=0, sum_y=3.3),
            _record("ga250c-1", 0.3, [0.3], [0.5], rounded_open=0, sum_y=3.1),
            _record("gs250a-1", 0.3, [0.3], [0.5], rounded_open=13, sum_y=29.0),
        ]
        md = _render(records)
        assert "On **2** of 3 instances the relaxation puts *no* facility" in md

    def test_no_callout_when_thresholding_always_selects_something(self):
        """COUNTERWEIGHT: the collapse claim must not appear when it is untrue."""
        records = [_record("gs250a-1", 0.3, [0.3], [0.5], rounded_open=13, sum_y=29.0)]
        md = _render(records)
        assert "puts *no* facility" not in md


class TestProvenanceIsStated:
    def test_report_declares_the_instances_are_generated_to_spec(self):
        """
        The one thing a reader must not miss: these are not the official files,
        so the numbers do not compare to published KG results.
        """
        md = _render([_record("gs250a-1", 0.3, [0.3], [0.5])])
        flat = _prose(md)
        assert "not the official UflLib files" in flat
        assert "not comparable with published KG results" in flat
        assert "generated to the published specification" in flat.lower() or \
               "follows the published specification" in flat
        assert "render_report" in flat
        assert "testcommit" in flat


class TestEndToEnd:
    def test_run_one_instance_produces_a_complete_record(self):
        rec = run_one_instance(size=30, klass="a", symmetric=True, index=1, seeds=[42, 7])

        assert rec["name"] == "gs30a-1"
        assert set(rec["arms"]) == {
            "lp_rounding_control", "lp_rounding_plus_search", "local_search_only",
            "lp_biased", "alpha_grasp",
        }
        assert len(rec["arms"]["lp_biased"]) == 2
        assert len(rec["arms"]["alpha_grasp"]) == 2

        # The recorded best must actually be the minimum over every arm.
        costs = [
            rec["arms"]["lp_rounding_control"]["final_cost"],
            rec["arms"]["lp_rounding_plus_search"]["final_cost"],
            rec["arms"]["local_search_only"]["final_cost"],
            *[r["final_cost"] for r in rec["arms"]["lp_biased"]],
            *[r["final_cost"] for r in rec["arms"]["alpha_grasp"]],
        ]
        assert math.isclose(rec["best_cost_found"], min(costs))
        # No arm may beat the LP bound -- it is a valid lower bound.
        assert rec["best_cost_found"] >= rec["lp"]["bound"] - 1e-6

    def test_main_writes_agreeing_markdown_and_json(self, tmp_path, monkeypatch):
        out = tmp_path / "kg"
        monkeypatch.setattr(sys, "argv", [
            "run_koerkel_ghosh.py", "--size", "25", "--classes", "a",
            "--instances", "1", "--seeds", "42", "--out-dir", str(out),
        ])
        main()

        md = (out / "koerkel_ghosh_results.md").read_text()
        payload = json.loads((out / "koerkel_ghosh_results.json").read_text())

        assert payload["size"] == 25
        assert payload["seeds"] == [42]
        assert "NOT the official UflLib files" in payload["provenance"]
        # One class, both symmetries, one instance each.
        assert len(payload["records"]) == 2
        for rec in payload["records"]:
            assert rec["name"] in md

        for heading in ("## 1. Why this family", "## 2. Results",
                        "## 3. The ablation: does randomization help here?",
                        "## 4. Why rounding alone collapses here",
                        "## 5. Findings", "## 6. Reproduction"):
            assert heading in md, f"missing {heading}"


class TestNoLpControlArm:
    """
    Arm D never reads the relaxation. It answers a different question from the
    A+/B ablation: not "does randomization help?" but "does the LP help at all?"
    Its verdict must follow the measurements in every direction.
    """

    def test_no_lp_arm_matching_is_reported_as_the_lp_contributing_nothing(self):
        records = [
            _record("i1", ap_excess=0.10, b_excesses=[0.3], c_excesses=[0.4], d_excess=0.10),
            _record("i2", ap_excess=0.20, b_excesses=[0.3], c_excesses=[0.4], d_excess=0.20),
        ]
        md = _render(records)
        assert "The LP contributes little or nothing on this family" in md
        assert "The LP does contribute here" not in md

    def test_lp_guided_arm_winning_clearly_is_reported_as_the_lp_contributing(self):
        records = [
            _record("i1", ap_excess=0.10, b_excesses=[0.3], c_excesses=[0.4], d_excess=2.00),
            _record("i2", ap_excess=0.10, b_excesses=[0.3], c_excesses=[0.4], d_excess=2.00),
        ]
        md = _render(records)
        assert "The LP does contribute here" in md
        assert "contributes little or nothing" not in md

    def test_no_lp_arm_winning_clearly_is_reported_as_the_lp_being_unhelpful(self):
        """COUNTERWEIGHT: the harshest verdict must also be reachable."""
        records = [
            _record("i1", ap_excess=2.00, b_excesses=[0.3], c_excesses=[0.4], d_excess=0.10),
            _record("i2", ap_excess=2.00, b_excesses=[0.3], c_excesses=[0.4], d_excess=0.10),
        ]
        md = _render(records)
        assert "The LP is actively unhelpful here" in md

    def test_no_lp_arm_appears_in_the_summary_table_and_can_win_overall(self):
        records = [_record("i1", ap_excess=0.90, b_excesses=[0.9], c_excesses=[0.9], d_excess=0.10)]
        md = _render(records)
        assert "D · Local search only (no LP)" in md
        assert "Best arm overall: D · local search only, no LP" in md
