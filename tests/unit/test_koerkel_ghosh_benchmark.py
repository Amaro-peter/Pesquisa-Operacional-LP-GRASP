"""
Unit tests for the Körkel-Ghosh multistart benchmark harness.

The harness answers two questions, and these tests concentrate on the sentences
that answer them:

  1. With a restart budget, does the LP-biased construction beat the classical
     α-GRASP baseline?
  2. Does that restart budget pay for itself against a deterministic single run?

Both verdicts must follow the measurements in every direction — including the
directions that would overturn this project's earlier single-start conclusion.
A verdict that can only come out one way is not a verdict.

Reference handling is tested here too: a percentage may only be called an
"optimality gap" when the integer optimum was actually proven.
"""

import json
import math
import sys

import pytest

from scripts.run_koerkel_ghosh import _excess, main, render_report, run_one_instance

ENV = {"python": "3.14.6", "numpy": "2.5.3", "scipy": "1.18.1",
       "scikit_learn": "1.9.1", "platform": "test", "cpu_count": "12"}

REF = 100_000.0


def _det_arm(gap_pct, iterations=5):
    """A deterministic arm landing `gap_pct` above the reference."""
    return {
        "method": "det", "seed": None,
        "initial_cost": REF * 1.3, "initial_gap": 30.0, "initial_open": 20,
        "final_cost": REF * (1 + gap_pct / 100), "final_gap": gap_pct, "final_open": 10,
        "iterations": iterations, "moves": {"insert": 0, "delete": 3, "swap": 2},
        "construct_seconds": 1.0, "search_seconds": 2.0, "lp_seconds": 3.0,
    }


def _ms_arm(trajectory_gaps, best_found_at=None):
    """A multistart arm whose best-so-far trajectory is `trajectory_gaps`."""
    costs = [REF * (1 + g / 100) for g in trajectory_gaps]
    best = min(trajectory_gaps)
    if best_found_at is None:
        best_found_at = trajectory_gaps.index(best) + 1
    return {
        "method": "ms", "seeds": list(range(1, len(trajectory_gaps) + 1)),
        "best_cost": REF * (1 + best / 100), "best_gap": best, "best_open": 10,
        "best_found_at": best_found_at,
        "trajectory": list(trajectory_gaps), "trajectory_costs": costs,
        "per_restart_gaps": list(trajectory_gaps),
        "total_iterations": 40, "moves": {"insert": 0, "delete": 3, "swap": 2},
        "construct_seconds": 1.0, "search_seconds": 4.0, "lp_seconds": 3.0,
    }


def _record(name, a=5.0, ap=0.10, d=0.20, b=None, c=None, proven=True, lp_bound=99_000.0):
    b = b if b is not None else [1.0, 0.5, 0.20]
    c = c if c is not None else [2.0, 1.0, 0.30]
    return {
        "name": name, "size": 100, "klass": "a", "symmetric": True, "index": 1,
        "lp": {
            "bound": lp_bound, "solve_seconds": 1.0, "n_facilities": 100,
            "n_integral_open": 5, "n_fractional": 30, "n_near_zero": 65,
            "sum_y": 9.0, "rounded_open_count": 3,
        },
        "reference": {
            "value": REF, "lp_bound": lp_bound, "proven": proven,
            "ip_status": "Optimal" if proven else "Not Solved",
            "ip_seconds": 12.0, "incumbent": REF if proven else None,
        },
        "best_cost_found": REF,
        "arms": {
            "lp_rounding_control": _det_arm(a, iterations=0),
            "lp_rounding_plus_search": _det_arm(ap),
            "local_search_only": _det_arm(d),
            "lp_biased_multistart": _ms_arm(b),
            "alpha_grasp_multistart": _ms_arm(c),
        },
    }


def _render(records, restarts=3, size=100, alpha=0.2):
    return render_report(records, list(range(1, restarts + 1)), size, ENV,
                         "2026-09-12 00:00 UTC", "testcommit", 120.0, alpha_value=alpha)


def _prose(md):
    """Flatten wrapping and blockquote markers; assertions are about claims."""
    stripped = [ln.lstrip().removeprefix("> ").removeprefix(">") for ln in md.splitlines()]
    return " ".join(" ".join(stripped).split())


# --------------------------------------------------------------------------
# Reference values: what a percentage is allowed to be called
# --------------------------------------------------------------------------

class TestReferenceHonesty:
    def test_all_proven_is_reported_as_true_optimality_gaps(self):
        md = _prose(_render([_record("gs100a-1", proven=True),
                             _record("gs100a-2", proven=True)]))
        assert "Gaps here are true optimality gaps" in md
        assert "CBC proved the integer optimum on all" in md
        assert "OVERSTATE" not in md

    def test_unproven_instances_force_the_weaker_label(self):
        """COUNTERWEIGHT: an unproven optimum must not be called an optimality gap."""
        md = _prose(_render([_record("gs100a-1", proven=True),
                             _record("gs100a-2", proven=False)]))
        assert "Mixed references" in md
        assert "CBC proved the integer optimum on 1 of 2" in md
        assert "OVERSTATE the true optimality gap" in md
        assert "true optimality gaps" not in md

    def test_unproven_rows_are_flagged_in_the_per_instance_table(self):
        md = _render([_record("gs100a-1", proven=True), _record("gs100b-1", proven=False)])
        assert "`gs100b-1` ⚠" in md
        assert "integer optimum not proven" in md

    def test_proven_count_is_counted_not_asserted(self):
        md = _prose(_render([_record(f"i{i}", proven=(i < 3)) for i in range(5)]))
        assert "Integer optima proven | 3 / 5" in md


# --------------------------------------------------------------------------
# Question 1: LP-biased GRASP vs classical GRASP, at equal restarts
# --------------------------------------------------------------------------

class TestHeadToHead:
    def test_lp_biased_winning_is_reported_plainly(self):
        md = _prose(_render([
            _record("i1", b=[1.0, 0.10], c=[2.0, 0.50]),
            _record("i2", b=[1.0, 0.05], c=[2.0, 0.40]),
        ], restarts=2))
        assert "the LP-biased construction beats the classical baseline" in md
        assert "B beat C on 2" in md

    def test_baseline_winning_is_reported_plainly(self):
        """COUNTERWEIGHT: the baseline must be able to win, and be said to."""
        md = _prose(_render([
            _record("i1", b=[1.0, 0.50], c=[2.0, 0.10]),
            _record("i2", b=[1.0, 0.40], c=[2.0, 0.05]),
        ], restarts=2))
        assert "the classical baseline beats the LP-biased construction" in md
        assert "B beat C on 0" in md and "lost on 2" in md

    def test_a_tie_is_reported_as_indistinguishable(self):
        md = _prose(_render([_record("i1", b=[1.0, 0.2], c=[1.0, 0.2])], restarts=2))
        assert "indistinguishable on this family" in md

    def test_median_winning_restart_is_reported(self):
        """
        How much of the budget each arm actually needed. Without it a
        best-of-N number cannot be read as anything but best-of-N.
        """
        md = _prose(_render([_record("i1", b=[2.0, 1.0, 0.2], c=[2.0, 0.3, 0.3])], restarts=3))
        assert "Median restart that produced the winner" in md


# --------------------------------------------------------------------------
# Question 2: does the restart budget pay for itself?
# --------------------------------------------------------------------------

class TestMultistartValue:
    def test_deterministic_arm_holding_up_is_reported_as_such(self):
        md = _prose(_render([
            _record("i1", ap=0.05, b=[1.0, 0.30]),
            _record("i2", ap=0.05, b=[1.0, 0.30]),
        ], restarts=2))
        assert "The deterministic arm still holds up" in md
        assert "does not overturn it at this budget" in md

    def test_multistart_overturning_the_earlier_result_is_reported_as_such(self):
        """
        COUNTERWEIGHT, and the most important branch in this file: if the
        restart loop makes the randomized arm win, the report must say the
        earlier single-start finding was an artifact of running it once.
        """
        md = _prose(_render([
            _record("i1", ap=0.50, b=[1.0, 0.05]),
            _record("i2", ap=0.60, b=[1.0, 0.05]),
        ], restarts=2))
        assert "Multistart overturns the single-start result" in md
        assert "an artifact of running it exactly once" in md
        assert "still holds up" not in md

    def test_restarts_needed_to_match_the_deterministic_arm_are_measured(self):
        """The compute-matched metric: how many restarts to reach A+'s quality."""
        md = _prose(_render([
            _record("i1", ap=0.50, b=[2.0, 1.0, 0.40]),
            _record("i2", ap=0.50, b=[2.0, 0.45, 0.45]),
        ], restarts=3))
        assert "median of" in md and "restart" in md

    def test_instances_never_matched_within_budget_are_called_out(self):
        md = _prose(_render([
            _record("i1", ap=0.01, b=[2.0, 1.0, 0.40]),
            _record("i2", ap=0.01, b=[2.0, 1.0, 0.40]),
        ], restarts=3))
        assert "never matched A+ within its 3-restart budget" in md

    def test_the_budget_is_stated_so_best_of_n_cannot_be_read_without_it(self):
        md = _prose(_render([_record("i1")], restarts=32))
        assert "Restarts per randomized arm | **32**" in md
        assert "best-of-32" in md
        assert "costs N times the work of a single run" in md


# --------------------------------------------------------------------------
# Record shape and end-to-end
# --------------------------------------------------------------------------

class TestExcess:
    def test_excess_is_percent_above_the_reference(self):
        assert math.isclose(_excess(110.0, 100.0), 10.0)
        assert _excess(100.0, 100.0) == 0.0


class TestEndToEnd:
    def test_run_one_instance_produces_a_complete_record(self):
        rec = run_one_instance(size=30, klass="a", symmetric=True, index=1,
                               seeds=[1, 2, 3], alpha_value=0.2, ip_time_limit=60)

        assert rec["name"] == "gs30a-1"
        assert set(rec["arms"]) == {
            "lp_rounding_control", "lp_rounding_plus_search", "local_search_only",
            "lp_biased_multistart", "alpha_grasp_multistart",
        }
        assert set(rec["reference"]) >= {"value", "lp_bound", "proven", "ip_status"}

        ms = rec["arms"]["lp_biased_multistart"]
        assert len(ms["trajectory"]) == 3
        assert ms["best_cost"] == pytest.approx(min(ms["trajectory_costs"]))
        assert 1 <= ms["best_found_at"] <= 3

        costs = [
            rec["arms"]["lp_rounding_control"]["final_cost"],
            rec["arms"]["lp_rounding_plus_search"]["final_cost"],
            rec["arms"]["local_search_only"]["final_cost"],
            rec["arms"]["lp_biased_multistart"]["best_cost"],
            rec["arms"]["alpha_grasp_multistart"]["best_cost"],
        ]
        assert math.isclose(rec["best_cost_found"], min(costs))
        # No arm may beat a valid lower bound...
        assert rec["best_cost_found"] >= rec["lp"]["bound"] - 1e-6
        # ...nor a proven optimum.
        if rec["reference"]["proven"]:
            assert rec["best_cost_found"] >= rec["reference"]["value"] - 1e-6

    def test_more_restarts_never_worsen_the_best(self):
        few = run_one_instance(30, "a", True, 1, seeds=[1, 2], ip_time_limit=60)
        many = run_one_instance(30, "a", True, 1, seeds=[1, 2, 3, 4, 5, 6], ip_time_limit=60)
        assert (many["arms"]["lp_biased_multistart"]["best_cost"]
                <= few["arms"]["lp_biased_multistart"]["best_cost"] + 1e-9)

    def test_main_writes_agreeing_markdown_and_json(self, tmp_path, monkeypatch):
        out = tmp_path / "kg"
        monkeypatch.setattr(sys, "argv", [
            "run_koerkel_ghosh.py", "--size", "25", "--classes", "a",
            "--instances", "1", "--restarts", "3", "--ip-time-limit", "60",
            "--out-dir", str(out),
        ])
        main()

        md = (out / "koerkel_ghosh_results.md").read_text()
        payload = json.loads((out / "koerkel_ghosh_results.json").read_text())

        assert payload["size"] == 25
        assert payload["restarts"] == 3
        assert payload["seeds"] == [1, 2, 3]
        assert "NOT the official UflLib files" in payload["provenance"]
        assert len(payload["records"]) == 2
        for rec in payload["records"]:
            assert rec["name"] in md

        for heading in ("## 1. Setup", "## 2. Results",
                        "## 3. LP-biased GRASP vs. classical GRASP",
                        "## 4. Does multistart pay for itself?",
                        "## 5. Per-instance detail", "## 6. Reproduction"):
            assert heading in md, f"missing {heading}"


class TestProvenanceIsStated:
    def test_report_declares_the_instances_are_generated_to_spec(self):
        md = _prose(_render([_record("gs100a-1")]))
        assert "not the official UflLib files" in md
        assert "not comparable with published KG results" in md
        assert "render_report" in md
        assert "testcommit" in md
