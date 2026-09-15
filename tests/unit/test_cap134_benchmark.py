"""
Unit tests for the cap134 benchmark driver.

cap134 is the control instance: its LP relaxation is integral, so the bound
equals the optimum and every method finds it. The report must say what that
does and does not demonstrate rather than dress a tie up as a win, so these
tests concentrate on the wording that makes that distinction.
"""

import json
import sys

import pytest

from scripts.run_cap134 import main, render_report, resolve_instance_path, run_benchmark

ENV = {"python": "3.14.6", "numpy": "2.5.3", "scipy": "1.18.1",
       "scikit_learn": "1.9.1", "platform": "test", "cpu_count": "12"}


def _arm(cost, lp_seconds=0.0, construct=0.01, search=0.02):
    return {"method": "m", "seed": None, "initial_cost": cost * 1.1, "initial_gap": 10.0,
            "initial_open": 5, "final_cost": cost, "final_gap": 0.0, "final_open": 5,
            "iterations": 2, "moves": {"insert": 0, "delete": 1, "swap": 1},
            "construct_seconds": construct, "search_seconds": search,
            "lp_seconds": lp_seconds}


def _ms(cost, lp_seconds=0.0, construct=0.01, search=0.02, best_at=1, restarts=32):
    return {"method": "ms", "seeds": list(range(1, restarts + 1)), "best_cost": cost,
            "best_gap": 0.0, "best_open": 5, "best_found_at": best_at,
            "trajectory": [0.0] * restarts, "trajectory_costs": [cost] * restarts,
            "per_restart_gaps": [0.0] * restarts, "total_iterations": 10,
            "moves": {"insert": 0, "delete": 1, "swap": 1},
            "construct_seconds": construct, "search_seconds": search,
            "lp_seconds": lp_seconds}


def _rec(optimum=100.0, lp_bound=None, proven=True, a=None, ap=None, d=None,
         b=None, c=None, b_time=0.05, c_time=0.15):
    lp_bound = optimum if lp_bound is None else lp_bound
    return {
        "instance_path": "data/cap134.txt", "n_facilities": 50, "n_customers": 50,
        "lp": {"bound": lp_bound, "solve_seconds": 0.018, "n_facilities": 50,
               "n_integral_open": 10, "n_fractional": 0, "n_near_zero": 40,
               "sum_y": 10.0, "rounded_open_count": 10},
        "reference": {"value": optimum, "lp_bound": lp_bound, "proven": proven,
                      "ip_status": "Optimal" if proven else "Not Solved",
                      "ip_seconds": 0.4, "incumbent": optimum},
        "arms": {
            "lp_rounding_control": _arm(a if a is not None else optimum),
            "lp_rounding_plus_search": _arm(ap if ap is not None else optimum),
            "local_search_only": _arm(d if d is not None else optimum),
            "lp_biased_multistart": _ms(b if b is not None else optimum, search=b_time),
            "alpha_grasp_multistart": _ms(c if c is not None else optimum, search=c_time),
        },
    }


def _render(rec, restarts=32, alpha=0.2):
    return render_report(rec, list(range(1, restarts + 1)), ENV,
                         "2026-09-12 00:00 UTC", "testcommit", 0.33, alpha)


def _prose(md):
    stripped = [ln.lstrip().removeprefix("> ").removeprefix(">") for ln in md.splitlines()]
    return " ".join(" ".join(stripped).split())


class TestIntegralRelaxationIsCalledOut:
    def test_a_zero_duality_gap_is_stated_as_solving_the_instance(self):
        md = _prose(_render(_rec(optimum=100.0, lp_bound=100.0)))
        assert "The LP bound equals the integer optimum" in md
        assert "Solving the relaxation solves the instance" in md
        assert "cannot discriminate between the methods on quality" in md

    def test_a_nonzero_duality_gap_is_reported_as_leaving_work(self):
        """COUNTERWEIGHT: the claim must be conditional on the measured gap."""
        md = _prose(_render(_rec(optimum=110.0, lp_bound=100.0)))
        assert "duality gap, so the heuristic layer has something to contribute" in md
        assert "equals the integer optimum" not in md
        assert "cannot discriminate" not in md


class TestTieIsNotDressedUpAsAWin:
    def test_all_arms_reaching_the_optimum_is_said_plainly(self):
        md = _prose(_render(_rec()))
        assert "Every arm reaches the optimum" in md
        assert "including the one that never reads the LP" in md

    def test_quality_is_called_indistinguishable_when_it_is(self):
        md = _prose(_render(_rec()))
        assert "indistinguishable" in md

    def test_a_real_quality_difference_is_reported_as_one(self):
        """COUNTERWEIGHT: if an arm genuinely wins, say so."""
        md = _prose(_render(_rec(optimum=100.0, b=100.0, c=105.0)))
        assert "B ahead" in md
        assert "indistinguishable" not in md

    def test_partial_success_lists_only_the_arms_that_solved_it(self):
        md = _prose(_render(_rec(optimum=100.0, a=120.0, d=130.0)))
        assert "Arms reaching the optimum:" in md
        assert "Every arm reaches the optimum" not in md


class TestTimingComparisonFollowsTheData:
    def test_faster_and_slower_flip_with_the_measured_times(self):
        faster = _prose(_render(_rec(b_time=0.02, c_time=0.20)))
        slower = _prose(_render(_rec(b_time=0.20, c_time=0.02)))
        assert "faster" in faster and "slower" not in faster
        assert "slower" in slower and "faster" not in slower

    def test_the_lp_cost_is_attributed_to_the_arm_that_pays_it(self):
        md = _prose(_render(_rec()))
        assert "the LP solve it alone pays for" in md


class TestReferenceHonesty:
    def test_a_proven_optimum_permits_the_phrase_optimality_gap(self):
        md = _prose(_render(_rec(proven=True)))
        assert "true optimality gaps" in md
        assert "overstate" not in md.lower()

    def test_an_unproven_optimum_does_not(self):
        md = _prose(_render(_rec(proven=False)))
        assert "Integer optimum not proven" in md
        assert "overstate the true optimality gap" in md


class TestEndToEnd:
    def test_instance_path_resolution(self, tmp_path):
        assert resolve_instance_path(("does/not/exist.txt",)) is None
        real = tmp_path / "cap.txt"
        real.write_text("x")
        assert resolve_instance_path((str(real),)) == str(real)

    def test_run_benchmark_produces_every_arm_and_a_reference(self):
        path = resolve_instance_path()
        if path is None:
            pytest.skip("cap134.txt not present in this checkout")
        rec = run_benchmark(path, seeds=[1, 2, 3], alpha_value=0.2, ip_time_limit=120)

        assert set(rec["arms"]) == {
            "lp_rounding_control", "lp_rounding_plus_search", "local_search_only",
            "lp_biased_multistart", "alpha_grasp_multistart",
        }
        # cap134's relaxation is integral, so the bound must equal the optimum.
        assert rec["reference"]["proven"]
        assert rec["reference"]["value"] == pytest.approx(rec["lp"]["bound"], rel=1e-9)
        # And no arm may beat it.
        for key in ("lp_rounding_plus_search", "local_search_only"):
            assert rec["arms"][key]["final_cost"] >= rec["reference"]["value"] - 1e-6

    def test_main_writes_both_artifacts(self, tmp_path, monkeypatch):
        if resolve_instance_path() is None:
            pytest.skip("cap134.txt not present in this checkout")
        out = tmp_path / "out"
        monkeypatch.setattr(sys, "argv", [
            "run_cap134.py", "--restarts", "3", "--ip-time-limit", "120",
            "--out-dir", str(out),
        ])
        main()

        md = (out / "cap134_results.md").read_text()
        payload = json.loads((out / "cap134_results.json").read_text())
        assert payload["restarts"] == 3
        assert payload["generated_by"].endswith("render_report")
        for heading in ("## 1. Instance and relaxation", "## 2. Results",
                        "## 3. What this instance can and cannot show",
                        "## 4. Reproduction"):
            assert heading in md

    def test_missing_instance_exits_non_zero(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        monkeypatch.setattr(sys, "argv", ["run_cap134.py", "--restarts", "2"])
        with pytest.raises(SystemExit) as exc:
            main()
        assert exc.value.code == 1


def test_no_arm_reaching_the_reference_is_said_plainly():
    """
    COUNTERWEIGHT to the "every arm reaches the optimum" branch. If nothing
    attains the reference the report must say so, not fall silent -- on this
    instance that would indicate a defect, since the relaxation is integral.
    """
    rec = _rec(optimum=100.0, a=110.0, ap=110.0, d=110.0, b=110.0, c=110.0)
    md = _prose(_render(rec))
    assert "No arm reached the reference value" in md
    assert "Every arm reaches the optimum" not in md
    assert "Arms reaching the optimum:" not in md
