"""
Unit tests for the full-study orchestrator.

`run_full_study` is the module that decides what the headline of this project
is allowed to say. Two things must hold and are asserted here:

1.  **Aggregation is faithful.** Mean gaps, head-to-head counts and
    "at reference" tallies are computed from per-instance records, not
    restated from a summary someone typed.
2.  **The narrative is computed.** Every comparative word in the rendered
    markdown -- "widens", "ahead at every size", "in full up to N" -- must flip
    when the underlying measurements flip. A report whose prose survives
    inverted inputs is a report that was written by hand.

The second point is the one that has broken before: an earlier revision of a
sibling report hardcoded "under 1%" above a table that showed 10.51%.
"""

import json

import pytest

from scripts.run_full_study import (
    DEFAULT_PLAN,
    IP_TIME_LIMIT,
    _arm_cost,
    _arm_seconds,
    _cmp,
    _median,
    render_analysis,
    render_california_table,
    render_kg_stage_table,
    stage_plan,
    summarise_california,
    summarise_kg,
)

ENV = {"python": "3.14.0", "platform": "linux"}


def _arm(cost, seconds=1.0, multistart=False, best_at=1, per_restart=None):
    base = {"lp_seconds": seconds / 2, "construct_seconds": seconds / 4,
            "search_seconds": seconds / 4}
    if multistart:
        base.update({"best_cost": cost, "best_found_at": best_at,
                     "per_restart_gaps": per_restart or [0.0]})
    else:
        base["final_cost"] = cost
    return base


def _record(name, klass, reference, proven, costs, fractional=10, best_at=1):
    """One instance record shaped like `run_koerkel_ghosh` writes it."""
    a, ap, d, b, c = costs
    return {
        "name": name, "size": 250, "klass": klass, "symmetric": True, "index": 1,
        "lp": {"bound": reference * 0.9, "n_facilities": 20,
               "n_fractional": fractional, "solve_seconds": 0.5},
        "reference": {"value": reference, "lp_bound": reference * 0.9,
                      "proven": proven, "ip_status": "Optimal" if proven else "Not Solved",
                      "ip_seconds": 10.0, "incumbent": None},
        "best_cost_found": min(costs),
        "arms": {
            "lp_rounding_control": _arm(a),
            "lp_rounding_plus_search": _arm(ap),
            "local_search_only": _arm(d),
            "lp_biased_multistart": _arm(b, multistart=True, best_at=best_at),
            "alpha_grasp_multistart": _arm(c, multistart=True, best_at=3),
        },
    }


@pytest.fixture
def kg_payload():
    """Two instances: B wins one outright, ties the other."""
    return {
        "size": 250, "restarts": 32,
        "records": [
            # reference 100: A=110, A+=101, D=102, B=100, C=103
            _record("gs250a-1", "a", 100.0, True, (110.0, 101.0, 102.0, 100.0, 103.0)),
            # reference 200: everything at the optimum
            _record("gs250b-1", "b", 200.0, True, (200.0, 200.0, 200.0, 200.0, 200.0)),
        ],
        "stage_seconds": 120.0,
    }


@pytest.fixture
def ca_payload():
    return {
        "instance": {"n_facilities": 2000, "n_customers": 2000},
        "seeds": list(range(1, 11)),
        "lp_profile": {"bound": 1000.0, "n_facilities": 2000, "n_fractional": 40,
                       "solve_seconds": 30.0},
        "reference": {"value": 1000.0, "lp_bound": 1000.0, "proven": False,
                      "ip_status": "not attempted", "ip_seconds": 0.0, "incumbent": None},
        "arms": {
            "lp_rounding_control": _arm(1020.0, seconds=40),
            "lp_rounding_plus_search": _arm(1004.0, seconds=120),
            "local_search_only": _arm(1004.0, seconds=500),
            "lp_biased_multistart": _arm(1003.0, seconds=2000, multistart=True,
                                         best_at=2, per_restart=[0.4, 0.3, 0.42]),
            "alpha_grasp_multistart": _arm(1006.0, seconds=4000, multistart=True,
                                           best_at=4, per_restart=[0.9, 0.6, 1.2]),
        },
        "stage_seconds": 6800.0,
    }


class TestArmAccessors:
    def test_a_deterministic_arm_reports_its_final_cost(self):
        assert _arm_cost({"final_cost": 42.0}) == 42.0

    def test_a_multistart_arm_reports_its_best_not_its_last(self):
        """COUNTERWEIGHT: best-of-N is the whole point of the restart loop."""
        assert _arm_cost({"best_cost": 7.0, "final_cost": 99.0}) == 7.0

    def test_arm_time_sums_all_three_phases(self):
        assert _arm_seconds(
            {"lp_seconds": 1.0, "construct_seconds": 2.0, "search_seconds": 4.0}) == 7.0

    def test_median_of_an_empty_sequence_is_zero_not_an_exception(self):
        assert _median([]) == 0.0
        assert _median([5, 1, 3]) == 3


class TestComparativeWord:
    def test_it_names_the_smaller_side(self):
        assert _cmp(1.0, 2.0, "lo", "hi") == "lo"
        assert _cmp(2.0, 1.0, "lo", "hi") == "hi"

    def test_a_tie_within_tolerance_is_reported_as_a_tie(self):
        assert _cmp(1.0, 1.0 + 1e-12, "lo", "hi", tie="level") == "level"

    def test_a_difference_outside_tolerance_is_not_a_tie(self):
        """COUNTERWEIGHT: the tolerance must not swallow real differences."""
        assert _cmp(1.0, 1.1, "lo", "hi", tie="level") == "lo"


class TestSummariseKG:
    def test_mean_gap_is_computed_from_the_records(self, kg_payload):
        s = summarise_kg(kg_payload)
        # instance 1: (101-100)/100 = 1.0%, instance 2: 0.0% -> mean 0.5%
        assert s["a_plus"] == pytest.approx(0.5)
        assert s["no_lp"] == pytest.approx(1.0)      # (102-100)/100 = 2%, then 0%
        assert s["lp_biased"] == pytest.approx(0.0)
        assert s["alpha"] == pytest.approx(1.5)      # (103-100)/100 = 3%, then 0%

    def test_at_reference_counts_only_exact_hits(self, kg_payload):
        s = summarise_kg(kg_payload)
        assert s["b_at_reference"] == 2      # B is optimal on both
        assert s["c_at_reference"] == 1      # C only on the second

    def test_head_to_head_counts_wins_losses_and_leaves_ties_out(self, kg_payload):
        s = summarise_kg(kg_payload)
        assert (s["b_beats_c"], s["c_beats_b"]) == (1, 0)
        assert (s["b_beats_ap"], s["ap_beats_b"]) == (1, 0)
        # The tied instance is counted in neither column.
        assert s["b_beats_c"] + s["c_beats_b"] < s["instances"]

    def test_duality_gap_uses_proven_instances_only(self, kg_payload):
        kg_payload["records"][1]["reference"]["proven"] = False
        s = summarise_kg(kg_payload)
        assert s["proven"] == 1
        # Only the first record contributes: (100 - 90) / 90.
        assert s["mean_duality_gap"] == pytest.approx(10 / 90 * 100)

    def test_duality_gap_is_none_when_nothing_was_proven(self, kg_payload):
        """An unproven reference cannot yield a duality gap; it must not be faked."""
        for r in kg_payload["records"]:
            r["reference"]["proven"] = False
        s = summarise_kg(kg_payload)
        assert s["proven"] == 0
        assert s["mean_duality_gap"] is None

    def test_per_arm_spread_reports_best_and_worst_not_just_the_mean(self, kg_payload):
        s = summarise_kg(kg_payload)
        arm = s["per_arm"]["alpha_grasp_multistart"]
        assert arm["best_gap"] == pytest.approx(0.0)
        assert arm["worst_gap"] == pytest.approx(3.0)
        assert arm["mean_gap"] == pytest.approx(1.5)

    def test_rows_carry_one_entry_per_instance(self, kg_payload):
        s = summarise_kg(kg_payload)
        assert [r["name"] for r in s["rows"]] == ["gs250a-1", "gs250b-1"]
        assert s["rows"][0]["gaps"]["lp_biased_multistart"] == pytest.approx(0.0)


class TestSummariseCalifornia:
    def test_it_reads_size_from_the_instance_block(self, ca_payload):
        s = summarise_california(ca_payload)
        assert (s["size"], s["n_customers"]) == (2000, 2000)

    def test_restarts_are_the_number_of_seeds_actually_run(self, ca_payload):
        assert summarise_california(ca_payload)["restarts"] == 10

    def test_an_unattempted_ip_is_not_counted_as_proven(self, ca_payload):
        s = summarise_california(ca_payload)
        assert s["proven"] == 0
        assert s["mean_duality_gap"] is None


class TestStagePlan:
    def test_known_sizes_use_the_published_budget(self):
        assert stage_plan(250) == DEFAULT_PLAN[250]
        sizes = (100, 150, 200, 250, 500, 750)
        # Neither budget may vary with size: a best-of-N is only comparable to
        # another best-of-N, and a proof attempt only to an equally long one.
        assert len({stage_plan(n)["restarts"] for n in sizes}) == 1
        assert len({stage_plan(n)["ip_time_limit"] for n in sizes}) == 1

    def test_an_unlisted_size_still_gets_a_budget(self):
        plan = stage_plan(31337)
        assert plan["restarts"] >= 1 and plan["ip_time_limit"] > 0


class TestRenderedNarrative:
    def test_the_stage_table_states_when_every_optimum_was_proven(self, kg_payload):
        md = "\n".join(render_kg_stage_table(summarise_kg(kg_payload)))
        assert "true optimality gaps" in md
        assert "OVERSTATE" not in md.upper() or "overstate" not in md

    def test_the_stage_table_warns_when_nothing_was_proven(self, kg_payload):
        """The inverse of the previous test must produce the inverse wording."""
        for r in kg_payload["records"]:
            r["reference"]["proven"] = False
        md = "\n".join(render_kg_stage_table(summarise_kg(kg_payload)))
        assert "No optimum proven" in md
        assert "overstate" in md
        assert "true optimality gaps" not in md

    def test_a_mixed_reference_is_reported_as_mixed(self, kg_payload):
        kg_payload["records"][1]["reference"]["proven"] = False
        md = "\n".join(render_kg_stage_table(summarise_kg(kg_payload)))
        assert "Mixed reference" in md
        assert "1 of 2" in md

    def test_the_best_arm_named_in_prose_is_the_best_arm_in_the_table(self, kg_payload):
        md = "\n".join(render_kg_stage_table(summarise_kg(kg_payload)))
        assert "Best mean gap at this size: **B · MS-LP-GRASP (LP-biased)**" in md
        # Flip it: make B the worst and the sentence must follow.
        for r in kg_payload["records"]:
            r["arms"]["lp_biased_multistart"]["best_cost"] = r["reference"]["value"] * 10
        md2 = "\n".join(render_kg_stage_table(summarise_kg(kg_payload)))
        assert "Best mean gap at this size: **B" not in md2

    def test_california_table_flags_a_missing_optimum(self, ca_payload):
        md = "\n".join(render_california_table(summarise_california(ca_payload)))
        assert "integer optimum is not available" in md
        assert "overstate" in md
        assert "ranking between arms is still valid" in md

    def test_california_table_compares_b_worst_against_c_mean(self, ca_payload):
        md = "\n".join(render_california_table(summarise_california(ca_payload)))
        # B worst 0.42 vs C mean 0.90 -> B ahead.
        assert "B ahead even at its worst" in md

    def test_that_comparison_flips_when_the_numbers_flip(self, ca_payload):
        """COUNTERWEIGHT: the sentence must be measured, not asserted."""
        ca_payload["arms"]["lp_biased_multistart"]["per_restart_gaps"] = [9.0]
        md = "\n".join(render_california_table(summarise_california(ca_payload)))
        assert "B ahead even at its worst" not in md
        assert "C ahead" in md


class TestRenderAnalysis:
    def _two_sizes(self, kg_payload):
        small = summarise_kg(kg_payload)
        big = summarise_kg({**kg_payload, "size": 750, "restarts": 8})
        big["size"] = 750
        for row in big["rows"]:
            row["proven"] = False
        big["proven"] = 0
        big["mean_duality_gap"] = None
        return [small, big]

    def test_it_names_the_largest_fully_proven_size(self, kg_payload):
        md = render_analysis(self._two_sizes(kg_payload), None, "now", "abc", 60.0, ENV)
        assert "in full up to 250×250" in md
        assert "At **750×750** CBC proved **none**" in md

    def test_it_totals_head_to_head_across_every_size(self, kg_payload):
        md = render_analysis(self._two_sizes(kg_payload), None, "now", "abc", 60.0, ENV)
        assert "Across all 4 Körkel-Ghosh instances" in md
        assert "**B beats C on 2, loses on 0, ties on 2**" in md

    def test_the_verdict_reverses_when_c_wins(self, kg_payload):
        """COUNTERWEIGHT: the headline sentence is derived, not decided."""
        for r in kg_payload["records"]:
            r["arms"]["lp_biased_multistart"]["best_cost"] = r["reference"]["value"] * 2
        md = render_analysis(self._two_sizes(kg_payload), None, "now", "abc", 60.0, ENV)
        assert "the classical baseline overtakes" in md
        assert "The LP bias holds its advantage" not in md

    def test_an_inconsistent_lp_advantage_is_admitted_not_averaged_away(self, kg_payload):
        stages = self._two_sizes(kg_payload)
        stages[1]["no_lp"], stages[1]["a_plus"] = 0.0, 5.0    # LP hurts at 750
        md = render_analysis(stages, None, "now", "abc", 60.0, ENV)
        assert "not consistent across sizes" in md
        assert "ahead **at every size tested**" not in md

    def test_a_consistent_lp_advantage_is_stated_plainly(self, kg_payload):
        md = render_analysis(self._two_sizes(kg_payload), None, "now", "abc", 60.0, ENV)
        assert "ahead **at every size tested**" in md

    def test_the_margin_trend_is_measured(self, kg_payload):
        stages = self._two_sizes(kg_payload)
        stages[1]["alpha"], stages[1]["lp_biased"] = 9.0, 0.0   # margin grows with size
        md = render_analysis(stages, None, "now", "abc", 60.0, ENV)
        assert "margin **widens** with size" in md
        stages[1]["alpha"], stages[1]["lp_biased"] = 0.1, 0.0   # margin shrinks
        md2 = render_analysis(stages, None, "now", "abc", 60.0, ENV)
        assert "margin **narrows** with size" in md2

    def test_california_appears_in_every_cross_size_table(self, kg_payload, ca_payload):
        md = render_analysis(self._two_sizes(kg_payload),
                             summarise_california(ca_payload), "now", "abc", 60.0, ENV)
        assert "2000×2000 (California)" in md
        assert "out of reach" in md
        assert "California Housing 2000×2000" in md
        assert "`output/california_4M_results.md`" in md

    def test_a_constant_restart_budget_is_stated_as_making_rows_comparable(self, kg_payload):
        stages = self._two_sizes(kg_payload)
        stages[1]["restarts"] = stages[0]["restarts"]
        md = render_analysis(stages, None, "now", "abc", 60.0, ENV)
        assert "restart budget is the same at every size** (best of 32)" in md
        assert "not directly comparable between rows" not in md

    def test_an_uneven_restart_budget_is_flagged_as_not_comparable(self, kg_payload):
        """COUNTERWEIGHT: the comparability claim must depend on the budgets."""
        stages = self._two_sizes(kg_payload)
        stages[1]["restarts"] = 8
        md = render_analysis(stages, None, "now", "abc", 60.0, ENV)
        assert "**not** constant across sizes (8–32)" in md
        assert "searched less thoroughly" in md
        assert "restart budget is the same at every size" not in md

    def test_an_uneven_exact_solve_budget_is_disclosed(self, kg_payload):
        import scripts.run_full_study as mod
        stages = self._two_sizes(kg_payload)
        # The shipped plan is uniform, so force a difference to exercise the branch.
        monkeypatch = {750: {**mod.stage_plan(750), "ip_time_limit": 60.0}}
        original = dict(mod.DEFAULT_PLAN)
        mod.DEFAULT_PLAN.update(monkeypatch)
        try:
            md = render_analysis(stages, None, "now", "abc", 60.0, ENV)
        finally:
            mod.DEFAULT_PLAN.clear()
            mod.DEFAULT_PLAN.update(original)
        assert "exact-solve** budget does vary (60–1800 s" in md

    def test_a_uniform_exact_solve_budget_is_not_mentioned(self, kg_payload):
        """
        COUNTERWEIGHT: no caveat where there is nothing to caveat. This is the
        shipped configuration -- every rung gets the same thirty minutes.
        """
        md = render_analysis(self._two_sizes(kg_payload), None, "now", "abc", 60.0, ENV)
        assert "exact-solve** budget does vary" not in md

    def test_every_stage_gets_its_own_results_table(self, kg_payload):
        md = render_analysis(self._two_sizes(kg_payload), None, "now", "abc", 60.0, ENV)
        assert md.count("| Arm | Type | Mean") == 2
        assert "### Körkel-Ghosh 250×250" in md
        assert "### Körkel-Ghosh 750×750" in md

    def test_metadata_is_carried_through_verbatim(self, kg_payload):
        md = render_analysis(self._two_sizes(kg_payload), None,
                             "2026-01-01 00:00 UTC", "deadbeef", 600.0, ENV)
        assert "2026-01-01 00:00 UTC" in md
        assert "`deadbeef`" in md
        assert "10.0 min" in md
        assert "3.14.0" in md


class TestRemainingNarrativeBranches:
    def test_a_proven_california_optimum_is_reported_as_such(self, ca_payload):
        """COUNTERWEIGHT to `test_california_table_flags_a_missing_optimum`."""
        ca_payload["reference"].update({"proven": True, "ip_status": "Optimal"})
        md = "\n".join(render_california_table(summarise_california(ca_payload)))
        assert "**Gaps here are true optimality gaps.**" in md
        assert "integer optimum is not available" not in md

        cross = render_analysis([], summarise_california(ca_payload),
                                "now", "abc", 1.0, ENV)
        assert "closed exactly" in cross
        assert "out of reach" not in cross

    def test_a_partly_proven_size_is_reported_as_partly_proven(self, kg_payload):
        kg_payload["records"][1]["reference"]["proven"] = False
        md = render_analysis([summarise_kg(kg_payload)], None, "now", "abc", 1.0, ENV)
        assert "CBC proved 1 of 2 optima" in md
        assert "overstate the true gap" in md
        assert "in full up to" not in md

    def test_a_dead_heat_between_b_and_c_is_called_a_dead_heat(self, kg_payload):
        """Neither arm may be declared the winner when the wins are equal."""
        recs = kg_payload["records"]
        recs[1]["arms"]["lp_biased_multistart"]["best_cost"] = 210.0   # C wins here
        recs[1]["arms"]["alpha_grasp_multistart"]["best_cost"] = 200.0
        md = render_analysis([summarise_kg(kg_payload)], None, "now", "abc", 1.0, ENV)
        assert "the two arms are level (1 wins each, 0 ties)" in md
        assert "holds its advantage" not in md
        assert "overtakes" not in md

    def test_an_lp_that_hurts_everywhere_is_stated_as_such(self, kg_payload):
        """COUNTERWEIGHT: the report must be able to say the LP does not pay off."""
        s = summarise_kg(kg_payload)
        s["a_plus"], s["no_lp"] = 5.0, 0.0
        md = render_analysis([s], None, "now", "abc", 1.0, ENV)
        assert "**not** paying for itself as a seed" in md
        assert "ahead **at every size tested**" not in md

    def test_a_steady_margin_is_neither_widening_nor_narrowing(self, kg_payload):
        a, b = summarise_kg(kg_payload), summarise_kg(kg_payload)
        b["size"] = 750
        md = render_analysis([a, b], None, "now", "abc", 1.0, ENV)
        assert "margin **holds steady** with size" in md


class TestStageLaunching:
    """
    The orchestrator shells out to the per-stage scripts. What matters is that
    the budget it advertises in §6 is the budget it actually passes on the
    command line -- a report that says "32 restarts" over a 4-restart run is
    worse than no report.
    """

    def _fake_run(self, captured, stage_json, ca_json=None):
        import subprocess as _sp
        real = _sp.run

        def run(cmd, *args, **kwargs):
            # `_git_commit` also goes through subprocess.run; only intercept the
            # stage launches, or the metadata block would break.
            if not (isinstance(cmd, list) and "-m" in cmd):
                return real(cmd, *args, **kwargs)
            captured.append(cmd)
            import os as _os
            if "scripts.run_koerkel_ghosh" in cmd:
                out = cmd[cmd.index("--out-dir") + 1]
                _os.makedirs(out, exist_ok=True)
                with open(_os.path.join(out, "koerkel_ghosh_results.json"), "w") as fh:
                    json.dump(stage_json, fh)
            else:
                out = cmd[cmd.index("--out-dir") + 1]
                _os.makedirs(out, exist_ok=True)
                with open(_os.path.join(out, "california_4M_results.json"), "w") as fh:
                    json.dump(ca_json, fh)
            return None
        return run

    def test_the_command_line_carries_the_advertised_budget(self, kg_payload, tmp_path,
                                                            monkeypatch):
        import scripts.run_full_study as mod
        captured = []
        monkeypatch.setattr(mod.subprocess, "run",
                            self._fake_run(captured, kg_payload))
        cfg = mod.stage_plan(250)
        payload = mod.run_stage(250, cfg, str(tmp_path))
        cmd = captured[0]
        assert cmd[cmd.index("--size") + 1] == "250"
        assert cmd[cmd.index("--restarts") + 1] == str(cfg["restarts"])
        assert cmd[cmd.index("--instances") + 1] == str(cfg["instances"])
        assert "--no-exact-ip" not in cmd
        assert payload["stage_seconds"] >= 0.0

    def test_every_rung_attempts_a_proof_including_the_largest(self, kg_payload,
                                                               tmp_path, monkeypatch):
        """
        No size is exempted from the exact solve. "CBC did not close this in
        thirty minutes" is a measurement and belongs in the table; "we did not
        try" is not a result. An earlier plan skipped 500 and 750 on the
        prediction that they would time out -- a prediction is not evidence.
        """
        import scripts.run_full_study as mod
        for size in (100, 150, 200, 250, 500, 750):
            plan = mod.stage_plan(size)
            assert plan["attempt_exact"], size
            assert plan["ip_time_limit"] == mod.IP_TIME_LIMIT, size

            captured = []
            monkeypatch.setattr(mod.subprocess, "run",
                                self._fake_run(captured, kg_payload))
            mod.run_stage(size, plan, str(tmp_path))
            assert "--no-exact-ip" not in captured[0], size
            assert captured[0][captured[0].index("--ip-time-limit") + 1] == str(
                mod.IP_TIME_LIMIT)

    def test_the_skip_flag_still_works_when_a_caller_asks_for_it(self, kg_payload,
                                                                 tmp_path, monkeypatch):
        """
        COUNTERWEIGHT: the plan no longer uses it, but the plumbing must still
        be correct for a caller that does (e.g. the California instance).
        """
        import scripts.run_full_study as mod
        captured = []
        monkeypatch.setattr(mod.subprocess, "run", self._fake_run(captured, kg_payload))
        mod.run_stage(500, {**mod.stage_plan(500), "attempt_exact": False}, str(tmp_path))
        assert "--no-exact-ip" in captured[0]

    def test_skipping_the_exact_solve_is_passed_through_not_merely_recorded(
            self, kg_payload, tmp_path, monkeypatch):
        import scripts.run_full_study as mod
        captured = []
        monkeypatch.setattr(mod.subprocess, "run", self._fake_run(captured, kg_payload))
        mod.run_stage(500, {"instances": 1, "restarts": 4, "ip_time_limit": 1.0,
                            "attempt_exact": False}, str(tmp_path))
        assert "--no-exact-ip" in captured[0]

    def test_california_is_launched_at_full_size(self, ca_payload, tmp_path, monkeypatch):
        import scripts.run_full_study as mod
        captured = []
        monkeypatch.setattr(mod.subprocess, "run",
                            self._fake_run(captured, None, ca_payload))
        payload = mod.run_california(7, str(tmp_path))
        cmd = captured[0]
        assert cmd[cmd.index("--size") + 1] == "2000"
        assert cmd[cmd.index("--restarts") + 1] == "7"
        assert payload["instance"]["n_facilities"] == 2000

    def test_main_writes_both_artifacts_and_the_json_matches_the_markdown(
            self, kg_payload, ca_payload, tmp_path, monkeypatch):
        import scripts.run_full_study as mod
        captured = []
        monkeypatch.setattr(mod.subprocess, "run",
                            self._fake_run(captured, kg_payload, ca_payload))
        mod.main(["--sizes", "250", "--california-restarts", "10",
                  "--out-dir", str(tmp_path)])

        md = (tmp_path / "full_study.md").read_text()
        data = json.loads((tmp_path / "full_study.json").read_text())
        assert "# Full Study" in md
        assert "### Körkel-Ghosh 250×250" in md
        assert "### California Housing 2000×2000" in md
        # The summary written to JSON is the one the markdown was rendered from.
        assert data["stages"][0]["lp_biased"] == pytest.approx(0.0)
        assert f"**{data['stages'][0]['lp_biased']:.4f}%**" in md
        assert data["plan"]["250"] == DEFAULT_PLAN[250]
        assert data["california"]["restarts"] == 10

    def test_california_can_be_skipped_without_breaking_the_report(
            self, kg_payload, tmp_path, monkeypatch):
        """COUNTERWEIGHT: the ladder must still render on its own."""
        import scripts.run_full_study as mod
        monkeypatch.setattr(mod.subprocess, "run", self._fake_run([], kg_payload))
        mod.main(["--sizes", "250", "--skip-california", "--out-dir", str(tmp_path)])
        md = (tmp_path / "full_study.md").read_text()
        assert "### Körkel-Ghosh 250×250" in md
        assert "California Housing" not in md
        assert json.loads((tmp_path / "full_study.json").read_text())["california"] is None


class TestReRenderingWithoutReRunning:
    """
    The ladder costs hours, most of it exact-solve attempts that time out.
    Re-rendering must read what those runs already wrote rather than paying for
    them twice -- and must produce the same analysis either way.
    """

    def _write_stage(self, out_dir, payload):
        import os
        d = os.path.join(out_dir, f"kg{payload['size']}")
        os.makedirs(d, exist_ok=True)
        with open(os.path.join(d, "koerkel_ghosh_results.json"), "w") as fh:
            json.dump(payload, fh)

    def test_a_stage_is_read_back_exactly_as_it_was_written(self, kg_payload, tmp_path):
        from scripts.run_full_study import load_stage
        self._write_stage(str(tmp_path), kg_payload)
        assert summarise_kg(load_stage(250, str(tmp_path))) == summarise_kg(kg_payload)

    def test_a_stage_without_stage_seconds_falls_back_to_its_own_wall_clock(
            self, kg_payload, tmp_path):
        """Sidecars written by the stage script carry `wall_seconds`, not `stage_seconds`."""
        from scripts.run_full_study import load_stage
        payload = {k: v for k, v in kg_payload.items() if k != "stage_seconds"}
        payload["wall_seconds"] = 1234.0
        self._write_stage(str(tmp_path), payload)
        assert load_stage(250, str(tmp_path))["stage_seconds"] == 1234.0

    def test_california_is_read_back_too(self, ca_payload, tmp_path):
        from scripts.run_full_study import load_california
        with open(tmp_path / "california_4M_results.json", "w") as fh:
            json.dump(ca_payload, fh)
        assert summarise_california(
            load_california(str(tmp_path))) == summarise_california(ca_payload)

    def test_from_stages_runs_no_benchmark_at_all(self, kg_payload, ca_payload,
                                                  tmp_path, monkeypatch):
        """The point of the flag: nothing is recomputed."""
        import scripts.run_full_study as mod
        self._write_stage(str(tmp_path), kg_payload)
        with open(tmp_path / "california_4M_results.json", "w") as fh:
            json.dump(ca_payload, fh)

        def refuse(*_a, **_k):
            raise AssertionError("--from-stages must not launch a benchmark")

        monkeypatch.setattr(mod, "run_stage", refuse)
        monkeypatch.setattr(mod, "run_california", refuse)
        mod.main(["--sizes", "250", "--from-stages", "--out-dir", str(tmp_path)])

        md = (tmp_path / "full_study.md").read_text()
        assert "### Körkel-Ghosh 250×250" in md
        assert "### California Housing 2000×2000" in md

    def test_a_rung_can_be_added_to_an_existing_ladder(self, kg_payload, tmp_path,
                                                       monkeypatch):
        """
        The case this exists for: locating where provability stops means adding
        smaller sizes to a ladder whose expensive rungs are already measured.
        """
        import scripts.run_full_study as mod
        self._write_stage(str(tmp_path), kg_payload)
        small = {**kg_payload, "size": 100}
        for r in small["records"]:
            r["size"] = 100
        self._write_stage(str(tmp_path), small)

        monkeypatch.setattr(mod, "run_stage", lambda *a, **k: (_ for _ in ()).throw(
            AssertionError("must not re-run")))
        mod.main(["--sizes", "100", "250", "--from-stages", "--skip-california",
                  "--out-dir", str(tmp_path)])

        md = (tmp_path / "full_study.md").read_text()
        assert "### Körkel-Ghosh 100×100" in md
        assert "### Körkel-Ghosh 250×250" in md
        data = json.loads((tmp_path / "full_study.json").read_text())
        assert [s["size"] for s in data["stages"]] == [100, 250]


class TestNotRecomputingFinishedRungs:
    """
    A rung costs hours. If the ladder is interrupted at a later rung, restarting
    it must not throw away the rungs that already finished -- that happened, and
    cost a 5h33m rung. But reuse is only valid when the sidecar on disk came
    from the configuration now being asked for.
    """

    def _write(self, out_dir, payload):
        import os
        d = os.path.join(out_dir, f"kg{payload['size']}")
        os.makedirs(d, exist_ok=True)
        with open(os.path.join(d, "koerkel_ghosh_results.json"), "w") as fh:
            json.dump(payload, fh)

    @pytest.fixture
    def cfg(self):
        from scripts.run_full_study import stage_plan
        return {**stage_plan(250), "instances": 1}      # 1 x 6 = 6 records

    @pytest.fixture
    def finished(self, kg_payload):
        """A sidecar with the full complement of 6 records for instances=1."""
        recs = kg_payload["records"]
        return {**kg_payload, "records": [dict(recs[0], name=f"i{i}") for i in range(6)]}

    def test_a_matching_sidecar_is_reused_without_launching_anything(
            self, finished, cfg, tmp_path, monkeypatch):
        import scripts.run_full_study as mod
        self._write(str(tmp_path), finished)
        monkeypatch.setattr(mod.subprocess, "run", lambda *a, **k: (_ for _ in ()).throw(
            AssertionError("a completed rung must not be re-run")))
        payload = mod.run_stage(250, cfg, str(tmp_path))
        assert len(payload["records"]) == 6
        assert payload["stage_seconds"] == finished["stage_seconds"]

    def test_a_sidecar_with_too_few_instances_is_not_reused(self, kg_payload, cfg,
                                                            tmp_path):
        """A partial rung is not a finished rung."""
        from scripts.run_full_study import completed_stage
        self._write(str(tmp_path), kg_payload)          # only 2 records
        assert completed_stage(250, cfg, str(tmp_path)) is None

    def test_a_sidecar_from_a_different_restart_budget_is_not_reused(
            self, finished, cfg, tmp_path):
        """COUNTERWEIGHT: best-of-4 records may not stand in for best-of-32."""
        from scripts.run_full_study import completed_stage
        self._write(str(tmp_path), {**finished, "restarts": 4})
        assert completed_stage(250, cfg, str(tmp_path)) is None

    def test_a_sidecar_for_a_different_size_is_not_reused(self, finished, cfg, tmp_path):
        from scripts.run_full_study import completed_stage
        self._write(str(tmp_path), {**finished, "size": 250})
        # File lives under kg250 but claims size 500 -> mismatch.
        import os
        path = os.path.join(str(tmp_path), "kg250", "koerkel_ghosh_results.json")
        with open(path) as fh:
            data = json.load(fh)
        data["size"] = 500
        with open(path, "w") as fh:
            json.dump(data, fh)
        assert completed_stage(250, cfg, str(tmp_path)) is None

    def test_a_corrupt_sidecar_is_not_reused_and_does_not_raise(self, cfg, tmp_path):
        """A reboot mid-write is exactly how a truncated sidecar appears."""
        import os
        from scripts.run_full_study import completed_stage
        d = os.path.join(str(tmp_path), "kg250")
        os.makedirs(d, exist_ok=True)
        with open(os.path.join(d, "koerkel_ghosh_results.json"), "w") as fh:
            fh.write('{"size": 250, "reco')
        assert completed_stage(250, cfg, str(tmp_path)) is None

    def test_a_missing_sidecar_is_not_reused(self, cfg, tmp_path):
        from scripts.run_full_study import completed_stage
        assert completed_stage(250, cfg, str(tmp_path)) is None

    def test_force_recomputes_even_a_matching_sidecar(self, finished, cfg, kg_payload,
                                                      tmp_path, monkeypatch):
        """COUNTERWEIGHT: the escape hatch must actually bypass reuse."""
        import scripts.run_full_study as mod
        self._write(str(tmp_path), finished)
        captured = []

        def fake(cmd, *a, **k):
            captured.append(cmd)
            self._write(str(tmp_path), kg_payload)
            return None

        monkeypatch.setattr(mod.subprocess, "run", fake)
        mod.run_stage(250, cfg, str(tmp_path), force=True)
        assert captured, "force must launch the benchmark"
