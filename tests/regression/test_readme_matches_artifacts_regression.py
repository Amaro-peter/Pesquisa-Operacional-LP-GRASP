"""
Regression: the README is hand-written and cites generated measurements.

WHY THIS EXISTS
    Everything under `output/` is generated, and two regression suites already
    forbid a report from stating a conclusion its own table contradicts. The
    README is the one document exempt from that machinery -- it is written by
    hand -- and it quotes dozens of figures that live in the sidecars.

    That is exactly the shape of the original defect in this project: a
    published artifact (`fix_markdown.py`) whose numbers were typed rather than
    measured, and which drifted from the data it claimed to summarise.

    These tests do not check prose. They check that every headline FIGURE the
    README asserts still equals what the artifacts say, so a re-run that moves
    the numbers fails loudly instead of leaving a stale claim on the front page.

    If a rung's sidecar is absent (a fresh clone, or a partial run) the
    corresponding test skips rather than failing -- the guard is about drift,
    not about forcing everyone to own 30 hours of compute.
"""

import json
import os
import re

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
README = os.path.join(ROOT, "README.md")
SIZES = (100, 150, 200, 250, 500, 750)


def _sidecar(size):
    path = os.path.join(ROOT, "output", f"kg{size}", "koerkel_ghosh_results.json")
    if not os.path.exists(path):
        pytest.skip(f"kg{size} sidecar not present")
    with open(path) as fh:
        return json.load(fh)


def _gap(rec, key):
    arm = rec["arms"][key]
    cost = arm.get("best_cost", arm.get("final_cost"))
    ref = rec["reference"]["value"]
    return (cost - ref) / ref * 100.0


def _mean_gap(payload, key):
    recs = payload["records"]
    return sum(_gap(r, key) for r in recs) / len(recs)


@pytest.fixture(scope="module")
def readme():
    with open(README) as fh:
        return fh.read()


class TestHeadlineTable:
    @pytest.mark.parametrize("size", SIZES)
    def test_each_arms_mean_gap_appears_as_written(self, readme, size):
        """Every mean gap in the headline table must match its sidecar."""
        payload = _sidecar(size)
        for key in ("lp_rounding_plus_search", "local_search_only",
                    "lp_biased_multistart", "alpha_grasp_multistart"):
            value = f"{_mean_gap(payload, key):.4f}%"
            assert value in readme, (
                f"README does not contain {value} for {key} at {size}x{size}; "
                "the artifacts have moved and the README is stale"
            )

    @pytest.mark.parametrize("size", SIZES)
    def test_the_proven_optima_counts_are_current(self, readme, size):
        payload = _sidecar(size)
        recs = payload["records"]
        proven = sum(1 for r in recs if r["reference"]["proven"])
        assert f"{proven} / {len(recs)}" in readme, (
            f"README is missing the '{proven} / {len(recs)}' proven count for {size}"
        )


class TestHeadToHeadClaims:
    def _tally(self, key_a, key_b):
        wins = losses = total = 0
        for size in SIZES:
            for rec in _sidecar(size)["records"]:
                a, b = _gap(rec, key_a), _gap(rec, key_b)
                total += 1
                if a < b - 1e-9:
                    wins += 1
                elif b < a - 1e-9:
                    losses += 1
        return wins, losses, total

    def test_the_multistart_record_is_stated_correctly(self, readme):
        """
        The README's central claim: "B beats A+ on 31 of 48 instances and loses
        on 2." If a re-run changes that, the claim must fail here rather than
        stand as the headline.
        """
        wins, losses, total = self._tally("lp_biased_multistart",
                                          "lp_rounding_plus_search")
        assert f"beats A+ on {wins} of {total} instances and loses on {losses}" in readme

    def test_the_lp_bias_record_is_stated_correctly(self, readme):
        wins, losses, total = self._tally("lp_biased_multistart",
                                          "alpha_grasp_multistart")
        ties = total - wins - losses
        assert (f"B beats C on {wins}, loses on {losses}, and **ties on {ties}**"
                in readme)

    def test_the_size_split_is_stated_correctly(self, readme):
        """The claim that B's wins are confined to the smaller rungs."""
        def tally(sizes):
            w = l = 0
            for size in sizes:
                for rec in _sidecar(size)["records"]:
                    b = _gap(rec, "lp_biased_multistart")
                    c = _gap(rec, "alpha_grasp_multistart")
                    if b < c - 1e-9:
                        w += 1
                    elif c < b - 1e-9:
                        l += 1
            return w, l

        small_w, small_l = tally((100, 150, 200))
        large_w, large_l = tally((250, 500, 750))
        rows = re.findall(r"\|\s*\*?\*?(\d+)\*?\*?\s*\|\s*\*?\*?(\d+)\*?\*?\s*\|", readme)
        assert (str(small_w), str(small_l)) in rows, (
            f"README's small-size split should be {small_w}-{small_l}")
        assert (str(large_w), str(large_l)) in rows, (
            f"README's large-size split should be {large_w}-{large_l}")


class TestCaliforniaClaims:
    @pytest.fixture(scope="class")
    def california(self):
        path = os.path.join(ROOT, "output", "california_4M_results.json")
        if not os.path.exists(path):
            pytest.skip("California sidecar not present")
        with open(path) as fh:
            return json.load(fh)

    def test_the_quoted_gaps_match(self, readme, california):
        ref = california["reference"]["value"]
        arms = california["arms"]
        for key in ("lp_rounding_plus_search", "lp_biased_multistart",
                    "alpha_grasp_multistart"):
            cost = arms[key].get("best_cost", arms[key].get("final_cost"))
            assert f"{(cost - ref) / ref * 100:.4f}%" in readme

    def test_the_spread_claim_matches(self, readme, california):
        """
        "B's worst restart (0.0421%) is better than C's mean restart (0.1618%)"
        -- the sharpest claim in the README, and the one most worth guarding.
        """
        b = california["arms"]["lp_biased_multistart"]["per_restart_gaps"]
        c = california["arms"]["alpha_grasp_multistart"]["per_restart_gaps"]
        assert f"{max(b):.4f}%" in readme
        assert f"{sum(c) / len(c):.4f}%" in readme
        assert max(b) < sum(c) / len(c), "the claim itself no longer holds"

    def test_the_speed_ratio_matches(self, readme, california):
        arms = california["arms"]
        t = lambda k: (arms[k]["lp_seconds"] + arms[k]["construct_seconds"]
                       + arms[k]["search_seconds"])
        ratio = t("alpha_grasp_multistart") / t("lp_biased_multistart")
        assert f"{ratio:.2f}× faster" in readme


class TestControlInstance:
    def test_the_cap134_optimum_is_quoted_correctly(self, readme):
        path = os.path.join(ROOT, "output", "cap134_results.json")
        if not os.path.exists(path):
            pytest.skip("cap134 sidecar not present")
        with open(path) as fh:
            rec = json.load(fh)["record"]
        assert f"{rec['reference']['value']:,.2f}" in readme
        assert rec["reference"]["proven"], "README calls this a proven optimum"
