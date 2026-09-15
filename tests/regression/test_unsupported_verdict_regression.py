"""
Regression: a 0.0023 pp mean difference was reported as one arm "beating" another.

WHAT BROKE
    §3 of the Körkel-Ghosh report picked its verdict with
    `_cmp(mean_b, mean_c, "B", "C", tie="neither")`, whose tie tolerance is
    1e-9 -- float noise. Any difference above that, however meaningless,
    produced a bolded claim that one construction beats the other.

WHY IT MATTERED
    The 250x250 run printed:

        - B beat C on 2 instances, lost on 2, tied on 8.
        - Mean gap vs reference: 1.2030% (B) against 1.2053% (C).

        **With multistart, the LP-biased construction beats the classical
        baseline** by 0.0023 pp on average.

    The bullets describe a dead heat; the sentence underneath declares a
    winner. This is precisely the failure the project's "reports are generated,
    never written" rule exists to prevent -- a report stating a conclusion its
    own table contradicts. It would have become the headline claim of the
    README.

THE FIX
    The verdict is decided by the PAIRED PER-INSTANCE RECORD, not by the raw
    mean. The experiment is paired -- same instances, same restart budget, same
    local search -- so the win/loss record is the evidence and the mean is a
    summary of magnitude. When the record is split the arms are reported as
    indistinguishable whatever the means say, and when mean and record disagree
    the report says so instead of silently siding with one.
"""

import pytest

from scripts.run_koerkel_ghosh import render_report
from tests.unit.test_koerkel_ghosh_benchmark import ENV, _record


def _report(pairs, size=250):
    """
    Render the report for per-instance (B gap %, C gap %) pairs.

    Each arm's best-of-N gap is the minimum of its trajectory, so a one-element
    trajectory pins the arm's result exactly where the case needs it.
    """
    records = [_record(f"i{i}", b=[b], c=[c], proven=False)
               for i, (b, c) in enumerate(pairs)]
    return render_report(records, list(range(1, 33)), size, ENV,
                         "now", "abc", 1.0, alpha_value=0.2)


class TestASplitRecordIsNotAWin:
    def test_the_exact_case_from_the_250_run(self):
        """
        RED before the fix. Two wins each, the rest tied, B's mean a hair lower.
        """
        pairs = [(1.00, 2.00), (1.00, 2.00),          # B wins 2, by a lot
                 (1.90, 1.00), (1.90, 1.00),          # C wins 2, by less
                 *[(1.50, 1.50)] * 8]                 # 8 ties
        # -> mean B 1.4833% vs mean C 1.5000%: B's mean is lower, as in the
        #    real run, but the per-instance record is 2-2.
        md = _report(pairs)

        assert "indistinguishable" in md
        assert "beats the classical baseline" not in md
        assert "classical baseline beats" not in md

    def test_a_consistent_record_still_produces_a_verdict(self):
        """
        COUNTERWEIGHT: the fix must not mute every conclusion. A clean sweep is
        a real result and must still be reported as one.
        """
        md = _report([(1.0, 5.0)] * 6)
        assert "beats the classical baseline" in md
        assert "indistinguishable" not in md

    def test_the_reverse_sweep_reports_the_reverse_verdict(self):
        md = _report([(5.0, 1.0)] * 6)
        assert "classical baseline beats the LP-biased construction" in md

    def test_a_single_win_against_no_losses_is_reported_with_its_weakness_visible(self):
        """
        1-0 with 5 ties is directionally real but thin. The verdict may stand,
        but the record must be stated alongside it so a reader can judge.
        """
        md = _report([(1.0, 2.0), *[(1.5, 1.5)] * 5])
        assert "beats the classical baseline" in md
        assert "1 of 6" in md or "won 1" in md or "beat C on 1" in md


class TestMeanAndRecordDisagreeing:
    def test_a_mean_dragged_by_one_outlier_does_not_override_the_record(self):
        """
        B loses 3 instances narrowly and wins 1 enormously, so its mean is far
        better while it lost the head-to-head 1-3. Neither may be silently
        presented as the winner.
        """
        pairs = [(1.00, 0.50), (1.00, 0.50), (1.00, 0.50),   # C wins 3 narrowly
                 (0.00, 80.0)]                               # B wins 1 hugely
        md = _report(pairs)
        assert "disagree" in md.lower() or "indistinguishable" in md
        assert "**With multistart, the LP-biased construction beats" not in md


class TestSetupHeader:
    """
    The header read "3 classes × symmetric/asymmetric × 3" above a table of 12
    instances. 3 x 2 x 3 = 18, not 12: the multiplier was hardcoded and had
    stopped matching the run. It must be derived from the records.
    """

    def _shaped(self, classes, symmetries, per_combo):
        recs = []
        for klass in classes:
            for sym in symmetries:
                for i in range(1, per_combo + 1):
                    r = _record(f"{klass}{sym}{i}", b=[1.0], c=[2.0], proven=False)
                    r["klass"], r["symmetric"] = klass, sym
                    recs.append(r)
        return render_report(recs, list(range(1, 33)), 250, ENV, "now", "abc", 1.0,
                             alpha_value=0.2)

    def test_the_breakdown_matches_the_actual_run_shape(self):
        """The real 250 run: 3 classes x both symmetries x 2 each = 12."""
        md = self._shaped("abc", (True, False), 2)
        assert "12 at 250×250 (3 classes × symmetric/asymmetric × 2)" in md
        assert "× 3)" not in md

    def test_a_differently_shaped_run_reports_its_own_shape(self):
        """COUNTERWEIGHT: the multiplier must track the data, not a new constant."""
        md = self._shaped("abc", (True, False), 1)
        assert "6 at 250×250 (3 classes × symmetric/asymmetric × 1)" in md

    def test_a_single_class_run_is_described_as_such(self):
        md = self._shaped("b", (True,), 4)
        assert "4 at 250×250 (1 class × one symmetry × 4)" in md
