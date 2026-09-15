"""
Regression: the cross-size analysis asserted trends its own tables contradicted.

TWO DEFECTS, both in `render_analysis` in run_full_study.py.

1.  "The LP bias holds its advantage over the classical baseline AS INSTANCES
    GROW."
    Emitted whenever B's total win count exceeded C's. On the real six-rung
    ladder B led 9-6 overall -- but every one of those wins came from the small,
    provable rungs:

        100x100   B 1 - 0 C
        150x150   B 0 - 0 C
        200x200   B 2 - 0 C
        250x250   B 2 - 2 C     <- official sizes
        500x500   B 2 - 3 C
        750x750   B 2 - 1 C

    B leads 3-0 below the provability frontier and 6-6 above it. The claim
    "as instances grow" states precisely the opposite of what the data shows,
    and it was on course to become the README's headline.

2.  "The B-versus-C margin WIDENS with size: +0.0030 pp at 100x100,
    +0.0030 pp at 750x750."
    A trend word chosen by `_cmp` with a 1e-9 tie tolerance, applied to values
    that are identical at the precision the sentence itself prints. A report
    may not describe two equal numbers as a change.

THE FIXES
    1. The trend claim is split at the provability frontier and is only made
       when the record on the LARGER rungs supports it.
    2. The trend word is decided by comparing the values AS DISPLAYED. If the
       sentence prints them to four decimals, four decimals is what decides the
       verb -- so the wording can never contradict its own figures.
"""

import pytest

from scripts.run_full_study import render_analysis, summarise_kg

ENV = {"python": "3.14.0", "platform": "linux"}


def _arm(cost, multistart=False, best_at=1):
    base = {"lp_seconds": 0.5, "construct_seconds": 0.25, "search_seconds": 0.25}
    if multistart:
        base.update({"best_cost": cost, "best_found_at": best_at,
                     "per_restart_gaps": [0.0]})
    else:
        base["final_cost"] = cost
    return base


def _stage(size, pairs, proven):
    """One rung: per-instance (B gap %, C gap %), and how many optima were proven."""
    recs = []
    for i, (b, c) in enumerate(pairs):
        ref = 100.0
        recs.append({
            "name": f"i{size}-{i}", "size": size, "klass": "a", "symmetric": True,
            "index": i,
            "lp": {"bound": ref * 0.97, "n_facilities": size, "n_fractional": 10,
                   "solve_seconds": 0.1},
            "reference": {"value": ref, "lp_bound": ref * 0.97,
                          "proven": i < proven,
                          "ip_status": "Optimal" if i < proven else "not proven",
                          "ip_seconds": 10.0, "incumbent": None},
            "best_cost_found": ref,
            "arms": {
                "lp_rounding_control": _arm(ref * 1.10),
                "lp_rounding_plus_search": _arm(ref * 1.01),
                "local_search_only": _arm(ref * 1.02),
                "lp_biased_multistart": _arm(ref * (1 + b / 100), True),
                "alpha_grasp_multistart": _arm(ref * (1 + c / 100), True),
            },
        })
    return summarise_kg({"size": size, "restarts": 32, "records": recs,
                         "stage_seconds": 60.0})


def _ladder():
    """The shape of the real run: B wins only below the provability frontier."""
    return [
        _stage(100, [(0.0, 0.5), (0.0, 0.0)], proven=2),        # B 1-0, all proven
        _stage(200, [(0.0, 0.5), (0.0, 0.5)], proven=2),        # B 2-0, all proven
        _stage(500, [(0.5, 0.0), (0.5, 0.0), (0.0, 0.5)], proven=0),   # B 1-2
        _stage(750, [(0.5, 0.0), (0.0, 0.5)], proven=0),        # B 1-1
    ]


class TestTrendClaimMustSurviveTheLargeRungs:
    def test_wins_confined_to_provable_rungs_are_not_called_a_surviving_advantage(self):
        """RED before the fix: B leads 3-0 small, 2-3 large, 5-3 overall."""
        md = render_analysis(_ladder(), None, "now", "abc", 60.0, ENV)
        assert "holds its advantage over the classical baseline as instances grow" not in md
        assert "concentrated" in md.lower() or "does not persist" in md.lower()

    def test_an_advantage_that_does_hold_at_the_large_sizes_is_still_reported(self):
        """COUNTERWEIGHT: the fix must not mute a genuine, size-robust result."""
        stages = [
            _stage(100, [(0.0, 0.5), (0.0, 0.5)], proven=2),
            _stage(200, [(0.0, 0.5), (0.0, 0.5)], proven=2),
            _stage(500, [(0.0, 0.5), (0.0, 0.5)], proven=0),
            _stage(750, [(0.0, 0.5), (0.0, 0.5)], proven=0),
        ]
        md = render_analysis(stages, None, "now", "abc", 60.0, ENV)
        assert "as instances grow" in md
        assert "does not persist" not in md.lower()

    def test_the_classical_baseline_winning_overall_is_reported_plainly(self):
        stages = [
            _stage(100, [(0.5, 0.0), (0.5, 0.0)], proven=2),
            _stage(750, [(0.5, 0.0), (0.5, 0.0)], proven=0),
        ]
        md = render_analysis(stages, None, "now", "abc", 60.0, ENV)
        assert "the classical baseline overtakes" in md


class TestTrendWordMatchesThePrintedFigures:
    def test_two_equal_margins_are_not_described_as_widening(self):
        """
        RED before the fix: printed "widens ... +0.0030 pp ... +0.0030 pp".
        Margins equal at the printed precision must read as steady.
        """
        stages = [_stage(100, [(0.0, 0.0030)], proven=1),
                  _stage(750, [(0.0, 0.00300001)], proven=0)]
        md = render_analysis(stages, None, "now", "abc", 60.0, ENV)
        assert "holds steady" in md
        assert "**widens**" not in md
        assert "**narrows**" not in md

    def test_a_margin_that_really_grows_is_called_widening(self):
        """COUNTERWEIGHT: a visible change must still be named."""
        stages = [_stage(100, [(0.0, 0.10)], proven=1),
                  _stage(750, [(0.0, 5.00)], proven=0)]
        md = render_analysis(stages, None, "now", "abc", 60.0, ENV)
        assert "**widens**" in md

    def test_a_margin_that_really_shrinks_is_called_narrowing(self):
        stages = [_stage(100, [(0.0, 5.00)], proven=1),
                  _stage(750, [(0.0, 0.10)], proven=0)]
        md = render_analysis(stages, None, "now", "abc", 60.0, ENV)
        assert "**narrows**" in md
