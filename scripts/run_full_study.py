"""
Full study -- the complete experimental sequence, run end to end.

Runs, in order:

    1. Koerkel-Ghosh 250x250   (the library's smallest official size)
    2. Koerkel-Ghosh 500x500
    3. Koerkel-Ghosh 750x750   (the library's largest official size)
    4. California Housing 2000x2000

and then writes a cross-size analysis of how **hardness** and **provability**
trade off against each other as instances grow.

WHY THE SEQUENCE MATTERS
    A single size cannot answer the question this study asks. Small instances
    admit a proven integer optimum but stop being hard; large instances stay
    hard but put the optimum out of reach. Running the whole ladder makes that
    trade-off visible instead of forcing a choice between the two and reporting
    only the convenient half.

    Every stage records whether its reference value is a proven optimum or a
    lower bound, per instance, so the analysis can say exactly where ground
    truth ends and inference begins.

BUDGETS
    Restart counts and instance counts per size are inputs, not constants, and
    the report states the budget it ran under. A best-of-N result is only
    meaningful alongside N.

Usage:
    python -m scripts.run_full_study                      # the full ladder
    python -m scripts.run_full_study --skip-california    # Koerkel-Ghosh only
    python -m scripts.run_full_study --sizes 250 500      # a subset
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from typing import Dict, List

OUTPUT_DIR = "output"

# Per-size budgets, stated here rather than buried: every best-of-N number
# depends on them.
#
# The restart budget is deliberately CONSTANT across sizes. A best-of-N result is
# only comparable to another best-of-N, so varying N with size would confound
# "the method degrades on harder instances" with "the method was given less to
# work with".
#
# The exact-solve budget is also CONSTANT: thirty minutes per instance at every
# size, 750x750 included. Earlier revisions of this plan tapered it, and skipped
# the solve entirely at 500 and 750 on the argument that a proof was implausible
# there. That was a false economy. "CBC did not close this instance in thirty
# minutes" is a MEASUREMENT, and it is the measurement this study exists to
# report; "we did not try" is not a result at all. Whether a rung is provable is
# the question, so every rung gets the same honest attempt and the answer is
# whatever it is.
#
# What is known going in, from direct measurement:
#
#   * 100x100  CBC closes instances in 1.2-89.2 s. Proof is routine.
#   * 150x150  a 30 s budget is not enough.
#   * 250x250  a 1200 s budget was not enough on six instances -- every one
#              stopped at ~93% of budget holding an unproven incumbent.
#
# None of that licenses skipping 500 and 750. It predicts they will time out; a
# prediction is not a result, and the cost of being wrong about it is a silently
# missing row.
#
# Sizes 100-200 exist to locate where provability actually stops. They are not a
# substitute for the large rungs -- the ladder keeps 250/500/750 intact.
#
# "Proven" means CBC reported sol_status PROVEN_OPTIMAL *and* no arm beat the
# result -- see scripts/reference.py and
# tests/regression/test_unproven_optimum_regression.py for why the weaker check
# was wrong.
IP_TIME_LIMIT = 1800.0

def _rung(instances: int) -> Dict:
    """One rung. Restart and exact-solve budgets are the same at every size."""
    return {"instances": instances, "restarts": 32,
            "ip_time_limit": IP_TIME_LIMIT, "attempt_exact": True}


DEFAULT_PLAN: Dict[int, Dict] = {
    # Provability frontier.
    100: _rung(2),
    150: _rung(1),
    200: _rung(1),
    # The library's official sizes.
    250: _rung(2),
    500: _rung(1),
    750: _rung(1),
}

# The five arms, in the order every table prints them.
DETERMINISTIC_ARMS = [
    ("A · LP rounding only", "lp_rounding_control"),
    ("A+ · LP rounding + local search", "lp_rounding_plus_search"),
    ("D · Local search only (no LP)", "local_search_only"),
]
MULTISTART_ARMS = [
    ("B · MS-LP-GRASP (LP-biased)", "lp_biased_multistart"),
    ("C · α-GRASP multistart", "alpha_grasp_multistart"),
]


def stage_plan(size: int) -> Dict:
    return DEFAULT_PLAN.get(size, _rung(1))


def completed_stage(size: int, cfg: Dict, out_dir: str) -> Dict | None:
    """
    A finished sidecar for this exact rung, if one is already on disk.

    Rungs cost hours. Re-running one that has already completed -- because the
    ladder was interrupted at a *later* rung -- throws that away for nothing.
    The sidecar is reused only when it matches the configuration being asked
    for; a different restart budget or instance count is a different
    experiment and must be recomputed.
    """
    path = os.path.join(out_dir, f"kg{size}", "koerkel_ghosh_results.json")
    if not os.path.exists(path):
        return None
    try:
        with open(path) as fh:
            payload = json.load(fh)
    except (json.JSONDecodeError, OSError):
        return None
    expected = cfg["instances"] * 6          # classes a/b/c x symmetric/asymmetric
    if (payload.get("size") != size
            or payload.get("restarts") != cfg["restarts"]
            or len(payload.get("records", [])) != expected):
        return None
    payload.setdefault("stage_seconds", payload.get("wall_seconds", 0.0))
    return payload


def run_stage(size: int, cfg: Dict, out_dir: str, force: bool = False) -> Dict:
    """Run one Koerkel-Ghosh size as a subprocess and return its payload."""
    if not force:
        existing = completed_stage(size, cfg, out_dir)
        if existing is not None:
            print(f"\n=== Koerkel-Ghosh {size}x{size} -- already complete "
                  f"({len(existing['records'])} instances, {existing['restarts']} "
                  f"restarts); reusing sidecar", flush=True)
            return existing

    stage_dir = os.path.join(out_dir, f"kg{size}")
    cmd = [
        sys.executable, "-u", "-m", "scripts.run_koerkel_ghosh",
        "--size", str(size),
        "--classes", "a", "b", "c",
        "--instances", str(cfg["instances"]),
        "--restarts", str(cfg["restarts"]),
        "--ip-time-limit", str(cfg["ip_time_limit"]),
        "--out-dir", stage_dir,
    ]
    if not cfg.get("attempt_exact", True):
        cmd.append("--no-exact-ip")

    print(f"\n{'=' * 70}\n=== Koerkel-Ghosh {size}x{size} -- "
          f"{cfg['instances'] * 6} instances, {cfg['restarts']} restarts\n{'=' * 70}",
          flush=True)
    t0 = time.perf_counter()
    subprocess.run(cmd, check=True)
    elapsed = time.perf_counter() - t0
    print(f"=== {size}x{size} finished in {elapsed:.0f}s", flush=True)

    with open(os.path.join(stage_dir, "koerkel_ghosh_results.json")) as fh:
        payload = json.load(fh)
    payload["stage_seconds"] = elapsed
    return payload


def load_stage(size: int, out_dir: str) -> Dict:
    """
    Read a stage that has already been run, instead of running it again.

    The ladder costs hours, most of it exact-solve attempts that end in a
    timeout. Re-rendering the analysis -- adding a rung, fixing a sentence --
    must not require paying that again, so every stage is loaded from the JSON
    sidecar it wrote. This mirrors `--from-json` on the California benchmark.
    """
    path = os.path.join(out_dir, f"kg{size}", "koerkel_ghosh_results.json")
    with open(path) as fh:
        payload = json.load(fh)
    payload.setdefault("stage_seconds", payload.get("wall_seconds", 0.0))
    return payload


def load_california(out_dir: str) -> Dict:
    path = os.path.join(out_dir, "california_4M_results.json")
    with open(path) as fh:
        payload = json.load(fh)
    payload.setdefault("stage_seconds", payload.get("wall_seconds", 0.0))
    return payload


def run_california(restarts: int, out_dir: str) -> Dict:
    print(f"\n{'=' * 70}\n=== California Housing 2000x2000 -- {restarts} restarts\n{'=' * 70}",
          flush=True)
    cmd = [
        sys.executable, "-u", "-m", "scripts.download_and_run_real_world",
        "--size", "2000", "--restarts", str(restarts), "--out-dir", out_dir,
    ]
    t0 = time.perf_counter()
    subprocess.run(cmd, check=True)
    elapsed = time.perf_counter() - t0
    print(f"=== California finished in {elapsed:.0f}s", flush=True)

    with open(os.path.join(out_dir, "california_4M_results.json")) as fh:
        payload = json.load(fh)
    payload["stage_seconds"] = elapsed
    return payload


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def _mean(xs):
    xs = list(xs)
    return sum(xs) / len(xs) if xs else 0.0


def _median(xs):
    xs = sorted(xs)
    return xs[len(xs) // 2] if xs else 0.0


def _arm_cost(arm: Dict) -> float:
    """Final cost for a deterministic arm, best cost for a multistart one."""
    return arm["best_cost"] if "best_cost" in arm else arm["final_cost"]


def _arm_seconds(arm: Dict) -> float:
    return arm["lp_seconds"] + arm["construct_seconds"] + arm["search_seconds"]


def summarise_kg(payload: Dict) -> Dict:
    """Collapse one Koerkel-Ghosh stage into the numbers the report needs."""
    recs = payload["records"]
    n = len(recs)

    def gaps(key):
        return [(_arm_cost(r["arms"][key]) - r["reference"]["value"])
                / r["reference"]["value"] * 100.0 for r in recs]

    per_arm = {}
    for _label, key in DETERMINISTIC_ARMS + MULTISTART_ARMS:
        g = gaps(key)
        per_arm[key] = {
            "mean_gap": _mean(g),
            "best_gap": min(g),
            "worst_gap": max(g),
            "mean_seconds": _mean(_arm_seconds(r["arms"][key]) for r in recs),
            "at_reference": sum(1 for x in g if x <= 1e-9),
            "gaps": g,
        }

    proven = [r for r in recs if r["reference"]["proven"]]
    b, c = per_arm["lp_biased_multistart"]["gaps"], per_arm["alpha_grasp_multistart"]["gaps"]
    ap = per_arm["lp_rounding_plus_search"]["gaps"]
    d = per_arm["local_search_only"]["gaps"]

    return {
        "size": payload["size"],
        "instances": n,
        "restarts": payload["restarts"],
        "proven": len(proven),
        "mean_fractionality": _mean(
            r["lp"]["n_fractional"] / r["lp"]["n_facilities"] * 100 for r in recs),
        # A duality gap is only meaningful where the optimum is actually known.
        "mean_duality_gap": _mean(
            (r["reference"]["value"] - r["lp"]["bound"]) / r["lp"]["bound"] * 100
            for r in proven) if proven else None,
        "mean_ip_seconds": _mean(r["reference"]["ip_seconds"] for r in recs),
        "mean_lp_seconds": _mean(r["lp"]["solve_seconds"] for r in recs),
        "per_arm": per_arm,
        "a_plus": _mean(ap), "no_lp": _mean(d),
        "lp_biased": _mean(b), "alpha": _mean(c),
        "b_at_reference": per_arm["lp_biased_multistart"]["at_reference"],
        "c_at_reference": per_arm["alpha_grasp_multistart"]["at_reference"],
        "b_beats_c": sum(1 for x, y in zip(b, c) if x < y - 1e-9),
        "c_beats_b": sum(1 for x, y in zip(b, c) if y < x - 1e-9),
        "b_beats_ap": sum(1 for x, y in zip(b, ap) if x < y - 1e-9),
        "ap_beats_b": sum(1 for x, y in zip(b, ap) if y < x - 1e-9),
        "median_best_restart": _median(
            [r["arms"]["lp_biased_multistart"]["best_found_at"] for r in recs]),
        "median_restarts_to_best": _median(
            [r["arms"]["alpha_grasp_multistart"]["best_found_at"] for r in recs]),
        "rows": [{
            "name": r["name"],
            "klass": r["klass"],
            "symmetric": r["symmetric"],
            "lp_bound": r["lp"]["bound"],
            "reference": r["reference"]["value"],
            "proven": r["reference"]["proven"],
            "ip_status": r["reference"]["ip_status"],
            "ip_seconds": r["reference"]["ip_seconds"],
            "fractional": r["lp"]["n_fractional"],
            "n_facilities": r["lp"]["n_facilities"],
            "gaps": {k: (_arm_cost(r["arms"][k]) - r["reference"]["value"])
                     / r["reference"]["value"] * 100.0
                     for _l, k in DETERMINISTIC_ARMS + MULTISTART_ARMS},
            "best_found_at": r["arms"]["lp_biased_multistart"]["best_found_at"],
        } for r in recs],
        "stage_seconds": payload.get("stage_seconds", 0.0),
    }


def summarise_california(payload: Dict) -> Dict:
    """Collapse the California stage into the same shape as a KG stage."""
    ref, lp, arms = payload["reference"], payload["lp_profile"], payload["arms"]
    inst = payload["instance"]
    restarts = len(payload["seeds"])

    def gap(key):
        return (_arm_cost(arms[key]) - ref["value"]) / ref["value"] * 100.0

    per_arm = {}
    for _label, key in DETERMINISTIC_ARMS + MULTISTART_ARMS:
        g = gap(key)
        per_arm[key] = {
            "mean_gap": g, "best_gap": g, "worst_gap": g,
            "mean_seconds": _arm_seconds(arms[key]),
            "at_reference": 1 if g <= 1e-9 else 0,
            "gaps": [g],
        }
    b = arms["lp_biased_multistart"]
    c = arms["alpha_grasp_multistart"]
    return {
        "size": inst["n_facilities"],
        "n_customers": inst["n_customers"],
        "instances": 1,
        "restarts": restarts,
        "proven": 1 if ref["proven"] else 0,
        "ip_status": ref["ip_status"],
        "reference": ref["value"],
        "lp_bound": lp["bound"],
        "mean_fractionality": lp["n_fractional"] / lp["n_facilities"] * 100.0,
        "mean_duality_gap": None,
        "mean_ip_seconds": ref["ip_seconds"],
        "mean_lp_seconds": lp["solve_seconds"],
        "per_arm": per_arm,
        "a_plus": gap("lp_rounding_plus_search"),
        "no_lp": gap("local_search_only"),
        "lp_biased": gap("lp_biased_multistart"),
        "alpha": gap("alpha_grasp_multistart"),
        "b_best_found_at": b["best_found_at"],
        "c_best_found_at": c["best_found_at"],
        "b_per_restart_gaps": b["per_restart_gaps"],
        "c_per_restart_gaps": c["per_restart_gaps"],
        "stage_seconds": payload.get("stage_seconds", 0.0),
    }


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

def _cmp(a: float, b: float, lo: str, hi: str, tie: str = "level", tol: float = 1e-9) -> str:
    """Comparative word chosen by measurement, never written by hand."""
    if abs(a - b) <= tol:
        return tie
    return lo if a < b else hi


def render_kg_stage_table(s: Dict) -> List[str]:
    """One Koerkel-Ghosh size, in the same shape as the cap134 report."""
    lines: List[str] = []
    w = lines.append
    R, n = s["restarts"], s["instances"]
    label = "optimality gap" if s["proven"] == n else "gap vs reference"

    w(f"### Körkel-Ghosh {s['size']}×{s['size']}")
    w("")
    w(f"{n} instances (classes a/b/c, symmetric and asymmetric), {R} restarts per")
    w("randomized arm.")
    w("")
    if s["proven"] == n:
        w(f"> **Gaps here are true optimality gaps.** CBC proved all {n} integer optima "
          f"(mean {s['mean_ip_seconds']:.0f} s).")
    elif s["proven"] > 0:
        w(f"> **Mixed reference.** CBC proved {s['proven']} of {n} optima; the remaining "
          f"{n - s['proven']} are measured against the LP bound and therefore **overstate** "
          "the true optimality gap.")
    else:
        w(f"> **No optimum proven** at this size within the budget. All {n} gaps are measured "
          "against the LP bound and **overstate** the true optimality gap.")
    w("")

    # --- relaxation ------------------------------------------------------
    w("| Quantity | Value |")
    w("|---|---|")
    w(f"| Facilities × customers | {s['size']} × {s['size']} |")
    w(f"| Instances | {n} |")
    w(f"| Fractional facilities (mean) | **{s['mean_fractionality']:.1f}%** |")
    if s["mean_duality_gap"] is not None:
        w(f"| LP duality gap (mean, proven only) | **{s['mean_duality_gap']:.4f}%** |")
    else:
        w("| LP duality gap | not computable — no optimum proven |")
    w(f"| Optima proven | {s['proven']} / {n} |")
    w(f"| Mean LP solve time | {s['mean_lp_seconds']:.2f} s |")
    w(f"| Mean CBC exact-solve time | {s['mean_ip_seconds']:.1f} s |")
    w("")

    # --- arms ------------------------------------------------------------
    w(f"| Arm | Type | Mean {label} | Best | Worst | At reference | Mean time |")
    w("|---|---|---|---|---|---|---|")
    for name, key in DETERMINISTIC_ARMS:
        a = s["per_arm"][key]
        w(f"| {name} | deterministic | {a['mean_gap']:.4f}% | {a['best_gap']:.4f}% | "
          f"{a['worst_gap']:.4f}% | {a['at_reference']}/{n} | {a['mean_seconds']:.1f} s |")
    for name, key in MULTISTART_ARMS:
        a = s["per_arm"][key]
        w(f"| **{name}** | best of {R} | **{a['mean_gap']:.4f}%** | {a['best_gap']:.4f}% | "
          f"{a['worst_gap']:.4f}% | **{a['at_reference']}/{n}** | {a['mean_seconds']:.1f} s |")
    w("")

    ranked = sorted(
        [(name, s["per_arm"][key]["mean_gap"]) for name, key in
         DETERMINISTIC_ARMS + MULTISTART_ARMS],
        key=lambda t: t[1])
    w(f"Best mean gap at this size: **{ranked[0][0]}** ({ranked[0][1]:.4f}%); "
      f"worst: {ranked[-1][0]} ({ranked[-1][1]:.4f}%).")
    w("")

    # --- per instance ----------------------------------------------------
    w("<details><summary>Per-instance detail</summary>")
    w("")
    w("| Instance | Class | Reference | Proven | Fractional | A+ | D | **B** | C | B best at |")
    w("|---|---|---|---|---|---|---|---|---|---|")
    for r in s["rows"]:
        g = r["gaps"]
        w(f"| `{r['name']}` | {r['klass']}"
          f"{'/sym' if r['symmetric'] else '/asym'} | {r['reference']:,.0f} | "
          f"{'yes' if r['proven'] else 'no (' + r['ip_status'] + ')'} | "
          f"{r['fractional']}/{r['n_facilities']} | "
          f"{g['lp_rounding_plus_search']:.4f}% | {g['local_search_only']:.4f}% | "
          f"**{g['lp_biased_multistart']:.4f}%** | {g['alpha_grasp_multistart']:.4f}% | "
          f"{r['best_found_at']}/{R} |")
    w("")
    w("</details>")
    w("")
    return lines


def render_california_table(s: Dict) -> List[str]:
    lines: List[str] = []
    w = lines.append
    R = s["restarts"]
    label = "optimality gap" if s["proven"] else "gap vs LP bound"

    w(f"### California Housing {s['size']}×{s['n_customers']}")
    w("")
    w("Block-group centroids from the California Housing dataset — a real geographic")
    w(f"instance with {s['size'] * s['n_customers']:,} service-cost entries, {R} restarts.")
    w("")
    if s["proven"]:
        w("> **Gaps here are true optimality gaps.**")
    else:
        w(f"> **The integer optimum is not available** ({s['ip_status']}). All gaps are measured")
        w("> against the LP bound and therefore **overstate** the true optimality gap by an")
        w("> unknown amount. The ranking between arms is still valid — every arm is measured")
        w("> against the same reference.")
    w("")
    w("| Quantity | Value |")
    w("|---|---|")
    w(f"| Facilities × customers | {s['size']} × {s['n_customers']} |")
    w(f"| LP bound | {s['lp_bound']:,.2f} |")
    w(f"| Reference value | {s['reference']:,.2f} ({'proven' if s['proven'] else s['ip_status']}) |")
    w(f"| Fractional facilities | **{s['mean_fractionality']:.2f}%** |")
    w(f"| LP solve time | {s['mean_lp_seconds']:.1f} s |")
    w("")
    w(f"| Arm | Type | {label} | Time |")
    w("|---|---|---|---|")
    for name, key in DETERMINISTIC_ARMS:
        a = s["per_arm"][key]
        w(f"| {name} | deterministic | {a['mean_gap']:.4f}% | {a['mean_seconds']:.1f} s |")
    for name, key in MULTISTART_ARMS:
        a = s["per_arm"][key]
        w(f"| **{name}** | best of {R} | **{a['mean_gap']:.4f}%** | {a['mean_seconds']:.1f} s |")
    w("")
    w(f"B reached its best at restart **{s['b_best_found_at']} of {R}**; "
      f"C at restart **{s['c_best_found_at']} of {R}**.")
    w("")
    bw, cm = max(s["b_per_restart_gaps"]), _mean(s["c_per_restart_gaps"])
    w(f"B's **worst** restart is {bw:.4f}%; C's **mean** restart is {cm:.4f}% — "
      f"{_cmp(bw, cm, 'B ahead even at its worst', 'C ahead', tie='level')}.")
    w("")
    return lines


def render_analysis(stages: List[Dict], california: Dict | None,
                    generated_at: str, commit: str, wall: float,
                    env: Dict[str, str]) -> str:
    """Render the cross-size analysis purely from measured values."""
    lines: List[str] = []
    w = lines.append

    w("# Full Study — Hardness versus Provable Optima Across the Körkel-Ghosh Ladder")
    w("")
    w("The same five arms run across every official Körkel-Ghosh size and then on the")
    w("2000×2000 California instance. The question this document answers is not only")
    w("*which arm wins*, but **where ground truth stops being available**, and whether the")
    w("ranking survives past that point.")
    w("")
    w("| Arm | What it is | Reads the LP? | Randomized? |")
    w("|---|---|---|---|")
    w("| A | Round every `y ≥ 0.5`, no search | yes | no |")
    w("| A+ | A, then best-improvement local search | yes | no |")
    w("| D | Local search from the cheapest facility | **no** | no |")
    w("| **B · MS-LP-GRASP** | Multistart, construction sampled at `max(y, ε)` | yes | yes |")
    w("| C · α-GRASP | Multistart, savings-based RCL (α) | no | yes |")
    w("")
    w("> **Provenance.** Every number was measured by the run described in §6 and written by")
    w("> `render_analysis` in `run_full_study.py`. No value is hardcoded; every comparative")
    w("> word is computed. Raw records: `full_study.json`, plus each stage's own sidecar.")
    w("")
    w("---")
    w("")

    # ---- §1 the trade-off ------------------------------------------------
    w("## 1. Hardness versus provability")
    w("")
    w("| Size | Instances | Restarts | LP fractional | Duality gap | Optima proven | Mean CBC time |")
    w("|---|---|---|---|---|---|---|")
    for s in stages:
        dg = f"{s['mean_duality_gap']:.2f}%" if s["mean_duality_gap"] is not None else "—"
        w(f"| {s['size']}×{s['size']} | {s['instances']} | {s['restarts']} | "
          f"{s['mean_fractionality']:.1f}% | {dg} | "
          f"**{s['proven']} / {s['instances']}** | {s['mean_ip_seconds']:.1f} s |")
    if california:
        w(f"| {california['size']}×{california['n_customers']} (California) | 1 | "
          f"{california['restarts']} | {california['mean_fractionality']:.2f}% | — | "
          f"**{california['proven']} / 1** | {california['ip_status']} |")
    w("")

    fully = [s for s in stages if s["proven"] == s["instances"]]
    partly = [s for s in stages if 0 < s["proven"] < s["instances"]]
    none = [s for s in stages if s["proven"] == 0]

    if fully:
        big = max(s["size"] for s in fully)
        w(f"Ground truth is available **in full up to {big}×{big}**: every gap reported at that "
          "size and below is a true optimality gap.")
    for s in partly:
        w(f"At **{s['size']}×{s['size']}** CBC proved {s['proven']} of {s['instances']} optima "
          "within its budget; the rest are measured against the LP bound and therefore "
          "overstate the true gap.")
    for s in none:
        w(f"At **{s['size']}×{s['size']}** CBC proved **none** within its budget — at this size "
          "the family is beyond exact solution on this hardware.")
    if california:
        if california["proven"]:
            w("The California instance was closed exactly.")
        else:
            w("On the 2000×2000 California instance the integer program is out of reach "
              f"({california['ip_status']}): four million binary-linked assignment variables.")
    w("")
    w("**This is the trade-off the ladder exists to expose.** Small instances give certainty "
      "and little difficulty; large ones give difficulty and no certainty. A result reported "
      "at one size only is reporting one half of it — which is precisely the criticism this "
      "run was built to answer.")
    w("")

    # ---- §2 per-size tables ---------------------------------------------
    w("## 2. Results, size by size")
    w("")
    for s in stages:
        lines.extend(render_kg_stage_table(s))
    if california:
        lines.extend(render_california_table(california))

    # ---- §3 does the ranking survive -------------------------------------
    w("## 3. Does the ranking survive as instances get harder?")
    w("")
    w("| Size | A+ (det.) | D (no LP) | **B · MS-LP-GRASP** | C · α-GRASP | B at reference | C at reference |")
    w("|---|---|---|---|---|---|---|")
    for s in stages:
        w(f"| {s['size']}×{s['size']} | {s['a_plus']:.4f}% | {s['no_lp']:.4f}% | "
          f"**{s['lp_biased']:.4f}%** | {s['alpha']:.4f}% | "
          f"{s['b_at_reference']}/{s['instances']} | {s['c_at_reference']}/{s['instances']} |")
    if california:
        c = california
        w(f"| {c['size']}×{c['n_customers']} | {c['a_plus']:.4f}% | {c['no_lp']:.4f}% | "
          f"**{c['lp_biased']:.4f}%** | {c['alpha']:.4f}% | "
          f"{c['per_arm']['lp_biased_multistart']['at_reference']}/1 | "
          f"{c['per_arm']['alpha_grasp_multistart']['at_reference']}/1 |")
    w("")
    w("Head-to-head on the Körkel-Ghosh ladder, per size:")
    w("")
    w("| Size | B beats C | C beats B | B beats A+ | A+ beats B | Median winning restart (B) |")
    w("|---|---|---|---|---|---|")
    for s in stages:
        w(f"| {s['size']}×{s['size']} | {s['b_beats_c']} | {s['c_beats_b']} | "
          f"{s['b_beats_ap']} | {s['ap_beats_b']} | {s['median_best_restart']} / {s['restarts']} |")
    w("")

    b_wins = sum(s["b_beats_c"] for s in stages)
    c_wins = sum(s["c_beats_b"] for s in stages)
    total = sum(s["instances"] for s in stages)
    ties = total - b_wins - c_wins

    # An overall tally can be carried entirely by the easiest rungs. A claim about
    # what happens "as instances grow" must therefore be checked against the
    # instances that actually grew, so the record is split BY SIZE at the median
    # rung and the claim is tested on the larger half. Reporting only the total
    # once produced "the LP bias holds its advantage as instances grow" above a
    # table showing B leading 3-0 on the small rungs and 6-6 on the large ones.
    # See tests/regression/test_cross_size_trend_regression.py.
    by_size = sorted(stages, key=lambda st: st["size"])
    cut = len(by_size) // 2
    smaller, larger = by_size[:cut], by_size[cut:]
    b_small, c_small = (sum(st["b_beats_c"] for st in smaller),
                        sum(st["c_beats_b"] for st in smaller))
    b_large, c_large = (sum(st["b_beats_c"] for st in larger),
                        sum(st["c_beats_b"] for st in larger))
    hard = larger

    if b_wins > c_wins:
        w(f"Across all {total} Körkel-Ghosh instances, **B beats C on {b_wins}, loses on "
          f"{c_wins}, ties on {ties}**.")
        if hard and b_large > c_large:
            w(f"The advantage survives the move to the larger rungs — B leads "
              f"{b_large}–{c_large} on the biggest {len(larger)} size"
              f"{'' if len(larger) == 1 else 's'} "
              f"({', '.join(str(st['size']) for st in larger)}) — so the LP bias holds its "
              "advantage over the classical baseline as instances grow.")
        elif hard:
            w(f"But those wins are **concentrated on the smaller instances**: B leads "
              f"{b_small}–{c_small} on the smallest {len(smaller)} size"
              f"{'' if len(smaller) == 1 else 's'} "
              f"({', '.join(str(st['size']) for st in smaller)}) and only "
              f"{b_large}–{c_large} on the largest {len(larger)} "
              f"({', '.join(str(st['size']) for st in larger)}). On this evidence the "
              "advantage **does not persist** at the sizes the family exists to test.")
    elif c_wins > b_wins:
        w(f"Across all {total} Körkel-Ghosh instances, **C beats B on {c_wins}, loses on "
          f"{b_wins}, ties on {ties}** — the classical baseline overtakes the LP-biased "
          "construction at these sizes.")
    else:
        w(f"Across all {total} Körkel-Ghosh instances the two arms are level "
          f"({b_wins} wins each, {ties} ties).")
    w("")

    ordered = sorted(stages, key=lambda s: s["size"])
    if len(ordered) >= 2:
        first, last = ordered[0], ordered[-1]
        m0 = first["alpha"] - first["lp_biased"]
        m1 = last["alpha"] - last["lp_biased"]
        # Decide the verb from the values AS PRINTED. Comparing at full float
        # precision once produced "the margin widens ... +0.0030 pp ... +0.0030 pp":
        # a change described between two numbers the sentence shows to be equal.
        trend = _cmp(round(m0, 4), round(m1, 4), "widens", "narrows", tie="holds steady")
        w(f"The B-versus-C margin **{trend}** with size: {m0:+.4f} pp at "
          f"{first['size']}×{first['size']}, {m1:+.4f} pp at {last['size']}×{last['size']}.")
        w("")

    # ---- §4 what the LP contributes as size grows -------------------------
    w("## 4. What the LP contributes, by size")
    w("")
    w("Arm D never reads the relaxation; A+ is the *same* local search seeded from it. The")
    w("difference between the two columns is the LP's entire contribution, isolated from")
    w("randomization.")
    w("")
    w("| Size | A+ (LP-seeded) | D (no LP) | LP advantage |")
    w("|---|---|---|---|")
    for s in stages:
        w(f"| {s['size']}×{s['size']} | {s['a_plus']:.4f}% | {s['no_lp']:.4f}% | "
          f"{s['no_lp'] - s['a_plus']:+.4f} pp |")
    if california:
        c = california
        w(f"| {c['size']}×{c['n_customers']} | {c['a_plus']:.4f}% | {c['no_lp']:.4f}% | "
          f"{c['no_lp'] - c['a_plus']:+.4f} pp |")
    w("")
    adv = [s["no_lp"] - s["a_plus"] for s in stages]
    if california:
        adv.append(california["no_lp"] - california["a_plus"])
    if all(a > 1e-9 for a in adv):
        w("The LP-seeded arm is ahead **at every size tested**.")
    elif all(a < -1e-9 for a in adv):
        w("The LP-free arm is ahead at every size tested — on this evidence the relaxation is "
          "**not** paying for itself as a seed.")
    else:
        w("The LP advantage is **not consistent across sizes**; see the per-size figures above. "
          "Reporting a single headline number for it would misrepresent the evidence.")
    w("")

    # ---- §5 cost ----------------------------------------------------------
    w("## 5. What the ladder cost")
    w("")
    w("| Stage | Instances | Restarts | Wall-clock |")
    w("|---|---|---|---|")
    for s in stages:
        w(f"| Körkel-Ghosh {s['size']}×{s['size']} | {s['instances']} | {s['restarts']} | "
          f"{s['stage_seconds'] / 60:.1f} min |")
    if california:
        w(f"| California {california['size']}×{california['n_customers']} | 1 | "
          f"{california['restarts']} | {california['stage_seconds'] / 60:.1f} min |")
    w(f"| **Total** | | | **{wall / 60:.1f} min** |")
    w("")
    budgets = sorted({st["restarts"] for st in stages})
    if len(budgets) == 1 and stages:
        w(f"**The restart budget is the same at every size** (best of {budgets[0]}), so the "
          "quality columns above are directly comparable across the ladder: a difference "
          "between two rows is a difference in difficulty, not in how much search each row "
          "was given.")
    elif budgets:
        w(f"The restart budget is **not** constant across sizes ({budgets[0]}–{budgets[-1]}), "
          "so the larger sizes were searched less thoroughly than the smaller ones and the "
          "quality columns are not directly comparable between rows.")
    w("")
    ip_budgets = sorted({stage_plan(st["size"])["ip_time_limit"] for st in stages})
    if len(ip_budgets) > 1:
        w(f"The **exact-solve** budget does vary ({ip_budgets[0]:.0f}–{ip_budgets[-1]:.0f} s, "
          "§6), tightened only where a proof was already implausible. That changes how many "
          "optima get proven, not how well any arm searches.")
        w("")

    # ---- §6 reproduction --------------------------------------------------
    w("## 6. Reproduction")
    w("")
    w("```bash")
    w("python -m scripts.run_full_study")
    w("```")
    w("")
    w("| Stage | Instances | Restarts | CBC budget |")
    w("|---|---|---|---|")
    for s in stages:
        w(f"| KG {s['size']}×{s['size']} | {s['instances']} | {s['restarts']} | "
          f"{stage_plan(s['size'])['ip_time_limit']:.0f} s |")
    if california:
        w(f"| California {california['size']}×{california['n_customers']} | 1 | "
          f"{california['restarts']} | not attempted |")
    w("")
    w("| Run metadata | |")
    w("|---|---|")
    w(f"| Generated at | {generated_at} |")
    w("| Generated by | `run_full_study.py` → `render_analysis` |")
    w(f"| Git commit | `{commit}` |")
    w(f"| Total wall-clock | {wall / 60:.1f} min |")
    w(f"| Python | {env['python']} |")
    w("")
    w("Per-stage reports and raw data:")
    for s in stages:
        w(f"- `output/kg{s['size']}/koerkel_ghosh_results.md` · "
          f"`output/kg{s['size']}/koerkel_ghosh_results.json`")
    if california:
        w("- `output/california_4M_results.md` · `output/california_4M_results.json`")
    w("- `output/full_study.md` · `output/full_study.json`")
    w("")
    return "\n".join(lines) + "\n"


def main(argv: List[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument("--sizes", type=int, nargs="+", default=[250, 500, 750])
    parser.add_argument("--california-restarts", type=int, default=10)
    parser.add_argument("--skip-california", action="store_true")
    parser.add_argument("--force", action="store_true",
                        help="recompute every rung even if a matching sidecar exists")
    parser.add_argument("--from-stages", action="store_true",
                        help="re-render the analysis from stage sidecars already "
                             "in --out-dir, without re-running any benchmark")
    parser.add_argument("--out-dir", type=str, default=OUTPUT_DIR)
    args = parser.parse_args(argv)

    from scripts.download_and_run_real_world import _environment, _git_commit

    os.makedirs(args.out_dir, exist_ok=True)
    wall0 = time.perf_counter()

    stages: List[Dict] = []
    for size in args.sizes:
        payload = (load_stage(size, args.out_dir) if args.from_stages
                   else run_stage(size, stage_plan(size), args.out_dir,
                                  force=args.force))
        stages.append(summarise_kg(payload))

    california = None
    if not args.skip_california:
        california = summarise_california(
            load_california(args.out_dir) if args.from_stages
            else run_california(args.california_restarts, args.out_dir))

    wall = time.perf_counter() - wall0
    env = _environment()
    generated_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    commit = _git_commit()

    analysis = render_analysis(stages, california, generated_at, commit, wall, env)
    md_path = os.path.join(args.out_dir, "full_study.md")
    json_path = os.path.join(args.out_dir, "full_study.json")
    with open(md_path, "w") as fh:
        fh.write(analysis)
    with open(json_path, "w") as fh:
        json.dump({
            "generated_at": generated_at,
            "generated_by": "run_full_study.py::render_analysis",
            "git_commit": commit,
            "environment": env,
            "plan": {str(k): stage_plan(k) for k in args.sizes},
            "stages": stages,
            "california": california,
            "wall_seconds": wall,
        }, fh, indent=2)

    print(f"\nWrote {md_path}\nWrote {json_path}\nTotal wall-clock: {wall / 60:.1f} min",
          flush=True)


if __name__ == "__main__":
    main()
