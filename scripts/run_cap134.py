"""
OR-Library cap134 benchmark — the classical reference instance.

Runs the same five arms as the other benchmarks on Beasley's cap134 (50
facilities, 50 customers), against the **proven integer optimum** rather than
the LP bound.

cap134 is included precisely because it is easy: its LP relaxation is integral,
so the bound equals the optimum and every method finds it. That makes it a
control on the whole comparison — an instance family where the LP-guided method
should have no advantage to demonstrate, because there is nothing left to
discover once the relaxation is solved. A method that looked good here and
nowhere else would be suspect.

Usage:
    python -m scripts.run_cap134
    python -m scripts.run_cap134 --restarts 32 --alpha 0.2
"""

from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import asdict
from datetime import datetime, timezone
from typing import Dict, List

from scripts.download_and_run_real_world import (
    _cmp,
    _environment,
    _git_commit,
    _mean,
    _plural,
    profile_lp,
    run_local_search_only,
    run_lp_rounding,
    run_lp_rounding_plus_search,
)
from scripts.multistart import multistart_alpha_grasp, multistart_lp_biased
from scripts.reference import compute_reference
from scripts.run_experiments import get_lp_bound_and_probs
from scripts.uflp_solver import parse_orlib_instance

OUTPUT_DIR = "output"
INSTANCE_CANDIDATES = ("data/cap134.txt", "cap134.txt")


def resolve_instance_path(candidates=INSTANCE_CANDIDATES) -> str | None:
    """First existing candidate path, or None."""
    return next((p for p in candidates if os.path.exists(p)), None)


def run_benchmark(path: str, seeds: List[int], alpha_value: float,
                  ip_time_limit: float) -> Dict:
    instance = parse_orlib_instance(path)

    t0 = time.perf_counter()
    bound, probs = get_lp_bound_and_probs(instance)
    lp = profile_lp(bound, probs, time.perf_counter() - t0)

    ref = compute_reference(instance, lp.bound, time_limit=ip_time_limit)

    control = run_lp_rounding(instance, probs, lp)
    rounding_ls = run_lp_rounding_plus_search(instance, probs, lp)
    no_lp = run_local_search_only(instance, lp.bound)
    biased = multistart_lp_biased(instance, probs, lp.bound, seeds, lp.solve_seconds)
    alpha = multistart_alpha_grasp(instance, lp.bound, seeds, alpha=alpha_value)

    # A claimed optimum that one of our own feasible solutions beats was not an
    # optimum. See tests/regression/test_unproven_optimum_regression.py.
    ref = ref.validated_against(min(
        control.final_cost, rounding_ls.final_cost, no_lp.final_cost,
        biased.best_cost, alpha.best_cost,
    ))

    return {
        "instance_path": path,
        "n_facilities": len(instance.facilities),
        "n_customers": len(instance.customers),
        "lp": asdict(lp),
        "reference": asdict(ref),
        "arms": {
            "lp_rounding_control": asdict(control),
            "lp_rounding_plus_search": asdict(rounding_ls),
            "local_search_only": asdict(no_lp),
            "lp_biased_multistart": asdict(biased),
            "alpha_grasp_multistart": asdict(alpha),
        },
    }


def render_report(rec: Dict, seeds: List[int], env: Dict[str, str],
                  generated_at: str, commit: str, wall_seconds: float,
                  alpha_value: float) -> str:
    """Render the markdown purely from measured values."""
    lines: List[str] = []
    w = lines.append
    R = len(seeds)
    lp, ref, arms = rec["lp"], rec["reference"], rec["arms"]

    def gap(cost):
        return (cost - ref["value"]) / ref["value"] * 100.0

    g_a = gap(arms["lp_rounding_control"]["final_cost"])
    g_ap = gap(arms["lp_rounding_plus_search"]["final_cost"])
    g_d = gap(arms["local_search_only"]["final_cost"])
    g_b = gap(arms["lp_biased_multistart"]["best_cost"])
    g_c = gap(arms["alpha_grasp_multistart"]["best_cost"])
    label = "optimality gap" if ref["proven"] else "gap vs LP bound"

    w("# OR-Library cap134 — Classical Reference Instance")
    w("")
    w(f"Beasley's cap134: {rec['n_facilities']} facilities, {rec['n_customers']} customers.")
    w("Included as a control rather than as a challenge — its LP relaxation is integral, so")
    w("there is nothing for an LP-guided heuristic to discover that solving the relaxation has")
    w("not already revealed.")
    w("")
    if ref["proven"]:
        w(f"> **Gaps are true optimality gaps.** CBC proved the integer optimum "
          f"({ref['value']:,.2f}) in {ref['ip_seconds']:.1f}s.")
    else:
        w(f"> **Integer optimum not proven** ({ref['ip_status']}); gaps are measured against the")
        w("> LP bound and therefore overstate the true optimality gap.")
    w("")
    w("> **Provenance.** Every number was measured by the run described below and written by")
    w("> `render_report` in `run_cap134.py`. No value is hardcoded; every comparative word is")
    w("> computed. Raw records: `cap134_results.json`.")
    w("")
    w("---")
    w("")

    w("## 1. Instance and relaxation")
    w("")
    w("| Quantity | Value |")
    w("|---|---|")
    w(f"| Facilities × customers | {rec['n_facilities']} × {rec['n_customers']} |")
    w(f"| LP bound | {lp['bound']:,.2f} |")
    w(f"| Proven integer optimum | {ref['value']:,.2f}"
      f"{'' if ref['proven'] else ' (not proven)'} |")
    duality = (ref["value"] - lp["bound"]) / lp["bound"] * 100
    w(f"| LP duality gap | **{duality:.4f}%** |")
    w(f"| Fractional facilities | {lp['n_fractional']} / {lp['n_facilities']} |")
    w(f"| LP solve time | {lp['solve_seconds']:.3f} s |")
    w("")
    if duality < 1e-6:
        w("**The LP bound equals the integer optimum.** Solving the relaxation solves the")
        w("instance. Any heuristic layered on top can at best match it, and pays for the LP")
        w("either way.")
    else:
        w(f"The relaxation leaves a {duality:.4f}% duality gap, so the heuristic layer has")
        w("something to contribute.")
    w("")

    w("## 2. Results")
    w("")
    w(f"| Arm | Type | Final cost | {label} | Time |")
    w("|---|---|---|---|---|")
    for name, key, kind in [
        ("A · LP rounding only", "lp_rounding_control", "deterministic"),
        ("A+ · LP rounding + local search", "lp_rounding_plus_search", "deterministic"),
        ("D · Local search only (no LP)", "local_search_only", "deterministic"),
    ]:
        arm = arms[key]
        t = arm["lp_seconds"] + arm["construct_seconds"] + arm["search_seconds"]
        w(f"| {name} | {kind} | {arm['final_cost']:,.2f} | {gap(arm['final_cost']):.4f}% | "
          f"{t * 1000:.1f} ms |")
    for name, key in [("B · LP-biased multistart", "lp_biased_multistart"),
                      ("C · α-GRASP multistart", "alpha_grasp_multistart")]:
        arm = arms[key]
        t = arm["lp_seconds"] + arm["construct_seconds"] + arm["search_seconds"]
        w(f"| **{name}** | best of {R} | **{arm['best_cost']:,.2f}** | "
          f"**{gap(arm['best_cost']):.4f}%** | {t * 1000:.1f} ms |")
    w("")

    solved = [n for n, g in [("A", g_a), ("A+", g_ap), ("D", g_d), ("B", g_b), ("C", g_c)]
              if g <= 1e-9]
    if len(solved) == 5:
        w("**Every arm reaches the optimum**, including the one that never reads the LP.")
    elif solved:
        w(f"Arms reaching the optimum: **{', '.join(solved)}**.")
    else:
        w("**No arm reached the reference value.**")
    w("")

    w("## 3. What this instance can and cannot show")
    w("")
    b_arm, c_arm = arms["lp_biased_multistart"], arms["alpha_grasp_multistart"]
    b_t = b_arm["lp_seconds"] + b_arm["construct_seconds"] + b_arm["search_seconds"]
    c_t = c_arm["lp_seconds"] + c_arm["construct_seconds"] + c_arm["search_seconds"]
    faster = _cmp(b_t, c_t, "faster", "slower")
    ratio = (c_t / b_t) if b_t > 0 else float("inf")
    w(f"- On **quality** the two randomized arms are "
      f"{_cmp(g_b, g_c, 'B ahead', 'C ahead', tie='indistinguishable')} "
      f"({g_b:.4f}% against {g_c:.4f}%).")
    w(f"- On **time** the LP-biased arm is **{ratio:.2f}× {faster}** "
      f"({b_t * 1000:.0f} ms against {c_t * 1000:.0f} ms), of which "
      f"{lp['solve_seconds'] * 1000:.0f} ms is the LP solve it alone pays for.")
    w(f"- B found its best at restart {b_arm['best_found_at']} of {R}; "
      f"C at restart {c_arm['best_found_at']} of {R}.")
    w("")
    if duality < 1e-6:
        w("Because the relaxation is integral, this instance cannot discriminate between the")
        w("methods on quality — only on cost. It is reported for exactly that reason: a")
        w("comparison that showed a large quality difference here would indicate a problem with")
        w("the comparison, not a property of the methods.")
    w("")

    w("## 4. Reproduction")
    w("")
    w("```bash")
    w(f"python -m scripts.run_cap134 --restarts {R} --alpha {alpha_value}")
    w("```")
    w("")
    w("| Run metadata | |")
    w("|---|---|")
    w(f"| Generated at | {generated_at} |")
    w("| Generated by | `run_cap134.py` → `render_report` |")
    w(f"| Git commit | `{commit}` |")
    w(f"| Instance | `{rec['instance_path']}` |")
    w(f"| Restarts | {R} |")
    w(f"| Total wall-clock | {wall_seconds:.2f} s |")
    w(f"| Python | {env['python']} |")
    w("")
    w("Raw measurements: [`cap134_results.json`](cap134_results.json).")
    w("")
    return "\n".join(lines) + "\n"


def main(argv: List[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument("--restarts", type=int, default=32)
    parser.add_argument("--alpha", type=float, default=0.2)
    parser.add_argument("--ip-time-limit", type=float, default=300.0)
    parser.add_argument("--out-dir", type=str, default=OUTPUT_DIR)
    args = parser.parse_args(argv)

    path = resolve_instance_path()
    if path is None:
        print(f"Error: cap134.txt not found (looked in: {', '.join(INSTANCE_CANDIDATES)}).")
        raise SystemExit(1)

    seeds = list(range(1, args.restarts + 1))
    os.makedirs(args.out_dir, exist_ok=True)

    print(f"Running cap134 from {path} with {args.restarts} restarts...", flush=True)
    wall0 = time.perf_counter()
    rec = run_benchmark(path, seeds, args.alpha, args.ip_time_limit)
    wall = time.perf_counter() - wall0

    ref, arms = rec["reference"], rec["arms"]
    print(f"  LP={rec['lp']['bound']:,.2f} | ref={ref['value']:,.2f} "
          f"({'proven' if ref['proven'] else ref['ip_status']}) | "
          f"A+={arms['lp_rounding_plus_search']['final_cost']:,.2f} "
          f"B={arms['lp_biased_multistart']['best_cost']:,.2f} "
          f"C={arms['alpha_grasp_multistart']['best_cost']:,.2f}", flush=True)

    env = _environment()
    generated_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    commit = _git_commit()
    report = render_report(rec, seeds, env, generated_at, commit, wall, args.alpha)

    md_path = os.path.join(args.out_dir, "cap134_results.md")
    json_path = os.path.join(args.out_dir, "cap134_results.json")
    with open(md_path, "w") as fh:
        fh.write(report)
    with open(json_path, "w") as fh:
        json.dump({
            "generated_at": generated_at,
            "generated_by": "run_cap134.py::render_report",
            "git_commit": commit,
            "environment": env,
            "restarts": args.restarts,
            "alpha": args.alpha,
            "seeds": seeds,
            "record": rec,
            "wall_seconds": wall,
        }, fh, indent=2)

    print(f"Wrote {md_path}\nWrote {json_path}\nTotal wall-clock: {wall:.2f}s", flush=True)


if __name__ == "__main__":
    main()
