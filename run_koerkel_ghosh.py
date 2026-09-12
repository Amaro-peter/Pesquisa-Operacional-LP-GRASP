"""
Körkel-Ghosh benchmark: does the LP bias earn its name where the LP is weak?

Every other instance family in this repository has a near-integral LP
relaxation, which makes them a poor test of a method premised on the fractional
LP carrying information. This benchmark runs the same four arms on the standard
hard family instead:

    A.  LP rounding            -- open every y >= 0.5, no local search.
    A+. Rounding + search      -- the same, then the local search. Deterministic.
    B.  LP-biased GRASP        -- sample facilities with probability max(y, EPS),
                                  then the same local search.
    C.  alpha-GRASP            -- savings-based RCL construction, then the same
                                  local search.
    D.  Local search only      -- start from the single cheapest facility and run
                                  the same local search. Never reads the LP at
                                  all. Isolates what the relaxation contributes.

Arms A+ and B differ only in whether construction is deterministic or sampled,
so any difference between them is attributable to randomization alone.

Gaps are reported against the LP bound, which on this family is genuinely weak
-- that is the whole point of using it. A gap against the LP bound therefore
OVERSTATES the true optimality gap by an unknown amount, so the report also
gives each arm's excess over the best solution any arm found.

See `koerkel_ghosh.py` for the generation specification and for why these are
generated-to-spec rather than the official UflLib files.

Usage:
    python run_koerkel_ghosh.py                       # 250x250, all classes
    python run_koerkel_ghosh.py --size 500 --classes b c
    python run_koerkel_ghosh.py --instances 1 --seeds 42   # quick smoke run
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import time
from dataclasses import asdict
from datetime import datetime, timezone
from typing import Dict, List

from download_and_run_real_world import (
    ArmResult,
    _cmp,
    _environment,
    _git_commit,
    _mean,
    _plural,
    _signed,
    profile_lp,
    run_alpha_grasp,
    run_lp_biased,
    run_local_search_only,
    run_lp_rounding,
    run_lp_rounding_plus_search,
)
from koerkel_ghosh import (
    ALLOCATION_COST_RANGE,
    FIXED_COST_RANGES,
    generate_koerkel_ghosh_instance,
    instance_name,
)
from run_experiments import get_lp_bound_and_probs


def run_one_instance(size: int, klass: str, symmetric: bool, index: int,
                     seeds: List[int]) -> Dict:
    """Run all four arms on a single instance and return a record."""
    name = instance_name(size, klass, symmetric, index)
    instance = generate_koerkel_ghosh_instance(
        size=size, klass=klass, symmetric=symmetric, seed=index
    )

    t0 = time.perf_counter()
    bound, probs = get_lp_bound_and_probs(instance)
    lp = profile_lp(bound, probs, time.perf_counter() - t0)

    control = run_lp_rounding(instance, probs, lp)
    rounding_ls = run_lp_rounding_plus_search(instance, probs, lp)
    no_lp = run_local_search_only(instance, lp.bound)
    biased = [run_lp_biased(instance, probs, lp, seed) for seed in seeds]
    alpha = [run_alpha_grasp(instance, lp, seed) for seed in seeds]

    best_cost = min(
        [control.final_cost, rounding_ls.final_cost, no_lp.final_cost]
        + [r.final_cost for r in biased]
        + [r.final_cost for r in alpha]
    )

    return {
        "name": name,
        "size": size,
        "klass": klass,
        "symmetric": symmetric,
        "index": index,
        "lp": asdict(lp),
        "best_cost_found": best_cost,
        "arms": {
            "lp_rounding_control": asdict(control),
            "lp_rounding_plus_search": asdict(rounding_ls),
            "local_search_only": asdict(no_lp),
            "lp_biased": [asdict(r) for r in biased],
            "alpha_grasp": [asdict(r) for r in alpha],
        },
    }


def _excess(cost: float, best: float) -> float:
    """Percent above the best solution any arm found on this instance."""
    return (cost - best) / best * 100.0


def render_report(records: List[Dict], seeds: List[int], size: int,
                  env: Dict[str, str], generated_at: str, commit: str,
                  wall_seconds: float) -> str:
    """
    Render the markdown purely from measured values.

    As in `download_and_run_real_world.render_report`, every comparative phrase
    is computed from the measurements rather than written into the template, so
    the report cannot state a conclusion its own tables contradict.
    """
    lines: List[str] = []
    w = lines.append

    # ---- aggregates across instances -------------------------------------
    def arm_excesses(rec, key):
        best = rec["best_cost_found"]
        arm = rec["arms"][key]
        if isinstance(arm, list):
            return [_excess(r["final_cost"], best) for r in arm]
        return [_excess(arm["final_cost"], best)]

    all_ap = [e for r in records for e in arm_excesses(r, "lp_rounding_plus_search")]
    all_b = [e for r in records for e in arm_excesses(r, "lp_biased")]
    all_c = [e for r in records for e in arm_excesses(r, "alpha_grasp")]
    all_a = [e for r in records for e in arm_excesses(r, "lp_rounding_control")]
    all_d = [e for r in records for e in arm_excesses(r, "local_search_only")]

    mean_ap, mean_b, mean_c, mean_a = _mean(all_ap), _mean(all_b), _mean(all_c), _mean(all_a)
    mean_d = _mean(all_d)
    d_beats_ap = sum(
        1 for r in records
        if _mean(arm_excesses(r, "local_search_only")) < _mean(arm_excesses(r, "lp_rounding_plus_search")) - 1e-9
    )
    ap_beats_d = sum(
        1 for r in records
        if _mean(arm_excesses(r, "lp_rounding_plus_search")) < _mean(arm_excesses(r, "local_search_only")) - 1e-9
    )
    d_ties_ap = len(records) - d_beats_ap - ap_beats_d
    mean_frac = _mean([r["lp"]["n_fractional"] / r["lp"]["n_facilities"] * 100 for r in records])
    mean_duality = _mean([
        (r["best_cost_found"] - r["lp"]["bound"]) / r["lp"]["bound"] * 100 for r in records
    ])

    # Per-instance wins, counted on the arm mean for the randomized arms.
    b_beats_ap = sum(
        1 for r in records
        if _mean(arm_excesses(r, "lp_biased")) < _mean(arm_excesses(r, "lp_rounding_plus_search")) - 1e-9
    )
    ap_beats_b = sum(
        1 for r in records
        if _mean(arm_excesses(r, "lp_rounding_plus_search")) < _mean(arm_excesses(r, "lp_biased")) - 1e-9
    )
    ties = len(records) - b_beats_ap - ap_beats_b

    b_beats_c = sum(
        1 for r in records
        if _mean(arm_excesses(r, "lp_biased")) < _mean(arm_excesses(r, "alpha_grasp")) - 1e-9
    )
    c_beats_b = sum(
        1 for r in records
        if _mean(arm_excesses(r, "alpha_grasp")) < _mean(arm_excesses(r, "lp_biased")) - 1e-9
    )

    # ---- header ----------------------------------------------------------
    w("# Körkel-Ghosh Benchmark — Does the LP Bias Help Where the LP Is Weak?")
    w("")
    w("Every other instance family in this repository has a near-integral LP relaxation, which")
    w("makes them a weak test of a method whose premise is that the *fractional* LP carries")
    w("information. This benchmark uses the standard hard family instead.")
    w("")
    w("> **Provenance.** Every number below was measured by the run described in §6 and written")
    w("> directly by `render_report` in `run_koerkel_ghosh.py`. No value is hardcoded or")
    w("> back-computed, and every comparative word is derived from the measurements. Raw records")
    w("> are in `koerkel_ghosh_results.json`.")
    w("")
    w("> **These are generated-to-spec instances, not the official UflLib files.** The official")
    w("> archives are served from hosts that refused every request from this environment (HTTP 403")
    w("> from `resources.mpi-inf.mpg.de`; a redirect loop from the Frankfurt mirror). The")
    w("> generator follows the published specification exactly (see `koerkel_ghosh.py`), but the")
    w("> random draws differ, so **objective values here are not comparable with published KG")
    w("> results** and literature best-known bounds do not apply. The arm-versus-arm comparison,")
    w("> which is what this benchmark exists for, is unaffected.")
    w("")
    w("---")
    w("")

    # ---- §1 why this family ---------------------------------------------
    w("## 1. Why this family")
    w("")
    w("| Property | Other families in this repo | Körkel-Ghosh |")
    w("|---|---|---|")
    w("| Allocation costs | Euclidean distances | uniform random "
      f"[{ALLOCATION_COST_RANGE[0]:,}, {ALLOCATION_COST_RANGE[1]:,}] |")
    w("| LP fractionality | 0.35% (California), 0% (cap134) | "
      f"**{mean_frac:.1f}%** (mean here) |")
    w("| LP duality gap | 0% on cap134 (bound = optimum) | "
      f"**{mean_duality:.2f}%** (mean, vs best found) |")
    w("")
    w("Fixed-cost classes, per the published specification:")
    w("")
    w("| Class | Fixed cost range | Effect |")
    w("|---|---|---|")
    for k in sorted(FIXED_COST_RANGES):
        lo, hi = FIXED_COST_RANGES[k]
        w(f"| **{k}** | [{lo:,}, {hi:,}] | "
          f"{'many facilities open' if k == 'a' else 'few facilities open' if k == 'c' else 'intermediate'} |")
    w("")

    # ---- §2 per-instance results ----------------------------------------
    w("## 2. Results")
    w("")
    w(f"Instance size {size}×{size}. Randomized arms run on {len(seeds)} "
      f"{_plural(len(seeds), 'seed')} ({', '.join(str(s) for s in seeds)}); "
      "values are means. Excess is percent above the best solution any arm found on that instance.")
    w("")
    w("| Instance | LP fractional | LP gap | A · round | A+ · round+LS | B · LP-biased | "
      "C · α-GRASP | D · no LP |")
    w("|---|---|---|---|---|---|---|---|")
    for r in records:
        lp = r["lp"]
        frac_pct = lp["n_fractional"] / lp["n_facilities"] * 100
        duality = (r["best_cost_found"] - lp["bound"]) / lp["bound"] * 100
        w(f"| `{r['name']}` | {lp['n_fractional']}/{lp['n_facilities']} ({frac_pct:.0f}%) | "
          f"{duality:.2f}% | {_mean(arm_excesses(r, 'lp_rounding_control')):.3f}% | "
          f"{_mean(arm_excesses(r, 'lp_rounding_plus_search')):.3f}% | "
          f"{_mean(arm_excesses(r, 'lp_biased')):.3f}% | "
          f"{_mean(arm_excesses(r, 'alpha_grasp')):.3f}% | "
          f"{_mean(arm_excesses(r, 'local_search_only')):.3f}% |")
    w("")
    w("| Arm | Mean excess over best found |")
    w("|---|---|")
    w(f"| A · LP rounding only | {mean_a:.3f}% |")
    w(f"| A+ · Rounding + local search | {mean_ap:.3f}% |")
    w(f"| B · LP-biased GRASP | {mean_b:.3f}% |")
    w(f"| C · α-GRASP | {mean_c:.3f}% |")
    w(f"| D · Local search only (no LP) | {mean_d:.3f}% |")
    w("")

    # ---- §3 the ablation -------------------------------------------------
    w("## 3. The ablation: does randomization help here?")
    w("")
    w("On the California instance the randomized construction reached the identical solution")
    w("deterministic rounding did, so it contributed nothing. This family is the harder test.")
    w("")
    verdict = _cmp(mean_b, mean_ap, "B", "A+", tie="neither")
    w(f"Across {len(records)} {_plural(len(records), 'instance')}, arm B (randomized) beat arm A+")
    w(f"(deterministic) on **{b_beats_ap}**, lost on **{ap_beats_b}**, and tied on **{ties}**.")
    w(f"Mean excess: **{mean_b:.3f}%** for B against **{mean_ap:.3f}%** for A+.")
    w("")
    if verdict == "B":
        w(f"**Randomization earns its place on this family.** Arm B is better by "
          f"{mean_ap - mean_b:.3f} pp on average — the opposite of the California result. Where the")
        w("relaxation is strongly fractional, thresholding it at 0.5 throws information away that")
        w("sampling from it retains.")
    elif verdict == "A+":
        w(f"**Randomization does not earn its place even here.** Arm A+ is better by "
          f"{mean_b - mean_ap:.3f} pp on average, so the California finding survives the move to")
        w("a family built to be hard for LP-guided methods.")
    else:
        w("**The two are indistinguishable on this family**, so randomization is not paying for")
        w("the extra local-search work it creates.")
    w("")
    w(f"Against the classical baseline, arm B beat arm C on **{b_beats_c}** "
      f"{_plural(b_beats_c, 'instance')} and lost on **{c_beats_b}**.")
    w("")

    # ---- §4 rounding collapse -------------------------------------------
    w("## 4. Why rounding alone collapses here")
    w("")
    w("| Instance | Σy | facilities with y ≥ 0.5 | A · round-only excess |")
    w("|---|---|---|---|")
    for r in records:
        lp = r["lp"]
        w(f"| `{r['name']}` | {lp['sum_y']:.1f} | {lp['rounded_open_count']} | "
          f"{_mean(arm_excesses(r, 'lp_rounding_control')):.2f}% |")
    w("")
    zero_rounded = [r["name"] for r in records if r["lp"]["rounded_open_count"] == 0]
    if zero_rounded:
        w(f"On **{len(zero_rounded)}** of {len(records)} instances the relaxation puts *no* facility")
        w("at or above 0.5, so thresholding selects nothing at all and the construction falls back")
        w("to a single facility. The relaxation spreads its mass across many facilities rather than")
        w("concentrating it — which is exactly what a large duality gap looks like, and exactly the")
        w("condition under which a thresholding rule is the wrong way to read it.")
        w("")

    # ---- §5 findings -----------------------------------------------------
    w("## 5. Findings")
    w("")
    w(f"**1. The relaxation really is weak here.** Mean fractionality {mean_frac:.1f}% of facilities,")
    w(f"against 0.35% on the California instance; mean duality gap {mean_duality:.2f}% against the")
    w("best solution found, where cap134's LP bound equalled its integer optimum exactly.")
    w("")
    if verdict == "B":
        w(f"**2. The LP bias is vindicated on this family.** Arm B's mean excess is {mean_b:.3f}%")
        w(f"against arm A+'s {mean_ap:.3f}%. The California conclusion — that randomization adds")
        w("nothing — does not generalize to instances where the relaxation is genuinely fractional.")
    elif verdict == "A+":
        w(f"**2. The LP bias is not vindicated even here.** Arm A+ still matches or beats arm B")
        w(f"({mean_ap:.3f}% against {mean_b:.3f}%), so the finding holds on the family specifically")
        w("chosen to give randomization its best chance.")
    else:
        w("**2. The two constructions are indistinguishable on this family.**")
    w("")
    lp_verdict = _cmp(mean_ap, mean_d, "A+", "D", tie="neither")
    if lp_verdict == "neither" or abs(mean_ap - mean_d) < 0.01:
        w(f"**3. The LP contributes little or nothing on this family.** Arm D never reads the")
        w(f"relaxation at all and reaches {mean_d:.3f}% mean excess against {mean_ap:.3f}% for the")
        w(f"LP-guided arm A+ — a difference of {abs(mean_ap - mean_d):.3f} pp. Arm D beat A+ on")
        w(f"{d_beats_ap}, lost on {ap_beats_d} and tied on {d_ties_ap} of "
          f"{len(records)} {_plural(len(records), 'instance')}, while skipping the LP solve entirely.")
    elif lp_verdict == "A+":
        w(f"**3. The LP does contribute here.** Arm A+ reaches {mean_ap:.3f}% against")
        w(f"{mean_d:.3f}% for arm D, which never reads the relaxation — so the LP is worth its")
        w(f"solve time on this family even though thresholding it is the wrong way to use it.")
    else:
        w(f"**3. The LP is actively unhelpful here.** Arm D, which never reads the relaxation,")
        w(f"reaches {mean_d:.3f}% against {mean_ap:.3f}% for the LP-guided arm A+.")
    w("")
    best_arm = min(
        [("A · LP rounding", mean_a), ("A+ · rounding + local search", mean_ap),
         ("B · LP-biased GRASP", mean_b), ("C · α-GRASP", mean_c),
         ("D · local search only, no LP", mean_d)],
        key=lambda t: t[1],
    )
    w(f"**4. Best arm overall: {best_arm[0]}**, at {best_arm[1]:.3f}% mean excess.")
    w("")

    # ---- §6 reproduction -------------------------------------------------
    w("## 6. Reproduction")
    w("")
    w("```bash")
    w(f"python run_koerkel_ghosh.py --size {size} --seeds {' '.join(str(s) for s in seeds)}")
    w("```")
    w("")
    w("| Run metadata | |")
    w("|---|---|")
    w(f"| Generated at | {generated_at} |")
    w("| Generated by | `run_koerkel_ghosh.py` → `render_report` |")
    w(f"| Git commit | `{commit}` |")
    w(f"| Instances | {len(records)} |")
    w(f"| Heuristic seeds | {', '.join(str(s) for s in seeds)} |")
    w(f"| Total wall-clock | {wall_seconds:.1f} s |")
    w(f"| Python | {env['python']} |")
    w(f"| NumPy / SciPy | {env['numpy']} / {env['scipy']} |")
    w("")
    w("Raw measurements: [`koerkel_ghosh_results.json`](koerkel_ghosh_results.json).")
    w("")
    return "\n".join(lines) + "\n"


def main(argv: List[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument("--size", type=int, default=250)
    parser.add_argument("--classes", nargs="+", default=["a", "b", "c"])
    parser.add_argument("--instances", type=int, default=2,
                        help="instances per (class, symmetry) combination")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 7, 2024])
    parser.add_argument("--out-dir", type=str, default="output")
    args = parser.parse_args(argv)

    os.makedirs(args.out_dir, exist_ok=True)
    wall0 = time.perf_counter()
    records: List[Dict] = []

    for klass in args.classes:
        for symmetric in (True, False):
            for index in range(1, args.instances + 1):
                name = instance_name(args.size, klass, symmetric, index)
                print(f"Running {name} ...", flush=True)
                rec = run_one_instance(args.size, klass, symmetric, index, args.seeds)
                records.append(rec)
                lp = rec["lp"]
                print(f"  LP={lp['bound']:,.0f} ({lp['solve_seconds']:.1f}s) "
                      f"frac={lp['n_fractional']}/{lp['n_facilities']} "
                      f"y>=0.5:{lp['rounded_open_count']} | best={rec['best_cost_found']:,.0f}",
                      flush=True)

    wall = time.perf_counter() - wall0
    env = _environment()
    generated_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    commit = _git_commit()

    report = render_report(records, args.seeds, args.size, env, generated_at, commit, wall)

    md_path = os.path.join(args.out_dir, "koerkel_ghosh_results.md")
    json_path = os.path.join(args.out_dir, "koerkel_ghosh_results.json")
    with open(md_path, "w") as fh:
        fh.write(report)
    with open(json_path, "w") as fh:
        json.dump(
            {
                "generated_at": generated_at,
                "generated_by": "run_koerkel_ghosh.py::render_report",
                "git_commit": commit,
                "environment": env,
                "provenance": "generated to the published KG specification; NOT the official UflLib files",
                "size": args.size,
                "seeds": args.seeds,
                "records": records,
                "wall_seconds": wall,
            },
            fh,
            indent=2,
        )
    print(f"\nWrote {md_path}\nWrote {json_path}\nTotal wall-clock: {wall:.1f}s", flush=True)


if __name__ == "__main__":
    main()
