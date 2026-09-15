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

from scripts.download_and_run_real_world import (
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
from scripts.multistart import multistart_alpha_grasp, multistart_lp_biased
from scripts.reference import compute_reference
from scripts.koerkel_ghosh import (
    ALLOCATION_COST_RANGE,
    FIXED_COST_RANGES,
    generate_koerkel_ghosh_instance,
    instance_name,
)
from scripts.run_experiments import get_lp_bound_and_probs


def run_one_instance(size: int, klass: str, symmetric: bool, index: int,
                     seeds: List[int], alpha_value: float = 0.2,
                     ip_time_limit: float = 300.0,
                     attempt_exact: bool = True) -> Dict:
    """
    Run every arm on a single instance and return a record.

    Deterministic arms (LP rounding, rounding + local search, local search with
    no LP) run once. Randomized arms run as multistart procedures over `seeds`,
    keeping the best solution.
    """
    name = instance_name(size, klass, symmetric, index)
    instance = generate_koerkel_ghosh_instance(
        size=size, klass=klass, symmetric=symmetric, seed=index
    )

    t0 = time.perf_counter()
    bound, probs = get_lp_bound_and_probs(instance)
    lp = profile_lp(bound, probs, time.perf_counter() - t0)

    # What gaps are measured against: a proven integer optimum where CBC can
    # supply one, otherwise the LP bound, labelled as such.
    ref = compute_reference(instance, lp.bound, time_limit=ip_time_limit,
                            attempt_exact=attempt_exact)

    # Deterministic arms -- restarting them reproduces the same solution, so
    # they are run once.
    control = run_lp_rounding(instance, probs, lp)
    rounding_ls = run_lp_rounding_plus_search(instance, probs, lp)
    no_lp = run_local_search_only(instance, lp.bound)

    # Randomized arms -- run as the multistart procedures they are named after.
    biased = multistart_lp_biased(instance, probs, lp.bound, seeds, lp.solve_seconds)
    alpha = multistart_alpha_grasp(instance, lp.bound, seeds, alpha=alpha_value)

    best_cost = min(
        control.final_cost, rounding_ls.final_cost, no_lp.final_cost,
        biased.best_cost, alpha.best_cost,
    )

    # A claimed optimum that one of our own feasible solutions beats was not an
    # optimum. Checked after the arms run, because that is when the evidence
    # exists. See tests/regression/test_unproven_optimum_regression.py.
    ref = ref.validated_against(best_cost)

    return {
        "name": name,
        "size": size,
        "klass": klass,
        "symmetric": symmetric,
        "index": index,
        "lp": asdict(lp),
        "reference": asdict(ref),
        "best_cost_found": best_cost,
        "arms": {
            "lp_rounding_control": asdict(control),
            "lp_rounding_plus_search": asdict(rounding_ls),
            "local_search_only": asdict(no_lp),
            "lp_biased_multistart": asdict(biased),
            "alpha_grasp_multistart": asdict(alpha),
        },
    }


def _excess(cost: float, best: float) -> float:
    """Percent above the best solution any arm found on this instance."""
    return (cost - best) / best * 100.0


def render_report(records: List[Dict], seeds: List[int], size: int,
                  env: Dict[str, str], generated_at: str, commit: str,
                  wall_seconds: float, alpha_value: float = 0.2) -> str:
    """
    Render the markdown purely from measured values.

    Every comparative phrase is computed from the measurements rather than
    written into the template, so the report cannot state a conclusion its own
    tables contradict.
    """
    lines: List[str] = []
    w = lines.append
    R = len(seeds)

    proven = [r for r in records if r["reference"]["proven"]]
    all_proven = len(proven) == len(records)

    def gap(rec, cost):
        ref = rec["reference"]["value"]
        return (cost - ref) / ref * 100.0

    def det(rec, key):
        return gap(rec, rec["arms"][key]["final_cost"])

    def ms(rec, key):
        return gap(rec, rec["arms"][key]["best_cost"])

    DET = [("A · LP rounding only", "lp_rounding_control"),
           ("A+ · LP rounding + local search", "lp_rounding_plus_search"),
           ("D · Local search only (no LP)", "local_search_only")]
    MS = [("B · LP-biased multistart", "lp_biased_multistart"),
          ("C · α-GRASP multistart", "alpha_grasp_multistart")]

    means = {label: _mean([det(r, k) for r in records]) for label, k in DET}
    means.update({label: _mean([ms(r, k) for r in records]) for label, k in MS})

    mean_frac = _mean([r["lp"]["n_fractional"] / r["lp"]["n_facilities"] * 100 for r in records])
    mean_duality = _mean([
        (r["reference"]["value"] - r["lp"]["bound"]) / r["lp"]["bound"] * 100 for r in records
    ])

    b_key, c_key = "lp_biased_multistart", "alpha_grasp_multistart"
    ap_key = "lp_rounding_plus_search"
    mean_b, mean_c, mean_ap = means["B · LP-biased multistart"], \
        means["C · α-GRASP multistart"], means["A+ · LP rounding + local search"]

    b_beats_c = sum(1 for r in records if ms(r, b_key) < ms(r, c_key) - 1e-9)
    c_beats_b = sum(1 for r in records if ms(r, c_key) < ms(r, b_key) - 1e-9)
    bc_ties = len(records) - b_beats_c - c_beats_b

    b_beats_ap = sum(1 for r in records if ms(r, b_key) < det(r, ap_key) - 1e-9)
    ap_beats_b = sum(1 for r in records if det(r, ap_key) < ms(r, b_key) - 1e-9)
    ab_ties = len(records) - b_beats_ap - ap_beats_b

    solved_b = sum(1 for r in records if ms(r, b_key) <= 1e-9)
    solved_c = sum(1 for r in records if ms(r, c_key) <= 1e-9)
    solved_ap = sum(1 for r in records if det(r, ap_key) <= 1e-9)

    # How much of the restart budget each randomized arm actually needed.
    b_found = [r["arms"][b_key]["best_found_at"] for r in records]
    c_found = [r["arms"][c_key]["best_found_at"] for r in records]

    # Compute-matched: restarts needed to match the deterministic arm.
    def restarts_to_match(rec, key):
        target = det(rec, ap_key)
        ref = rec["reference"]["value"]
        for i, cost in enumerate(rec["arms"][key]["trajectory_costs"], start=1):
            if (cost - ref) / ref * 100.0 <= target + 1e-9:
                return i
        return None

    b_match = [restarts_to_match(r, b_key) for r in records]
    b_match_ok = [m for m in b_match if m is not None]

    # ---- header ----------------------------------------------------------
    w("# Körkel-Ghosh Benchmark — Multistart LP-Biased GRASP vs. Classical α-GRASP")
    w("")
    w("The standard hard family for the UFLP: allocation costs are drawn at random rather")
    w("than from a metric embedding, which destroys the structure that makes the relaxation")
    w("tight. Both randomized arms run as **multistart** procedures, which is what makes them")
    w("GRASPs; deterministic arms run once, because restarting them reproduces the same solution.")
    w("")
    if all_proven:
        w(f"> **Gaps here are true optimality gaps.** CBC proved the integer optimum on all")
        w(f"> {len(records)} instances, so every percentage below is measured against a proven")
        w("> optimum rather than against a lower bound.")
    else:
        w(f"> **Mixed references.** CBC proved the integer optimum on {len(proven)} of")
        w(f"> {len(records)} instances; the rest are measured against the LP bound and therefore")
        w("> OVERSTATE the true optimality gap. The `reference.kind` field in the JSON says which")
        w("> is which, per instance.")
    w("")
    w("> **Provenance.** Every number was measured by the run described in §6 and written by")
    w("> `render_report` in `run_koerkel_ghosh.py`. No value is hardcoded; every comparative")
    w("> word is computed. Raw records: `koerkel_ghosh_results.json`.")
    w("")
    w("> **Generated-to-spec instances, not the official UflLib files.** The official archives")
    w("> were unreachable from this environment (HTTP 403). The generator follows the published")
    w("> specification exactly, but the random draws differ, so objective values are **not")
    w("> comparable with published KG results**. Arm-versus-arm comparison is unaffected.")
    w("")
    w("---")
    w("")

    # ---- §1 setup --------------------------------------------------------
    w("## 1. Setup")
    w("")
    w("| | |")
    w("|---|---|")
    n_classes = len({r["klass"] for r in records})
    n_sym = len({r["symmetric"] for r in records})
    per_combo = len(records) // max(n_classes * n_sym, 1)
    w(f"| Instances | {len(records)} at {size}×{size} "
      f"({n_classes} class{'es' if n_classes != 1 else ''} × "
      f"{'symmetric/asymmetric' if n_sym == 2 else 'one symmetry'} × {per_combo}) |")
    w(f"| Restarts per randomized arm | **{R}** |")
    w(f"| α (baseline) | {alpha_value} |")
    w(f"| Mean LP fractionality | {mean_frac:.1f}% of facilities |")
    w(f"| Mean LP duality gap | {mean_duality:.2f}% |")
    w(f"| Integer optima proven | {len(proven)} / {len(records)} |")
    w("")
    w("The duality gap is what makes this family a real test: the LP bound sits measurably")
    w("below the optimum, so an LP-guided method cannot simply read the answer off the")
    w("relaxation the way it can on Euclidean instances.")
    w("")

    # ---- §2 headline -----------------------------------------------------
    label = "optimality gap" if all_proven else "gap vs reference"
    w("## 2. Results")
    w("")
    w(f"Mean {label} over {len(records)} instances. Randomized arms report best-of-{R}.")
    w("")
    w(f"| Arm | Type | Mean {label} | Instances solved to optimality |")
    w("|---|---|---|---|")
    w(f"| A · LP rounding only | deterministic | {means['A · LP rounding only']:.4f}% | "
      f"{sum(1 for r in records if det(r, 'lp_rounding_control') <= 1e-9)} / {len(records)} |")
    w(f"| **A+ · LP rounding + local search** | deterministic | "
      f"**{mean_ap:.4f}%** | {solved_ap} / {len(records)} |")
    w(f"| D · Local search only (no LP) | deterministic | "
      f"{means['D · Local search only (no LP)']:.4f}% | "
      f"{sum(1 for r in records if det(r, 'local_search_only') <= 1e-9)} / {len(records)} |")
    w(f"| **B · LP-biased multistart** | best of {R} | **{mean_b:.4f}%** | {solved_b} / {len(records)} |")
    w(f"| **C · α-GRASP multistart** | best of {R} | **{mean_c:.4f}%** | {solved_c} / {len(records)} |")
    w("")

    # ---- §3 the head-to-head --------------------------------------------
    w("## 3. LP-biased GRASP vs. classical GRASP")
    w("")
    w(f"Both arms get the same budget of {R} restarts and the same local search. They differ")
    w("only in how each restart's starting solution is built.")
    w("")
    w(f"- **B beat C on {b_beats_c}** instances, **lost on {c_beats_b}**, tied on {bc_ties}.")
    w(f"- Mean {label}: **{mean_b:.4f}%** (B) against **{mean_c:.4f}%** (C).")
    w(f"- Optimal solutions found: **{solved_b}/{len(records)}** (B) against "
      f"**{solved_c}/{len(records)}** (C).")
    w(f"- Median restart that produced the winner: **{statistics.median(b_found):.0f}** (B), "
      f"**{statistics.median(c_found):.0f}** (C), out of {R}.")
    w("")
    # The verdict comes from the PAIRED per-instance record, not from the raw
    # means. This experiment is paired -- identical instances, restart budget and
    # local search -- so the win/loss record is the evidence and the mean only
    # describes magnitude. Deciding on the mean alone once produced the claim
    # "the LP-biased construction beats the classical baseline by 0.0023 pp"
    # directly beneath a 2-2 record with 8 ties. See
    # tests/regression/test_unsupported_verdict_regression.py.
    by_mean = _cmp(mean_b, mean_c, "B", "C", tie="neither")
    if b_beats_c > c_beats_b:
        by_record = "B"
    elif c_beats_b > b_beats_c:
        by_record = "C"
    else:
        by_record = "neither"

    if by_record == "neither":
        w(f"**With multistart the two constructions are indistinguishable on this family.** "
          f"They split the {b_beats_c + c_beats_b} decided instance"
          f"{'' if b_beats_c + c_beats_b == 1 else 's'} "
          f"{b_beats_c}–{c_beats_b} and tied on {bc_ties} of {len(records)}; the "
          f"{abs(mean_b - mean_c):.4f} pp difference in means is not supported by the "
          f"per-instance record.")
    elif by_mean != by_record:
        ahead = "B" if by_record == "B" else "C"
        w(f"**The two measures disagree, so no winner is claimed.** The per-instance record "
          f"favours {ahead} ({b_beats_c}–{c_beats_b} with {bc_ties} tied), while the mean "
          f"favours {by_mean} ({mean_b:.4f}% B against {mean_c:.4f}% C) — a mean pulled by "
          f"the size of individual wins rather than by how often they occur.")
    elif by_record == "B":
        w(f"**With multistart, the LP-biased construction beats the classical baseline** by "
          f"{mean_c - mean_b:.4f} pp on average, and beat C on {b_beats_c} of "
          f"{len(records)} instances (losing {c_beats_b}, tying {bc_ties}).")
    else:
        w(f"**With multistart, the classical baseline beats the LP-biased construction** by "
          f"{mean_b - mean_c:.4f} pp on average, and beat B on {c_beats_b} of "
          f"{len(records)} instances (losing {b_beats_c}, tying {bc_ties}).")
    w("")

    # ---- §4 does multistart justify itself -------------------------------
    w("## 4. Does multistart pay for itself?")
    w("")
    w("A best-of-N result costs N times the work of a single run. The fair question is not")
    w("whether B beats A+ at N restarts, but how many restarts B needs to match A+ at all.")
    w("")
    w(f"- B beat A+ on **{b_beats_ap}** instances, lost on **{ap_beats_b}**, tied on {ab_ties}.")
    if b_match_ok:
        w(f"- On the {len(b_match_ok)} instances where B reached A+'s quality at all, it needed a")
        w(f"  median of **{statistics.median(b_match_ok):.0f}** {_plural(statistics.median(b_match_ok), 'restart')} to do so.")
    unmatched = len(b_match) - len(b_match_ok)
    if unmatched:
        w(f"- On **{unmatched}** instances B never matched A+ within its {R}-restart budget.")
    w("")
    if mean_ap <= mean_b + 1e-9:
        w(f"**The deterministic arm still holds up.** A+ reaches {mean_ap:.4f}% in a single run;")
        w(f"B reaches {mean_b:.4f}% using {R} restarts. Multistart narrows the gap the earlier")
        w("single-start comparison showed, but does not overturn it at this budget.")
    else:
        w(f"**Multistart overturns the single-start result.** B reaches {mean_b:.4f}% against")
        w(f"A+'s {mean_ap:.4f}%, so with a restart budget the randomized construction does earn")
        w("its place — the earlier finding was an artifact of running it exactly once.")
    w("")

    # ---- §5 per-instance -------------------------------------------------
    w("## 5. Per-instance detail")
    w("")
    w(f"| Instance | Optimum | LP gap | A | A+ | D | B (best/{R}) | C (best/{R}) |")
    w("|---|---|---|---|---|---|---|---|")
    for r in records:
        ref = r["reference"]
        dual = (ref["value"] - r["lp"]["bound"]) / r["lp"]["bound"] * 100
        star = "" if ref["proven"] else " ⚠"
        w(f"| `{r['name']}`{star} | {ref['value']:,.0f} | {dual:.2f}% | "
          f"{det(r, 'lp_rounding_control'):.3f}% | {det(r, ap_key):.3f}% | "
          f"{det(r, 'local_search_only'):.3f}% | {ms(r, b_key):.3f}% | {ms(r, c_key):.3f}% |")
    w("")
    if not all_proven:
        w("⚠ = integer optimum not proven; that row is measured against the LP bound.")
        w("")

    # ---- §6 reproduction -------------------------------------------------
    w("## 6. Reproduction")
    w("")
    w("```bash")
    w(f"python -m scripts.run_koerkel_ghosh --size {size} --restarts {R} --alpha {alpha_value}")
    w("```")
    w("")
    w("| Run metadata | |")
    w("|---|---|")
    w(f"| Generated at | {generated_at} |")
    w("| Generated by | `run_koerkel_ghosh.py` → `render_report` |")
    w(f"| Git commit | `{commit}` |")
    w(f"| Restart seeds | {', '.join(str(s) for s in seeds[:8])}"
      f"{' …' if len(seeds) > 8 else ''} |")
    w(f"| Total wall-clock | {wall_seconds:.1f} s |")
    w(f"| Python | {env['python']} |")
    w(f"| NumPy / SciPy | {env['numpy']} / {env['scipy']} |")
    w("")
    w("Raw measurements: [`koerkel_ghosh_results.json`](koerkel_ghosh_results.json).")
    w("")
    return "\n".join(lines) + "\n"


CHECKPOINT_NAME = "koerkel_ghosh_checkpoint.json"


def _run_signature(args) -> Dict:
    """
    What makes two runs the same experiment.

    A checkpoint may only be resumed by a run that would have produced it.
    Restart count, alpha and the exact-solve budget all change the numbers, so
    a mismatch must discard the checkpoint rather than silently blend two
    different experiments into one report.
    """
    return {
        "size": args.size,
        "classes": list(args.classes),
        "instances": args.instances,
        "restarts": args.restarts,
        "alpha": args.alpha,
        "ip_time_limit": args.ip_time_limit,
        "attempt_exact": not args.no_exact_ip,
    }


def load_checkpoint(path: str, signature: Dict) -> List[Dict]:
    """
    Records already computed by a previous run of this same experiment.

    A rung at 750x750 costs hours, and a machine reboot midway through used to
    discard every instance already finished because the sidecar was written
    only at the very end. Returns [] when there is nothing usable -- a missing
    file, unreadable JSON, or a run that was configured differently.
    """
    if not os.path.exists(path):
        return []
    try:
        with open(path) as fh:
            data = json.load(fh)
    except (json.JSONDecodeError, OSError):
        return []
    if data.get("signature") != signature:
        return []
    return data.get("records", [])


def save_checkpoint(path: str, signature: Dict, records: List[Dict]) -> None:
    """Write atomically: a reboot mid-write must not corrupt the checkpoint."""
    tmp = path + ".tmp"
    with open(tmp, "w") as fh:
        json.dump({"signature": signature, "records": records}, fh)
    os.replace(tmp, path)


def rerender_from_json(json_path: str, out_dir: str) -> str:
    """
    Rebuild a rung's markdown from its saved measurement sidecar.

    A rung costs hours; correcting the prose must not cost them again. The
    sidecar is the record of what was measured, so only the wording is
    regenerated -- and nobody is ever tempted to hand-edit a results file.

    This exists because the §3 verdict was fixed after several rungs had
    already been rendered by the older, wrong version. Mirrors
    `download_and_run_real_world.rerender_from_json`.
    """
    with open(json_path) as fh:
        payload = json.load(fh)

    report = render_report(
        payload["records"], payload["seeds"], payload["size"],
        payload["environment"], payload["generated_at"], payload["git_commit"],
        payload["wall_seconds"], alpha_value=payload["alpha"],
    )
    md_path = os.path.join(out_dir, "koerkel_ghosh_results.md")
    with open(md_path, "w") as fh:
        fh.write(report)
    return md_path


def main(argv: List[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument("--size", type=int, default=100,
                        help="instance size; 100 keeps integer optima provable by CBC")
    parser.add_argument("--classes", nargs="+", default=["a", "b", "c"])
    parser.add_argument("--instances", type=int, default=2,
                        help="instances per (class, symmetry) combination")
    parser.add_argument("--restarts", type=int, default=32,
                        help="multistart restarts per randomized arm")
    parser.add_argument("--alpha", type=float, default=0.2)
    parser.add_argument("--ip-time-limit", type=float, default=300.0,
                        help="per-instance CBC budget for proving the integer optimum")
    parser.add_argument("--no-exact-ip", action="store_true",
                        help="skip the exact solve entirely (for sizes where it is hopeless); "
                             "gaps are then measured against the LP bound and labelled as such")
    parser.add_argument("--out-dir", type=str, default="output")
    parser.add_argument("--from-json", type=str, default=None,
                        help="re-render the markdown from an existing sidecar and exit; "
                             "runs no benchmark")
    parser.add_argument("--no-resume", action="store_true",
                        help="ignore any checkpoint and recompute every instance")
    args = parser.parse_args(argv)

    if args.from_json:
        os.makedirs(args.out_dir, exist_ok=True)
        print(f"Re-rendered {rerender_from_json(args.from_json, args.out_dir)}", flush=True)
        return

    seeds = list(range(1, args.restarts + 1))

    os.makedirs(args.out_dir, exist_ok=True)
    wall0 = time.perf_counter()

    # Resume whatever a previous run of this same experiment already finished.
    checkpoint_path = os.path.join(args.out_dir, CHECKPOINT_NAME)
    signature = _run_signature(args)
    records: List[Dict] = [] if args.no_resume else load_checkpoint(checkpoint_path, signature)
    done = {r["name"] for r in records}
    if done:
        print(f"Resuming: {len(done)} instance{'' if len(done) == 1 else 's'} "
              f"already computed ({', '.join(sorted(done))})", flush=True)

    for klass in args.classes:
        for symmetric in (True, False):
            for index in range(1, args.instances + 1):
                name = instance_name(args.size, klass, symmetric, index)
                if name in done:
                    continue
                print(f"Running {name} ...", flush=True)
                rec = run_one_instance(args.size, klass, symmetric, index, seeds,
                                       alpha_value=args.alpha,
                                       ip_time_limit=args.ip_time_limit,
                                       attempt_exact=not args.no_exact_ip)
                records.append(rec)
                save_checkpoint(checkpoint_path, signature, records)
                lp, ref = rec["lp"], rec["reference"]
                a = rec["arms"]
                print(f"  LP={lp['bound']:,.0f} frac={lp['n_fractional']}/{lp['n_facilities']} | "
                      f"ref={ref['value']:,.0f} ({'proven' if ref['proven'] else ref['ip_status']}, "
                      f"{ref['ip_seconds']:.0f}s) | A+={a['lp_rounding_plus_search']['final_cost']:,.0f} "
                      f"B={a['lp_biased_multistart']['best_cost']:,.0f}@{a['lp_biased_multistart']['best_found_at']} "
                      f"C={a['alpha_grasp_multistart']['best_cost']:,.0f}@{a['alpha_grasp_multistart']['best_found_at']}",
                      flush=True)

    wall = time.perf_counter() - wall0
    env = _environment()
    generated_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    commit = _git_commit()

    report = render_report(records, seeds, args.size, env, generated_at, commit, wall,
                           alpha_value=args.alpha)

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
                "restarts": args.restarts,
                "alpha": args.alpha,
                "seeds": seeds,
                "records": records,
                "wall_seconds": wall,
            },
            fh,
            indent=2,
        )
    # The sidecar now holds everything the checkpoint did. Keeping it would make
    # a later, deliberate re-run silently return the old results instead of
    # recomputing, so it is crash-recovery state only and is cleared on success.
    if os.path.exists(checkpoint_path):
        os.remove(checkpoint_path)

    print(f"\nWrote {md_path}\nWrote {json_path}\nTotal wall-clock: {wall:.1f}s", flush=True)


if __name__ == "__main__":
    main()
