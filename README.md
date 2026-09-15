# MS-LP-GRASP — Multistart GRASP with LP-Biased Construction for the UFLP

A from-scratch implementation and **adversarial evaluation** of a hybrid heuristic for the
Uncapacitated Facility Location Problem: a GRASP whose constructive phase samples
facilities using the fractional values of the LP relaxation instead of a uniform or
savings-based rule, followed by a best-improvement local search with exact `save`/`loss`/`extra`
delta evaluation (Resende & Werneck, 2006).

The interesting part of this repository is not the heuristic. It is the evaluation.
The method was run against four ablations on **48 Körkel-Ghosh instances spanning six
sizes**, a 2000×2000 real-world instance, and a classical OR-Library control — and the
headline claim it started with did not survive.

---

## Contents

- [Headline results](#headline-results)
- [What the evidence supports — and what it does not](#what-the-evidence-supports--and-what-it-does-not)
- [The problem](#the-problem)
- [The method](#the-method)
- [Repository structure](#repository-structure)
- [Datasets](#datasets)
- [Results in detail](#results-in-detail)
- [Defects found during evaluation](#defects-found-during-evaluation)
- [Limitations and threats to validity](#limitations-and-threats-to-validity)
- [Reproducing](#reproducing)
- [Verification methodology](#verification-methodology)
- [References](#references)

---

## Headline results

**48 Körkel-Ghosh instances, six sizes, 32 restarts per randomized arm, 30-minute exact-solve
attempt on every instance.** Gaps are true optimality gaps where CBC closed the instance and
gaps against the LP bound otherwise; the per-size table says which.

| Size | Instances | Optima proven | A+ · rounding + search | D · no LP | **B · MS-LP-GRASP** | C · α-GRASP |
|---|---|---|---|---|---|---|
| 100×100 | 12 | **12 / 12** | 0.1129% | 0.0920% | **0.0000%** | 0.0030% |
| 150×150 | 6 | **6 / 6** | 0.0460% | 0.1510% | **0.0000%** | 0.0000% |
| 200×200 | 6 | 4 / 6 | 0.4786% | 0.4865% | **0.3625%** | 0.3667% |
| 250×250 | 12 | 1 / 12 | 1.2522% | 1.3034% | **1.2030%** | 1.2053% |
| 500×500 | 6 | 0 / 6 | 1.5099% | 1.5532% | **1.4173%** | 1.4180% |
| 750×750 | 6 | 0 / 6 | 1.3425% | 1.3836% | **1.2386%** | 1.2416% |

Three findings, in descending order of how well the evidence supports them.

### 1. Multistart is the contribution. It is large, and it holds at every size.

Arm **B** and arm **A+** use the *same* LP-biased construction and the *same* local search.
They differ only in whether it runs once or 32 times.

**B beats A+ on 31 of 48 instances and loses on 2.** The margin is positive at every single
size: `+0.1129`, `+0.0460`, `+0.1161`, `+0.0492`, `+0.0927`, `+0.1039` percentage points.

This repository previously carried the opposite conclusion — that the randomized construction
contributed nothing. That was correct *of the code as it then stood*, which performed one
construction and one local search. Randomization had no mechanism through which to pay off.
Adding the restart loop reverses the finding.

### 2. The LP bias does **not** beat a classical GRASP at the sizes that matter.

Across all 48 instances B beats C on 9, loses on 6, and **ties on 33**. Split by size, the
picture is unambiguous:

| | B beats C | C beats B |
|---|---|---|
| 100 / 150 / 200 | **3** | 0 |
| 250 / 500 / 750 (official KG sizes) | 6 | **6** |

Every win is on the smaller instances. At the library's real sizes it is a dead heat, and the
mean gaps differ by 0.0023 / 0.0007 / 0.0030 pp — differences far below anything that would
survive a change of seed. **The report says so in its own generated words:**

> *"But those wins are **concentrated on the smaller instances**: B leads 3–0 on the smallest
> 3 sizes (100, 150, 200) and only 6–6 on the largest 3 (250, 500, 750). On this evidence the
> advantage **does not persist** at the sizes the family exists to test."*

### 3. Where the LP bias *does* pay: per-restart reliability, not best-of-N.

Comparing the arms restart-by-restart rather than best-of-32 reverses the verdict:

| Size | B mean per-restart gap | C mean per-restart gap | B better on |
|---|---|---|---|
| 100×100 | **1.3596%** | 1.4052% | 8 / 12 |
| 150×150 | **1.2680%** | 1.3041% | 5 / 6 |
| 200×200 | **1.5556%** | 1.5764% | 6 / 6 |
| 250×250 | **1.5598%** | 1.5772% | 8 / 12 |
| 500×500 | **1.4905%** | 1.5188% | 6 / 6 |
| 750×750 | **1.3269%** | 1.3419% | 6 / 6 |

**B's average restart is better than C's at every size, on 39 of 48 instances.** Taking the
best of 32 washes that advantage out — enough independent draws let the savings-based
construction find an equally good solution eventually.

The practical consequence shows up clearly on California (2000×2000, 10 restarts), where both
arms reach 0.0358%:

- B reaches its best at **restart 2**; C at **restart 4**.
- **B's *worst* restart (0.0421%) is better than C's *mean* restart (0.1618%)** — an order of
  magnitude less spread.
- B takes 2099 s against C's 3628 s: **1.73× faster**, despite paying 126 s for the LP solve.

So the LP bias buys **variance reduction and time-to-good-solution**, not a better asymptote.
That matters when the restart budget is small. It stops mattering when it is large.

---

## What the evidence supports — and what it does not

**Supported.**
- Multistart substantially improves the LP-biased heuristic at every size tested (31–2–15 over 48).
- LP-biased construction produces more reliable individual restarts than savings-based α-GRASP
  (39 of 48 instances), and reaches a good solution in fewer restarts and less wall-clock.
- The LP relaxation on Körkel-Ghosh is genuinely weak — 22–28% of facilities fractional,
  measured duality gaps of 1.19–3.01% where an optimum was proven — so this is a real test of a
  method premised on the relaxation carrying information.

**Not supported.**
- That LP-biased construction beats a classical GRASP on final solution quality. At the
  library's official sizes it does not (6–6).
- That the LP relaxation is worth its cost as a *seed*. Comparing A+ against D isolates exactly
  that, and the result is inconsistent across sizes: `−0.0209`, `+0.1050`, `+0.0079`, `+0.0512`,
  `+0.0432`, `+0.0411` pp. It is **negative at 100×100**. The generated report refuses to
  summarise it: *"The LP advantage is **not consistent across sizes** … Reporting a single
  headline number for it would misrepresent the evidence."*
- Anything at all about instances larger than 750×750 on this family, or about the official
  UflLib files (see [Limitations](#limitations-and-threats-to-validity)).

**A negative result worth stating plainly:** on the near-integral California instance
(7 of 2000 facilities fractional, 0.35%), arm D — a local search that never reads the LP —
reaches **exactly** the same 0.0421% as the full LP-rounding-plus-search pipeline, and on
cap134 every one of the five arms reaches the proven optimum. On easy instances the LP machinery
buys nothing but time.

---

## The problem

Given facilities `F` with setup costs `c_f`, customers `U` with service costs `d_uf`, choose
`S ⊆ F, S ≠ ∅` minimising

```
cost(S) = Σ_{f∈S} c_f  +  Σ_{u∈U} min_{f∈S} d_uf
```

Every customer is served by its cheapest open facility; there are no capacities. The strong
formulation used throughout disaggregates the linking constraints as `x_uf ≤ y_f`, which is what
makes the relaxation tight enough to be informative on Euclidean instances — and conspicuously
weak on Körkel-Ghosh.

## The method

### Pipeline

1. **LP relaxation** (`solve_lp_relaxation`) — `y_f ∈ [0,1]`, solved with SciPy/HiGHS. Returns
   the fractional `y` vector as per-facility opening probabilities.
2. **Probabilistic construction** (`construct_solution`) — open each facility independently with
   probability `max(y_f, ε)`, `ε = 0.01`. Falls back to the single cheapest facility if none open.
3. **Best-improvement local search** (`local_search`) — insertions, deletions and swaps, with
   move profit from **exact** delta formulas rather than incremental bookkeeping.
4. **Multistart** (`multistart.py`) — repeat 2–3 over `N` seeds, keep the best. This is the step
   that makes it a GRASP, and the step that turned out to matter.

### The five arms

| Arm | Construction | Reads LP? | Randomized? | Restarts |
|---|---|---|---|---|
| A | open every `y_f ≥ 0.5`, no search | yes | no | 1 |
| A+ | A, then local search | yes | no | 1 |
| D | cheapest facility, then local search | **no** | no | 1 |
| **B · MS-LP-GRASP** | sample `p = max(y_f, ε)` | yes | yes | N |
| C · α-GRASP | savings-based RCL, α = 0.2 | no | yes | N |

The design is what makes the conclusions attributable:

- **A+ vs B** differ *only* in deterministic rounding versus sampling → isolates **randomization**.
- **A+ vs D** differ *only* in whether the start is LP-seeded → isolates **the LP**.
- **B vs C** differ *only* in how each restart is built → isolates **the LP bias itself**.

Deterministic arms run once, because restarting a deterministic construction reproduces the
same solution.

### Exact delta evaluation

`save`/`loss`/`extra` are maintained per Resende–Werneck, but acceptance always uses the exact
delta formulas — the auxiliary structures are informational. `_apply_move_and_recompute` fully
recomputes closest/second-closest assignments after every move, so the implementation is
correctness-first rather than maximally fast. The exactness is not assumed: a test brute-forces
every subset of ~400 random instances and checks each predicted delta against the true cost change.

---

## Repository structure

```
scripts/
  uflp_solver.py              canonical solver: data structures, I/O, LP, construction, local search
  multistart.py               the restart loop; fairness accessors (restarts/seconds to reach a target)
  reference.py                what a gap is measured against; proven optimum vs LP bound
  koerkel_ghosh.py            generator for the hard benchmark family, to published spec
  run_koerkel_ghosh.py        the five arms over one KG size; checkpointing; --from-json re-render
  run_full_study.py           the whole size ladder + California; cross-size analysis
  run_cap134.py               OR-Library control instance
  download_and_run_real_world.py   California Housing 2000×2000
  run_experiments.py          cap134 + LP duality-gap correlation suite
  run_scaling_benchmark.py    wall-clock scaling sweep
tests/
  unit/                       boundaries, invariants, exact deltas, report narrative
  regression/                 one file per fixed defect, each proven red before the fix
output/                       every generated artifact (never hand-edited)
```

### Reports are generated, never written

Every figure in a results file is a measured value carried in a typed record, and every
comparative phrase — "lower", "faster", "indistinguishable", "widens" — is **computed from those
values**, so a report cannot state a conclusion its own table contradicts. Regression suites
enforce it by feeding the renderers mirror-image datasets and requiring the wording to flip, and
by scanning generator source for hardcoded verdicts.

This is not a stylistic preference. Three separate defects in this repository were reports
asserting things their own tables denied; see [Defects found](#defects-found-during-evaluation).

---

## Datasets

### Körkel-Ghosh — the decisive benchmark

The standard hard family. Allocation costs are drawn uniformly from `[1000, 2000]` rather than
from a metric embedding, which destroys the structure that makes the relaxation tight. Setup
costs by class: **a** `[100,200]`, **b** `[1000,2000]`, **c** `[10000,20000]`. Symmetric (`gs`)
and asymmetric (`ga`) variants; official sizes 250, 500 and 750.

Measured LP fractionality: **22.3–28.1%** of facilities, against 0.35% on California and 0% on
cap134. This is the family where an LP-guided method has something to prove.

> **Generated to specification, not the official UflLib files** — those archives returned HTTP 403
> from this environment. The generator follows the published spec exactly, but the random draws
> differ, so **objective values are not comparable with published KG results**. Arm-versus-arm
> comparison, which is what this study measures, is unaffected.

### California Housing — 2000×2000, real-world geography

Block-group centroids from the California Housing dataset; service cost is great-circle distance
× 10/km, setup cost scaled by local density. Four million service-cost entries. The integer
program is not attempted at this size — four million binary-linked assignment variables.

### OR-Library cap134 — the control

Beasley's 50×50 instance, included *because* it is easy: its LP relaxation is integral, so the
bound equals the optimum (928,941.75) and all five arms find it. A method that looked good here
and nowhere else would be suspect.

---

## Results in detail

Full generated reports:

| Artifact | What it covers |
|---|---|
| [`output/full_study.md`](output/full_study.md) | The six-rung ladder + California; hardness vs provability |
| [`output/kg{100,150,200,250,500,750}/`](output/) | One report per size, with per-instance detail |
| [`output/california_4M_results.md`](output/california_4M_results.md) | 2000×2000 real-world instance |
| [`output/cap134_results.md`](output/cap134_results.md) | The control instance |

### Where exact solution stops being possible

This is the other half of the study, and it is why the ladder spans six sizes rather than one.

| Size | Optima proven (30 min CBC each) | Mean CBC time | Mean duality gap |
|---|---|---|---|
| 100×100 | **12 / 12** | 17.4 s | 1.25% |
| 150×150 | **6 / 6** | 254.9 s | 1.19% |
| 200×200 | 4 / 6 | 834.1 s | 1.68% |
| 250×250 | 1 / 12 | 1634.8 s | 3.01% |
| 500×500 | **0 / 6** | 2214.2 s | — |
| 750×750 | **0 / 6** | 3867.2 s | — |
| 2000×2000 | not attempted | — | — |

Ground truth is available in full only up to **150×150**. It is already almost gone at the
*smallest official Körkel-Ghosh size*. At 500 and 750 it is gone entirely, and the exact solve is
not merely slow — a 750×750 model carries 562,500 binary-linked assignment variables and peaked
at 2.2 GB of solver memory, enough to require swap on a 12 GB machine.

**Small instances give certainty and little difficulty; large ones give difficulty and no
certainty.** Reporting only one end of that trade-off reports half the result, which is why the
gaps above 200×200 are labelled as measured against a bound, per instance, in both the tables and
the JSON.

---

## Defects found during evaluation

The evaluation found more in the harness than in the heuristic. Each has a regression test under
`tests/regression/`, proven to fail on the pre-fix code.

| Defect | Consequence |
|---|---|
| **An unproven CBC incumbent reported as a proven optimum.** PuLP's `LpStatus` reads `"Optimal"` whenever CBC terminates holding an integer-feasible solution — *including a time-limit stop*. | Reports labelled gaps "true optimality gaps" against values CBC never proved. On `ga250a-1` the "proven optimum" of 257,553 was **beaten by three of the run's own arms**; the winning 257,538 solution was rebuilt and its cost recomputed from first principles to confirm the heuristic was right and the label was wrong. Would have printed *negative* optimality gaps. Fixed by requiring `sol_status == "Optimal Solution Found"`, plus a solver-agnostic guard that demotes any claimed optimum a feasible solution beats. |
| **A 0.0023 pp difference reported as one arm "beating" another.** The verdict came from a comparison with a 1e-9 tie tolerance. | The 250×250 report printed *"the LP-biased construction beats the classical baseline"* directly beneath *"B beat C on 2, lost on 2, tied on 8"*. Fixed so the verdict is decided by the paired per-instance record; when record and mean disagree the report says so and claims no winner. |
| **Cross-size trends asserted against the tables.** "The LP bias holds its advantage **as instances grow**" (all wins were on the small rungs); "the margin **widens** … +0.0030 pp … +0.0030 pp" (two equal numbers). | Fixed by splitting the win record at the median size and testing the claim on the larger half, and by deciding trend words from the values *as printed*. |
| **`forkserver` broke LP probability passing.** Python 3.14 changed the default start method; a module-global dict never reached workers. | Silent wrong-instance results in parallel benchmarks. |
| **`conftest.py` added the wrong root to `sys.path`.** | mutmut tested *unmutated* code and reported "could not find any test case for any mutant". |
| **Coverage silently omitted three modules.** `multistart.py`, `reference.py`, `run_cap134.py` were in the `omit` list. | The gate reported 100% while measuring nothing about them. |
| **A rung wrote its sidecar only at the end.** | A machine reboot 5 h into a 6-instance 750×750 rung discarded all four finished instances. Fixed with per-instance atomic checkpointing that refuses to resume across a changed configuration. |

---

## Limitations and threats to validity

1. **Generated instances, not official UflLib files.** Objective values are not comparable with
   published Körkel-Ghosh results. Arm-versus-arm comparison is unaffected.
2. **Most large-instance gaps are against the LP bound, not a proven optimum**, and therefore
   overstate the true optimality gap by an unknown amount. Every table labels which is which.
3. **Single α.** The classical baseline uses α = 0.2 throughout. A tuned or reactive α might
   perform differently.
4. **Equal restarts, not equal time.** B and C each get 32 restarts, but B additionally pays for
   the LP solve — 217 s at 750×750. Given equal *wall-clock*, C would get roughly 1.4× more
   restarts at that size. The per-restart reliability advantage in finding 3 is real; the
   best-of-N comparison is favourable to B in a way equal-time budgeting would erode.
5. **One machine, one solver.** CBC, not Gurobi or CPLEX. A stronger solver would push the
   provability frontier up; it would not change the arm comparison.
6. **Instance counts are small at the large sizes** — 6 instances at 500 and 750. A 6–6 split is
   consistent with "no difference" but is not a tight bound on the difference.

---

## Reproducing

```bash
pip install -r requirements.txt

# One instance
python -m scripts.uflp_solver --file data/cap134.txt

# The full six-rung ladder + California  (~30 h; checkpoints and reuses finished rungs)
python -m scripts.run_full_study

# A single size
python -m scripts.run_koerkel_ghosh --size 250 --instances 2 --restarts 32

# Re-render prose from measurements, running no benchmark
python -m scripts.run_koerkel_ghosh --from-json output/kg250/koerkel_ghosh_results.json --out-dir output/kg250
python -m scripts.run_full_study --from-stages --sizes 100 150 200 250 500 750
python -m scripts.download_and_run_real_world --from-json output/california_4M_results.json
```

Run everything from the repository root; scripts resolve paths relative to the working directory.

## Verification methodology

| Layer | Gate | Status |
|---|---|---|
| Unit tests & invariants | 100% pass, zero hollow assertions | 376 passing |
| Coverage | ≥ 85% line coverage | **100%** across all 9 measured modules |
| Mutation testing | ≥ 85% on changed logic | `mutmut run` over solver, reference and multistart |
| Regression contract | one test per defect, proven red first | 8 regression suites |

Every bug fix carries a dedicated regression test that was **run against the pre-fix code and
observed to fail for the expected reason**, documents the defect, and includes a counterweight
case asserting the fix did not overshoot.

## References

- Resende, M. G. C., & Werneck, R. F. (2006). *A hybrid multistart heuristic for the
  uncapacitated facility location problem.* European Journal of Operational Research, 174(1), 54–68.
- Körkel, M. (1989). *On the exact solution of large-scale simple plant location problems.*
  European Journal of Operational Research, 39(2), 157–173.
- Ghosh, D. (2003). *Neighborhood search heuristics for the uncapacitated facility location
  problem.* European Journal of Operational Research, 150(1), 150–162.
- Beasley, J. E. (1990). *OR-Library: distributing test problems by electronic mail.*
  Journal of the Operational Research Society, 41(11), 1069–1072.
