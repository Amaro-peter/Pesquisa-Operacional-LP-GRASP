# Real-World UFLP Benchmark — California Housing

A 2000×2000 uncapacitated facility location instance built from real census
block coordinates and populations, used to compare an LP-biased hybrid GRASP against a
classical α-GRASP baseline **and** against a no-heuristic control.

> **Provenance.** Every number below was measured by the run described in
> §11 and written directly by `render_report` in `download_and_run_real_world.py`.
> No value in this file is hardcoded or back-computed, and every comparative
> word ("lower", "faster", "tighter") is derived from the measured values
> rather than written into the template. The raw records are in
> `output/california_4M_results.json`.

---

## 1. Instance

| Property | Value |
|---|---|
| Source dataset | California Housing (`sklearn.datasets.fetch_california_housing`) |
| Facilities × customers | 2,000 × 2,000 |
| Sampling | two **disjoint** random samples of census blocks, `np.random.seed(42)` |
| Facility coordinates | block latitude / longitude (columns 6, 7) |
| Setup cost | `5,000 × (population / median population) + 1,000` |
| Service cost | Euclidean degree distance × 111.0 km/deg × 10.0 per km |

### LP model size

| Quantity | Count |
|---|---|
| Decision variables | 4,002,000 &nbsp;(`2,000` y + `4,000,000` x) |
| Assignment rows (`Σ_f x[u][f] = 1`) | 2,000 |
| Linking rows (`x[u][f] ≤ y[f]`) | 4,000,000 |
| Total constraint rows | 4,002,000 |
| Nonzeros in the linking matrix | 8,000,000 |

*Describing this instance as “4,000,000 variables” counts only the assignment
variables `x`. The full model carries 4,002,000 variables and 4,002,000 constraint rows —
it is the linking rows, not the variables, that make the model heavy.*

### Known approximation in the distance model

Longitude degrees are converted at the same 111.0 km/degree as latitude degrees.
At Californian latitudes (~37°N) a longitude degree is closer to 88 km, so east–west
separation is overstated by roughly 26%. The instance is geographically *derived*,
not geographically *accurate*. This is pinned by a unit test so it cannot drift silently.

## 2. LP relaxation

| Quantity | Value |
|---|---|
| **LP bound** | **397,600.48** |
| Solve time | 164.24 s |
| Solver | SciPy `linprog(method='highs')` — HiGHS selects the algorithm |
| Facilities with `y ≥ 0.99` (integral, open) | 54 / 2,000 |
| Facilities with `0.01 < y < 0.99` (**fractional**) | **7** / 2,000 (0.35%) |
| Facilities with `y ≤ 0.01` | 1,939 / 2,000 |
| `Σ y[f]` (fractional facility count) | 57.50 |
| Facilities opened by rounding at `y ≥ 0.5` | 61 |

**Read:** the relaxation is essentially integral — 7 of 2000 facilities (0.35%) take a strictly fractional value. This matters for
interpretation: a near-integral relaxation means the LP has very nearly *solved* the
problem, and the heuristic's job is repair rather than search.

> **The LP bound is a bound, not a proven optimum.** All gaps below are measured
> against it. Because the relaxation is not perfectly integral, the true integer
> optimum lies somewhere in `[397,600.48, 397,767.97]`, so the *true*
> optimality gaps are **smaller** than the figures reported here. Solving the exact
> IP at this scale is not tractable with CBC; `run_experiments.solve_exact_ip` does
> it for the smaller OR-Library instances.

## 3. Methods compared

| | Method | Construction | Local search | Pays for LP |
|---|---|---|---|---|
| **A** | LP rounding (control) | open every `y ≥ 0.5` | none | yes |
| **A+** | Rounding + local search | open every `y ≥ 0.5` (deterministic) | best-improvement (insert / delete / swap) | yes |
| **B** | LP-biased hybrid GRASP | open `f` with probability `max(y[f], 0.01)` | best-improvement (insert / delete / swap) | yes |
| **C** | α-GRASP baseline | savings-based RCL, `α = 0.2` | same | no |

Arm **A** is the control the earlier revision of this report lacked: without it there is no
way to tell whether the GRASP layer contributes anything beyond what the LP already knew.

Arm **A+** is the ablation. It differs from arm B in exactly one respect — the construction
is deterministic rounding rather than probabilistic sampling — and shares the local search,
the LP and the instance. Any difference between A+ and B is therefore attributable to
randomization and to nothing else. §6 reports it.

## 4. Results

Heuristic arms were run on 3 seeds (42, 7, 2024); values are means across seeds. The control is deterministic.

| Metric | A · LP rounding | A+ · Rounding + LS | B · LP-biased GRASP | C · α-GRASP | B vs C |
|---|---|---|---|---|---|
| Initial cost | 401,706.88 | 401,706.88 | 525,827.21 | 471,474.95 | B starts higher |
| Initial gap vs LP bound | 1.0328% | 1.0328% | 32.2501% | 18.5801% | +13.6701 pp |
| Facilities opened by construction | 61 | 61 | 76.0 | 64.0 | — |
| Local-search iterations | 0 | 5 | 20.3 | 34.7 | B does fewer |
| **Final cost** | **401,706.88** | **397,767.97** | **397,767.97** | **398,355.75** | -587.78 |
| **Final gap vs LP bound** | **1.0328%** | **0.0421%** | **0.0421%** | **0.1900%** | B is tighter by 0.1478 pp |
| Facilities open at the end | 61 | 58 | 58.0 | 58.0 | — |
| **End-to-end time** | **177.01 s** | **221.61 s** | **373.90 s** | **393.79 s** | B is faster by 19.89 s |

*A, A+ and B include the 164.24 s LP solve in their end-to-end time; C does not.*

**Best final gap: LP rounding + local search at 0.0421%.**

### Per-seed detail

| Seed | B init gap | B iters | B final gap | B time | C init gap | C iters | C final gap | C time |
|---|---|---|---|---|---|---|---|---|
| 42 | 34.2001% | 20 | 0.0421% | 426.86 s | 18.0699% | 36 | 0.3444% | 424.23 s |
| 7 | 40.6400% | 27 | 0.0421% | 406.87 s | 19.3404% | 38 | 0.1834% | 420.73 s |
| 2024 | 21.9104% | 14 | 0.0421% | 287.96 s | 18.3299% | 30 | 0.0421% | 336.41 s |

Final-gap spread: B 0.0421–0.0421%, C 0.0421–0.3444%.

**Arm B reached the identical solution on all 3 seeds** — the same
58 open facilities at 397,767.97 — despite starting from
initial gaps spanning 21.91%–40.64%. The EPS noise changes
where the search *starts* but not where it *ends*.

Arm B produced 1 distinct final solution across the 3 seeds; arm C produced
3. Measured as final-gap spread, **B** is the more consistent arm
(0.0000 pp for B versus 0.3022 pp for C). Consistency is a separate
property from mean quality, and on a randomised heuristic it is often the more
useful one: it is what lets a single run be trusted without replication.

## 5. What the construction actually produces

The LP wants about **61** facilities open (`y ≥ 0.5`). The LP-biased
construction opened **76.0** on average — an excess of **+15.0**.

That excess is not a property of the LP. It is the `EPS = 0.01` floor in
`construct_solution`: every facility is opened with probability `max(y[f], EPS)`, so each
of the **1,939** facilities the LP set to `y ≤ 0.01` still gets a
1% chance. The expected number of such spurious openings is
`0.01 × 1,939 = 19.4`, which is the same order as the
+15.0 observed.

Each spurious facility carries a setup cost, and that is what produces arm B's
32.25% initial gap. **The initial gap is an artifact of the EPS floor, not of LP
fractionality** — §2 shows the relaxation is only 0.35% fractional.

Because the floor is a fixed per-facility constant, its damage grows linearly with
instance size: negligible on a 50-facility instance, ~19 spurious
openings here. Scaling it as `EPS ∝ 1/|F|`, or
skipping facilities the LP zeroes confidently, would cut both the initial gap and the
repair work it creates.

## 6. What does the randomized construction contribute?

Arms A+ and B share the LP, the instance and the local search. They differ only in how
the starting solution is built: **A+ rounds the LP deterministically, B samples from it**.
So the difference between them is what randomization buys.

| | A+ · deterministic rounding | B · randomized sampling |
|---|---|---|
| Facilities opened by construction | 61 | 76.0 |
| Construction gap | 1.0328% | 32.2501% |
| Local-search iterations | **5** | **20.3** |
| Move mix | 0 insert / 3 delete / 2 swap | see §7 |
| **Final cost** | **397,767.97** | **397,767.97** |
| **Final gap** | **0.0421%** | **0.0421%** |
| Heuristic time, excluding the shared LP | 57.37 s | 209.65 s |

**Both arms reach the identical solution** — 397,767.97 at
0.0421%, on every seed. Randomization changes nothing about the
answer.

It is not free. The randomized construction opens +15.0
more facilities than rounding does, and the local search then needs 4.1× as many
iterations (20.3 against 5) to undo them — 3.7× the heuristic
time once the shared LP solve is excluded from both.

> **This is the experiment that decides what the method contributes**, and it is the one the
> original study did not run. Without arm A+, a comparison against α-GRASP cannot separate
> "the LP relaxation is informative" — which it plainly is — from "biasing a random
> construction with the LP is informative", which is the actual claim being made.

One further caveat on the framing: GRASP is a *multistart* procedure, and randomization
exists in it so that restarts explore different basins and the best is kept. This pipeline
performs a single construction and a single local search — there is no restart loop anywhere
in the codebase. With one start, randomization has no mechanism through which to pay off.

## 7. What the local search actually does

Move mix, summed across all seeds:

| Arm | Inserts | Deletes | Swaps | Total | Open facilities |
|---|---|---|---|---|---|
| B · LP-biased | 0 | 54 | 7 | 61 | 76.0 → 58.0 |
| C · α-GRASP | 0 | 18 | 86 | 104 | 64.0 → 58.0 |

**The two arms are not doing the same work.** 89% of arm B's moves are
deletions — it is removing the facilities the EPS floor opened by mistake. 83% of
arm C's moves are swaps — genuinely relocating facilities, which is the more expensive
kind of move to find and the harder kind of improvement to make.

So the iteration counts are not directly comparable as a measure of “effort saved”:
fewer iterations here partly reflects that deletions are cheap repairs of a self-inflicted
problem, not that arm B started from a structurally better configuration.

## 8. Where the time goes

| Stage | B · LP-biased | C · α-GRASP | Difference (B − C) |
|---|---|---|---|
| LP relaxation | 164.24 s | 0.00 s | +164.24 s |
| Construction | 9.31 s | 108.59 s | -99.28 s |
| Local search | 200.34 s | 285.20 s | -84.86 s |
| **Total** | **373.90 s** | **393.79 s** | **-19.89 s** |

Arm B is **faster** overall by 19.89 s. That net figure is the sum of three
stage-level differences pulling in different directions:

- **Local search — saves arm B 84.86 s.** Arm B runs 20.3 iterations to arm C's
  34.7, at a broadly similar cost per iteration.
- **Construction — saves arm B 99.28 s.** α-GRASP's constructor is a sequential greedy loop
  that re-scores every closed facility against every customer on every pass; the LP-biased
  constructor is a single sampling sweep. This term is a property of the *baseline's
  constructor*, not of the LP bias, and it is easy to overlook.
- **LP relaxation — costs arm B 164.24 s**, paid only by arm B.

The largest single term is **the LP solve**. Attributing the whole difference to any one
stage would misstate where the time actually goes.

## 9. Findings

**1. The LP relaxation is essentially integral.** 7 of 2000 facilities
(0.35%) are fractional. Any explanation of arm B's behaviour that rests on
heavy fractionality is not supported by this instance.

**2. Arm B's 32.25% initial gap is caused by the EPS floor.** It opens about
19 facilities the LP had zeroed
(+15.0 observed above the LP's 61), each carrying a setup cost. See §5.

**3. The randomized construction contributes no measurable quality.** Deterministic LP
rounding followed by the same local search (arm A+) reaches the identical solution —
397,767.97 at 0.0421% — in 5
local-search iterations against 20.3 for the
randomized arm. The sampling opens facilities the LP had already ruled out and the local
search then removes them again. See §6.

**4. The local search earns its keep.** Arm A+ takes the same rounded solution the
control stops at (1.0328%) and improves it to 0.0421% —
better by a factor of 24.5× — in 5 iterations. Because arm A+ introduces no construction
noise of its own, this is a clean demonstration that the local search does real work
rather than merely repairing a randomized start. Rounding the LP alone is not enough.

**5. Arm B ends better than arm C** (0.0421% vs 0.1900%) and is
faster by 19.89 s.
Both results favour arm B. The speed margin, however, is not a single effect: §8
shows it splitting across local search (-84.86 s), construction
(-99.28 s) and the LP solve (+164.24 s). Attributing it entirely
to reduced local-search workload would overstate the role of the LP bias, because the
constructor term belongs to the baseline's greedy loop rather than to anything the LP did.

**6. Arm B is the more reproducible.** Across 3 seeds arm B's final gap
spanned 0.0000 pp (1 distinct solution) and arm C's spanned 0.3022 pp
(3 distinct solutions). A single-seed comparison
cannot see this, and it is arguably the more practically important difference:
an arm whose answer does not depend on the seed can be run once.

Note the scope of that claim, though: arm A+ is *deterministic*, so it reaches
0.0421% with no seed at all. Reproducibility here is a property of
not randomizing, which arm A+ achieves more directly than arm B does.

**7. The iteration counts measure different work.** 89% of arm B's moves are
deletions; 83% of arm C's are swaps (§7). Deletions repair the construction's own
EPS noise and are cheap to find; swaps relocate facilities and are the harder improvement.
A raw “fewer iterations” comparison is therefore a weaker claim than it first appears.

## 10. Limitations

- **Gaps are measured against the LP bound, not a proven integer optimum.** The true
  optimality gaps are smaller than reported. Calling these “optimality gaps” without
  qualification would be incorrect.
- **3 seeds.** Enough to show a spread, not enough for a confidence interval.
- **Only compare timings within this run.** Every figure here comes from one sequential
  session on one machine. Wall-clock for this workload is sensitive to memory pressure
  and concurrent load, and repeated runs of the same configuration have differed by
  tens of percent. The cost, gap, iteration and move-mix figures are fully deterministic
  under the stated seeds; the timings are not, and should not be quoted across runs.
- **The distance model overstates east–west separation by ~26%** (§1).
- **The HiGHS algorithm is not recorded.** `method='highs'` lets the solver choose between
  simplex and interior point, so the LP solve time should not be labelled “simplex time”.
- **A single alpha value** (0.2) was tested for the baseline; it was not tuned.

## 11. Reproduction

```bash
pip install -r requirements.txt
python download_and_run_real_world.py --size 2000 --seeds 42 7 2024
```

| Run metadata | |
|---|---|
| Generated at | 2026-09-12 04:40 UTC |
| Generated by | `download_and_run_real_world.py` → `render_report` |
| Git commit | `f7fbf89` |
| Dataset sample seed | 42 |
| Heuristic seeds | 42, 7, 2024 |
| Total wall-clock | 2052.6 s |
| Python | 3.14.6 |
| NumPy / SciPy / scikit-learn | 2.5.3 / 1.18.1 / 1.9.1 |
| Platform | Linux-6.18.33.2-microsoft-standard-WSL2-x86_64-with-glibc2.39 |
| CPU count | 12 |

Raw measurements: [`california_4M_results.json`](california_4M_results.json).

