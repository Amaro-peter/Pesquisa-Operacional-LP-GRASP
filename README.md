# LP-XLS — LP-Seeded Exact-Delta Local Search for the Uncapacitated Facility Location Problem

A solver, a benchmark harness, and an honest evaluation of whether an LP relaxation
can usefully guide a local-search heuristic for the **Uncapacitated Facility Location
Problem (UFLP)**, also known as the Simple Plant Location Problem.

The short answer, stated up front because it is the least flattering one: **the LP
relaxation helps, the local search helps, and the randomized construction — the part
originally proposed as the contribution — does not.**

---

## Contents

- [Summary of findings](#summary-of-findings)
- [A note on naming](#a-note-on-naming)
- [The problem](#the-problem)
- [The method](#the-method)
- [Repository structure](#repository-structure)
- [Datasets](#datasets)
- [Results](#results)
- [What the evidence supports](#what-the-evidence-supports)
- [Limitations and threats to validity](#limitations-and-threats-to-validity)
- [Reproducing](#reproducing)
- [Verification methodology](#verification-methodology)
- [References](#references)

---

## Summary of findings

| Claim | Verdict | Evidence |
|---|---|---|
| The LP relaxation carries useful information | **Supported** | On Körkel-Ghosh, an LP-seeded search reaches 0.045% mean excess vs 0.115% for an identical search seeded without the LP |
| Local search is necessary | **Supported** | Rounding the LP and stopping gives 1.03% (California) and 16.87% (Körkel-Ghosh); adding local search gives 0.04% and 0.045% |
| The randomized construction improves quality | **Not supported** | Deterministic rounding matches or beats it on both families; on California both reach the *identical* solution |
| The method is a GRASP | **Not supported** | There is no multistart loop anywhere in the codebase |
| Gaps reported are optimality gaps | **Only on cap134** | Elsewhere gaps are measured against the LP bound, which is a lower bound, not a proven optimum |

The single most consequential measurement in this repository is the ablation in
[Randomization: the ablation](#randomization-the-ablation). Everything else is context for it.

---

## A note on naming

This repository was originally titled *"Hybrid LP-GRASP"*, after Resende & Werneck's
hybrid multistart heuristic. That name does not describe what is implemented here, for
two independent reasons:

1. **GRASP is a multistart procedure.** Its randomization exists so that repeated
   restarts explore different basins and the best result is kept. This codebase performs
   **one** construction followed by **one** local search. `grep -rn "multistart"` returns
   only the bibliography entry. With a single start, randomization has no mechanism
   through which to pay off.
2. **The randomization measurably contributes nothing.** See the ablation below.

Accordingly, this README names the implemented method by what it does:

> **LP-XLS — LP-Seeded Exact-Delta Local Search**

with two seeding policies:

| Policy | Name | Construction rule |
|---|---|---|
| Deterministic | **LP-XLS/R** | open every facility with `y_f ≥ 0.5` (threshold rounding) |
| Stochastic | **LP-XLS/S** | open facility `f` with probability `max(y_f, ε)`, `ε = 0.01` |

`LP-XLS/S` is the configuration the project originally proposed. `LP-XLS/R` is the
ablation that removes its randomness and changes nothing else. This is not a claim of
novelty — LP rounding followed by local search is long-established. The name is used
here only so that the two configurations can be discussed without ambiguity.

---

## The problem

Given a set of candidate facilities `F`, a set of customers `U`, a setup cost `c_f` for
opening facility `f`, and a service cost `d_uf` for serving customer `u` from facility
`f`, choose a non-empty subset `S ⊆ F` minimising

```
cost(S) = Σ_{f ∈ S} c_f  +  Σ_{u ∈ U} min_{f ∈ S} d_uf
```

Every customer is served by exactly one open facility and facilities have no capacity
limit. The problem is NP-hard.

The solver uses the **strong formulation** of the LP relaxation:

```
minimise    Σ_f c_f · y_f  +  Σ_u Σ_f d_uf · x_uf
subject to  Σ_f x_uf = 1            for every customer u
            x_uf ≤ y_f              for every (u, f)
            0 ≤ x, y ≤ 1
```

The `x_uf ≤ y_f` linking constraints (rather than the aggregated
`Σ_u x_uf ≤ |U| · y_f`) are what make this relaxation tight. That tightness is central
to every result below: it is why the heuristic layer has so little left to do on most
instance families.

At 2000×2000 this model has 4,002,000 variables and 4,002,000 constraint rows, of which
4,000,000 are linking constraints. Model size, not variable count, is the scaling
bottleneck.

---

## The method

### Pipeline

```
   ┌─────────────────────────────────────────────────────────────┐
   │  Phase 1 — LP relaxation                                    │
   │  SciPy linprog / HiGHS on the strong formulation            │
   │  → fractional y_f ∈ [0,1], and a valid lower bound          │
   └───────────────────────────┬─────────────────────────────────┘
                               │
   ┌───────────────────────────▼─────────────────────────────────┐
   │  Phase 2 — Seeding             LP-XLS/R:  y_f ≥ 0.5         │
   │                                LP-XLS/S:  p = max(y_f, ε)   │
   │  → initial open set S₀ (falls back to one facility if empty)│
   └───────────────────────────┬─────────────────────────────────┘
                               │
   ┌───────────────────────────▼─────────────────────────────────┐
   │  Phase 3 — Best-improvement local search                    │
   │  moves: insert f, delete f, swap (f_in, f_out)              │
   │  profit via EXACT O(1) deltas (save / loss / extra)         │
   │  → local optimum                                            │
   └─────────────────────────────────────────────────────────────┘
```

### Exact delta evaluation

Move profits use the `save` / `loss` / `extra` structures of Resende & Werneck. With
`φ₁(u)` and `φ₂(u)` the closest and second-closest open facilities to customer `u`:

```
save[f_i]        = −c_{f_i} + Σ_u max(0, d(u,φ₁(u)) − d(u,f_i))
loss[f_r]        = −c_{f_r} + Σ_{u: φ₁(u)=f_r} (d(u,φ₂(u)) − d(u,f_r))
extra[f_i,f_r]   = Σ_{u: φ₁(u)=f_r, d(u,f_i)<d(u,φ₂(u))}
                     (d(u,φ₂(u)) − d(u,f_i)) − max(0, d(u,f_r) − d(u,f_i))

profit(insert f_i)        =  save[f_i]
profit(delete f_r)        = −loss[f_r]
profit(swap f_i ← f_r)    =  save[f_i] − loss[f_r] + extra[f_i,f_r]
```

**These formulas are verified exact, not approximate.** Every one is cross-validated
against from-scratch recomputation over *every non-empty subset* of small instances —
Euclidean, non-Euclidean, and integer-valued with deliberate ties in `φ₁`/`φ₂` —
across roughly 400 instances with zero mismatches
(`tests/unit/test_delta_exactness.py`).

The source comments flag the `extra` correction as possibly incomplete. It is not: the
branch it skips contributes exactly zero, because `d(u,f_r) ≤ d(u,φ₂(u))` holds whenever
`φ₁(u) = f_r`.

### Complexity and a deliberate trade-off

`_apply_move_and_recompute` rebuilds the closest/second-closest assignments and the
auxiliary structures from scratch after **every** accepted move, rather than maintaining
them incrementally. Rebuilding `extra` costs `O(|F_closed| · |F_open| · |U|)` per
iteration and dominates runtime.

This is correctness-first by design: it removes any possibility of numerical drift
across iterations, at a substantial constant-factor cost. It is the main reason
wall-clock times here are not comparable with tuned implementations in the literature.

---

## Repository structure

```
scripts/
├── uflp_solver.py                 Canonical solver. Data structures, instance I/O,
│                                  LP relaxation, construction, local search,
│                                  exact delta formulas, CLI.
├── run_experiments.py             cap134 benchmark (20 seeds) + the 10-instance
│                                  LP-duality-gap correlation suite, with exact IP
│                                  optima from CBC as ground truth.
├── run_scaling_benchmark.py       Size sweep, 600 → 200,000 variables.
├── download_and_run_real_world.py California Housing benchmark; four arms, three
│                                  seeds; writes the report and a JSON sidecar.
├── koerkel_ghosh.py               Körkel-Ghosh instance generator (to published spec).
└── run_koerkel_ghosh.py           Körkel-Ghosh benchmark; five arms.

data/
└── cap134.txt                     OR-Library (Beasley) instance.

output/                            All generated artifacts. Never hand-edited.
├── california_4M_results.{md,json}
├── koerkel_ghosh_results.{md,json}
├── cap134.{md,png}
└── scaling_results.md, scaling_analysis.png

tests/
├── unit/                          10 files — invariants, exact deltas, LP correctness,
│                                  construction, report rendering.
└── regression/                    5 files — one per fixed defect, each proven to fail
                                   on the pre-fix code.
```

Entry points import one another as `scripts.<module>`, so run them as modules from the
repository root (`python -m scripts.uflp_solver`), not as files.

### Reports are generated, never written

Every figure in every results file is a measured value carried in a typed record
(`LPProfile`, `ArmResult`), and **every comparative phrase** — "lower", "faster",
"tighter", "earns its keep" — is *computed* from those values rather than written into
a template. A report therefore cannot state a conclusion its own table contradicts.

This is enforced, not merely intended. Two regression suites feed the renderers
mirror-image datasets and require the wording to flip, and scan the generator sources
for hardcoded comparative verdicts. This discipline exists because an earlier revision
of this project failed exactly that way: its template asserted *"LP-biased starts
lower"* unconditionally while the run it described produced the opposite, and the
published results file was subsequently overwritten by a script containing hardcoded
numeric literals.

---

## Datasets

| Family | Size | Source | LP fractionality | Purpose |
|---|---|---|---|---|
| **cap134** | 50×50 | OR-Library (Beasley) | **0** (integral) | Classical reference; LP bound *equals* the IP optimum |
| **Duality-gap suite** | 50×50 ×10 | Generated | 0 → 15/50 | Interpolates Euclidean (k=1) → random (k=10) to vary the duality gap |
| **Scaling sweep** | 20×30 → 250×800 | Generated Euclidean | low | Wall-clock scaling, 600 → 200,000 variables |
| **California Housing** | 2000×2000 | scikit-learn census blocks | **7 / 2000** (0.35%) | Real geographic coordinates and block populations |
| **Körkel-Ghosh** | 250×250 ×18 | Generated to published spec | **15–40%** | The standard *hard* family; deliberately weak LP |

### California Housing instance

Facilities and customers are two **disjoint** random samples of census blocks
(`np.random.seed(42)`). Setup cost is `5000 × (population / median population) + 1000`.
Service cost is Euclidean distance between (latitude, longitude) pairs, converted at
111 km/degree and priced at 10 per km.

> **Known approximation.** Longitude degrees are converted with the same 111 km factor
> as latitude degrees. At Californian latitudes (~37°N) a longitude degree is closer to
> 88 km, so east–west separation is overstated by roughly 26%. The instance is
> geographically *derived*, not geographically *accurate*. A unit test pins this so it
> cannot drift silently.

### Körkel-Ghosh instances

Generated to the specification quoted in Karapetyan & Goldengorin (arXiv:1711.06347):
fixed costs `U[100,200]` (class A), `U[1000,2000]` (B), `U[10000,20000]` (C);
allocation costs always `U[1000,2000]`; symmetric variants satisfy `c_ij = c_ji`.

> **These are generated-to-spec instances, not the official UflLib files.** The official
> archives were unreachable from the build environment (HTTP 403 from
> `resources.mpi-inf.mpg.de`; redirect loop from the Frankfurt mirror). The generator
> follows the published specification exactly, but the random draws differ, so
> **objective values here are not comparable with published Körkel-Ghosh results** and
> literature best-known bounds do not apply. Arm-versus-arm comparison, which is what
> this benchmark exists for, is unaffected.

---

## Results

Gaps are measured against the **LP bound** unless stated otherwise. On every family
except cap134 the LP bound is strictly below the integer optimum, so these figures
**overstate** the true optimality gap by an unknown amount.

### cap134 — the classical reference

| | LP-XLS/S | α-GRASP baseline |
|---|---|---|
| Optimal solutions found | **20 / 20** | **20 / 20** |
| Mean initial gap | 0.5180% | 1.9988% |
| Mean local-search iterations | 0.35 | 2.45 |
| Mean solve time | 30.81 ms | **10.96 ms** |

The LP bound equals the IP optimum exactly (928,941.75, duality gap 0.0000%). Both
methods find the optimum on every seed. **The LP-seeded method is ~3× slower for
identical quality**, because it pays for an LP solve that tells it nothing the baseline
could not reach on its own.

### Duality-gap suite — quality is not the issue, the ε floor is

As instances move from Euclidean (k=1) to random (k=10), the duality gap grows from
0.0000% to 9.5238% and fractional facilities from 0/50 to 15/50. Over the same range the
LP-XLS/S **initial** gap grows from 4.86% to 27.51%, while the α-GRASP initial gap stays
between roughly 1% and 7%.

This is not caused by fractionality. It is the `ε = 0.01` floor: every facility the LP
zeroes still gets a 1% chance of opening, so the expected number of spurious openings
grows linearly with `|F|`. The local search then spends its iterations deleting them.

### California Housing — 2000×2000

LP bound 397,600.48; LP solve 164.24 s; **7 of 2000** facilities fractional (99.65%
integral); `Σy = 57.50`; 61 facilities at `y ≥ 0.5`. Three seeds (42, 7, 2024).

| Arm | Final gap | LS iterations | End-to-end |
|---|---|---|---|
| A · LP rounding only | 1.0328% | 0 | 177.0 s |
| **A+ · LP-XLS/R** (deterministic) | **0.0421%** | **5** | 221.6 s |
| B · LP-XLS/S (randomized) | 0.0421% | 20.3 | 373.9 s |
| C · α-GRASP baseline | 0.1900% | 34.7 | 393.8 s |

LP-XLS/S reached the **identical solution on all three seeds** — the same 58 facilities
at 397,767.9731, from starting points whose gaps ranged 21.91% to 40.64%. The α-GRASP
baseline produced three different solutions spanning 0.3022 percentage points, and
matched that solution on one seed in three.

Local-search move mix for LP-XLS/S: **89% deletions, zero insertions**. The search is
not refining a good configuration; it is removing the ~15 facilities the ε floor opened
against the LP's advice.

### Körkel-Ghosh — where the LP is genuinely weak

18 instances at 250×250 (classes a/b/c × symmetric/asymmetric × 3), five seeds.
Mean fractionality 27%, mean duality gap 1.50%. Excess is percent above the best
solution any arm found on that instance.

| Arm | Mean excess |
|---|---|
| A · LP rounding only | 16.868% |
| **A+ · LP-XLS/R** (deterministic) | **0.045%** |
| B · LP-XLS/S (randomized) | 0.078% |
| C · α-GRASP baseline | 0.089% |
| D · Local search only, **no LP** | 0.115% |

By fixed-cost class:

| Class | LP fractional | Duality gap | Facilities at `y ≥ 0.5` | A+ | B | C | D |
|---|---|---|---|---|---|---|---|
| a (cheap) | 40% | 0.15% | 14.8 | 0.015% | 0.029% | 0.045% | 0.040% |
| b (medium) | 26% | 1.01% | 0.3 | 0.116% | 0.113% | 0.114% | 0.221% |
| c (expensive) | 15% | 3.35% | 0.0 | 0.005% | 0.091% | 0.108% | 0.083% |

On **10 of 18** instances, thresholding at `y ≥ 0.5` selects *no facility at all*. On
classes b and c the deterministic arm therefore starts from a single fallback facility
and inserts its way up — and still wins. Move mix across all instances: **zero
deletions, all insertions**, the exact inverse of California.

### Scaling — 600 to 200,000 variables

| Size | Variables | LP-XLS/S time | init / final gap | α-GRASP time | init / final gap | LP portion |
|---|---|---|---|---|---|---|
| 20×30 | 600 | 0.006 s | 0.00% / 0.0000% | 0.001 s | 1.86% / 0.0000% | 0.006 s |
| 50×100 | 5,000 | 0.030 s | 3.09% / 0.0000% | 0.014 s | 16.41% / 0.0000% | 0.027 s |
| 100×200 | 20,000 | 0.125 s | 10.51% / 0.0000% | 0.102 s | 19.56% / 0.1736% | 0.102 s |
| 150×400 | 60,000 | 0.408 s | 5.10% / 0.0000% | 0.580 s | 15.52% / 0.0627% | 0.324 s |
| 200×500 | 100,000 | 0.804 s | 7.94% / 0.0000% | 1.562 s | 17.26% / 0.0000% | 0.557 s |
| 250×800 | 200,000 | 1.908 s | 8.08% / 0.0000% | 4.158 s | 10.86% / 0.0000% | 1.115 s |

The LP-seeded arm is slower below ~20,000 variables and faster above it. At the largest
size it is 2.2× faster, with the LP solve accounting for 58% of its own runtime.

### Randomization: the ablation

`LP-XLS/R` and `LP-XLS/S` share the LP, the instance, and the local search. They differ
in exactly one respect — whether the construction is deterministic rounding or
probabilistic sampling — so any difference between them is attributable to
randomization and nothing else.

| | California 2000×2000 | Körkel-Ghosh, 18 instances |
|---|---|---|
| Final quality | **identical solution** (397,767.97 both) | R better on 13, S better on 4, tied on 1 |
| Mean excess | — | R 0.045% vs S 0.078% |
| LS iterations | 5 vs 20.3 (**4.1×**) | — |
| Heuristic time (excl. shared LP) | 57.4 s vs 209.7 s (**3.7×**) | — |

**On neither family does the randomized construction produce a better solution.** On
California it produces the same solution after four times the work. On Körkel-Ghosh —
the family chosen specifically because it is hard for LP-guided methods — the
deterministic variant wins outright.

The three-seed stability of `LP-XLS/S` on California, which looks like a strength in
isolation, has a deflationary explanation: the randomness never mattered. The
deterministic variant reaches that same solution with no seed at all.

---

## What the evidence supports

**The LP relaxation contributes.** Arm D — an identical local search seeded without ever
reading the relaxation — reaches 0.115% mean excess against 0.045% for the LP-seeded
arm, and loses in all three Körkel-Ghosh classes. The mechanism is narrower than
expected: on classes b and c, where thresholding selects nothing, the LP's *entire*
contribution is the choice of a single seed facility. That facility is **more expensive**
by immediate cost than the cost-greedy alternative in 12 of 12 cases, yet leads to a
better local optimum in 6 of them and ties in 5. The relaxation identifies structure a
greedy cost rule does not.

**Local search is doing the heavy lifting.** Rounding the LP and stopping yields 1.03%
on California and 16.87% on Körkel-Ghosh. Adding the search yields 0.04% and 0.045%.

**The exact delta formulas are correct**, including the `extra` correction the source
comments doubted.

**The randomized construction does not contribute.** Two instance families, opposite
regimes (89% deletions vs 100% insertions), same verdict.

**The method is not a GRASP**, and the `ε` floor is actively harmful at scale — its
damage grows linearly with `|F|`, from negligible at 50 facilities to ~19 spurious
openings at 2000.

---

## Limitations and threats to validity

- **Gaps are against the LP bound, not proven optima.** Only cap134 has a verified
  integer optimum (where the LP bound attains it). Elsewhere the reported gaps
  overstate true optimality gaps by an unknown amount. `solve_exact_ip` exists but CBC
  does not close instances at these sizes.
- **Körkel-Ghosh instances are generated to spec, not the official files.** Objective
  values are not comparable with published results.
- **The no-LP control (arm D) was run on Körkel-Ghosh only.** The California benchmark
  compares four arms, not five, so "does the LP contribute" is answered for one family.
- **Seed counts are small** — three on California, five on Körkel-Ghosh. Enough to show
  a spread, not enough for confidence intervals.
- **Timings are not portable.** Repeated runs of the same configuration on the same
  machine differed by tens of percent; the California LP solve alone ranged 67–231 s
  depending on memory pressure. Compare timings *within* a run only. Cost, gap,
  iteration and move-mix figures are fully deterministic under the stated seeds.
- **The local search is correctness-first, not fast.** Full recomputation after every
  move costs a large constant factor. Wall-clock comparisons against tuned literature
  implementations would not be meaningful.
- **A single α (0.2)** was used for the baseline; it was not tuned.
- **The HiGHS algorithm is not recorded.** `method='highs'` lets the solver choose
  between simplex and interior point, so LP solve time should not be labelled
  "simplex time".

---

## Reproducing

```bash
pip install -r requirements.txt

# Solve a single instance
python -m scripts.uflp_solver --file data/cap134.txt
python -m scripts.uflp_solver --n-facilities 25 --n-customers 40 --seed 7

# Benchmarks (run from the repository root)
python -m scripts.run_experiments          # cap134 + duality-gap suite   (~1 min)
python -m scripts.run_scaling_benchmark    # size sweep                   (~1 min)
python -m scripts.run_koerkel_ghosh        # hard family, 18 instances    (~2 min)
python -m scripts.download_and_run_real_world   # California 2000x2000    (~35 min, ~8 GB RAM)

# Re-render a report from its measurement sidecar, without re-running the benchmark
python -m scripts.download_and_run_real_world --from-json output/california_4M_results.json
```

Requires Python 3.11+ (developed on 3.14), SciPy, NumPy, scikit-learn, PuLP, matplotlib.

---

## Verification methodology

| Layer | Tool | Status |
|---|---|---|
| Unit tests and invariants | `pytest tests/unit` | **208 passing** |
| Line coverage | `pytest --cov` | **100%** |
| Mutation testing (core algorithm) | `mutmut run` | **92%** — gate is 85% |
| Mutation testing (whole module) | `mutmut run` | 76% — survivors concentrate in CLI print formatting |
| Regression tests | `pytest tests/regression` | **49**, each proven to fail on the pre-fix code |
| Static analysis | `sonar-scanner` | not run (requires a server) |

Each regression test carries a documented defect context, is verified RED against the
pre-fix code before being accepted, and includes a counterweight case asserting the fix
did not overshoot. Defects caught and fixed during this work included a worker-process
bug that silently degraded the LP-seeded arm into uniform random sampling under Python
3.14's `forkserver` default, a benchmark that pointed at a non-existent directory, a
missing-validation path that returned a wrong answer for customer-free instances, and
two report generators whose prose contradicted their own tables.

---

## References

- Resende, M. G. C. & Werneck, R. F. (2006). *A hybrid multistart heuristic for the
  uncapacitated facility location problem.* European Journal of Operational Research,
  174(1), 54–68. — source of the `save`/`loss`/`extra` delta formulation.
- Ghosh, D. (2003). *Neighborhood search heuristics for the uncapacitated facility
  location problem.* European Journal of Operational Research, 150(1), 150–162.
- Karapetyan, D. & Goldengorin, B. (2017). *Conditional Markov Chain Search for the
  Simple Plant Location Problem improves upper bounds on twelve Körkel-Ghosh instances.*
  [arXiv:1711.06347](https://arxiv.org/abs/1711.06347) — source of the Körkel-Ghosh
  generation specification used here.
- Beasley, J. E. *OR-Library.* — `cap134` instance.
- [UflLib](https://resources.mpi-inf.mpg.de/departments/d1/projects/benchmarks/UflLib/)
  — canonical home of the Körkel-Ghosh benchmark set.
