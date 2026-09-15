# Real-World UFLP Benchmark — California Housing, 2000×2000

A 2,000×2,000 instance built from real census-block coordinates and
populations. Randomized arms run as **multistart** procedures; deterministic arms run
once, because restarting a deterministic construction reproduces the same solution.

> **What the gaps are measured against.** LP bound 397,600.48 (integer optimum not proven: CBC not attempted after 0.0s); gaps below are measured against a lower bound and therefore OVERSTATE the true optimality gap.
> The true optimum lies in `[397,600.48, 397,742.77]`.

> **Provenance.** Every number was measured by the run described in §8 and written by
> `render_report` in `download_and_run_real_world.py`. No value is hardcoded or
> back-computed; every comparative word is computed from the measurements. Raw records
> are in `california_4M_results.json`; `--from-json` re-renders the prose from them
> without re-running the benchmark.

---

## 1. Instance

| Property | Value |
|---|---|
| Source | California Housing (`sklearn.datasets.fetch_california_housing`) |
| Facilities × customers | 2,000 × 2,000 |
| Sampling | two **disjoint** random samples of census blocks, `np.random.seed(42)` |
| Setup cost | `5,000 × (population / median population) + 1,000` |
| Service cost | Euclidean degree distance × 111.0 km/deg × 10.0 per km |
| LP model | 4,002,000 variables, 4,002,000 rows (4,000,000 linking) |

Longitude degrees are converted at the same 111.0 km/degree as latitude degrees.
At ~37°N a longitude degree is closer to 88 km, so east–west separation is overstated by
roughly 26%. The instance is geographically *derived*, not geographically *accurate*.

## 2. LP relaxation

| Quantity | Value |
|---|---|
| LP bound | **397,600.48** |
| Solve time | 125.95 s |
| Fractional facilities (`0.01 < y < 0.99`) | **7** / 2,000 (0.35%) |
| `Σ y[f]` | 57.50 |
| Facilities at `y ≥ 0.5` | 61 |

The relaxation is **essentially integral** — 7 of 2000 facilities
(0.35%) take a strictly fractional value. A near-integral relaxation
means the LP has very nearly solved the problem already, leaving the heuristic layer
little to contribute beyond repair.

## 3. Arms

| | Arm | Type | Construction |
|---|---|---|---|
| A | LP rounding only | deterministic | open every `y ≥ 0.5`, no local search |
| A+ | LP rounding + local search | deterministic | same, then local search |
| D | Local search only | deterministic | cheapest facility, **never reads the LP** |
| B | LP-biased multistart | best of 10 | sample `p = max(y, 0.01)` per restart |
| C | α-GRASP multistart | best of 10 | savings RCL, α = 0.2, per restart |

A+ and B differ in exactly one respect — deterministic rounding versus probabilistic
sampling — so any difference between them is attributable to randomization alone.
D isolates what the LP contributes at all.

## 4. Results

| Arm | Type | Final cost | gap vs LP bound | Work | Time |
|---|---|---|---|---|---|
| A · LP rounding | deterministic | 401,706.88 | 1.0328% | 0 iters | 133.6 s |
| **A+ · rounding + LS** | deterministic | **397,767.97** | **0.0421%** | 5 iters | 170.5 s |
| D · local search, no LP | deterministic | 397,767.97 | 0.0421% | 73 iters | 422.1 s |
| **B · LP-biased multistart** | best of 10 | **397,742.77** | **0.0358%** | 237 iters / 10 restarts | 2098.7 s |
| **C · α-GRASP multistart** | best of 10 | **397,742.77** | **0.0358%** | 335 iters / 10 restarts | 3627.6 s |

*A, A+ and B include the 125.9 s LP solve; C and D do not read the LP.*

**Best arm: B · LP-biased multistart, at 0.0358%.**

## 5. LP-biased GRASP vs. classical GRASP

Both arms get 10 restarts and the same local search.

| | B · LP-biased | C · α-GRASP |
|---|---|---|
| Best gap vs LP bound | **0.0358%** | **0.0358%** |
| Restart producing the winner | 2 / 10 | 4 / 10 |
| Mean gap per restart | 0.0409% | 0.1618% |
| Best single restart | 0.0358% | 0.0358% |
| Worst single restart | 0.0421% | 0.5409% |
| Total local-search iterations | 237 | 335 |
| Heuristic time (excl. LP) | 1972.8 s | 3627.6 s |

**The two constructions tie at this restart budget.**

## 6. Does multistart pay for itself?

A best-of-10 result costs 10× the work of one run. The question is therefore not
whether B beats A+ at full budget, but how many restarts B needs to match A+ at all.

- B reaches A+'s quality (0.0421%) after **1** restart, at roughly 323 s against A+'s 171 s.
- B's best came from restart 2 of 10; the remaining 8 produced no improvement.

**Multistart overturns the single-start result.** With 10 restarts the randomized
construction reaches 0.0358% against the deterministic 0.0421%, so the earlier
finding that randomization contributes nothing was an artifact of running it once.

## 7. What the LP contributes

Arm D never reads the relaxation. It reaches **0.0421%**; the LP-guided A+ reaches
**0.0421%**.

**The LP makes no measurable difference here** — the LP-free arm reaches the same
quality without paying 126 s for the relaxation.

Rounding the LP without searching reaches only 1.0328%, so the local search is doing
substantial work regardless of how its starting point is chosen.

## 8. Limitations

- **Gaps are measured against the LP bound, not a proven optimum.** At this size the
  integer program has four million binary-linked assignment variables and is out of
  reach for CBC, so the figures above **overstate** true optimality gaps by an unknown
  amount. The Körkel-Ghosh benchmark in this repository reports proven optima instead.
- **10 restarts** is a modest budget for a GRASP; published studies commonly use 32+.
- **One instance.** Conclusions here describe this instance, not the family.
- **Timings are not portable.** Repeated runs of the same configuration on this machine
  differed by tens of percent; the LP solve alone has ranged 67–231 s. Compare timings
  within a run only. Cost, gap, iteration and move-mix figures are deterministic under
  the stated seeds.
- **The distance model overstates east–west separation by ~26%** (§1).
- **A single α (0.2)** for the baseline; not tuned.

## 9. Reproduction

```bash
python -m scripts.download_and_run_real_world --size 2000 --restarts 10 --alpha 0.2
python -m scripts.download_and_run_real_world --from-json output/california_4M_results.json
```

| Run metadata | |
|---|---|
| Generated at | 2026-09-14 08:58 UTC |
| Generated by | `download_and_run_real_world.py` → `render_report` |
| Git commit | `ae61a4e (working tree modified)` |
| Dataset sample seed | 42 |
| Restarts | 10 |
| Total wall-clock | 6208.0 s |
| Python | 3.14.6 |
| NumPy / SciPy / scikit-learn | 2.5.3 / 1.18.1 / 1.9.1 |

Raw measurements: [`california_4M_results.json`](california_4M_results.json).

