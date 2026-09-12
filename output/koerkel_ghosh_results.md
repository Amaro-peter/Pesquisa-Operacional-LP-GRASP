# Körkel-Ghosh Benchmark — Does the LP Bias Help Where the LP Is Weak?

Every other instance family in this repository has a near-integral LP relaxation, which
makes them a weak test of a method whose premise is that the *fractional* LP carries
information. This benchmark uses the standard hard family instead.

> **Provenance.** Every number below was measured by the run described in §6 and written
> directly by `render_report` in `run_koerkel_ghosh.py`. No value is hardcoded or
> back-computed, and every comparative word is derived from the measurements. Raw records
> are in `koerkel_ghosh_results.json`.

> **These are generated-to-spec instances, not the official UflLib files.** The official
> archives are served from hosts that refused every request from this environment (HTTP 403
> from `resources.mpi-inf.mpg.de`; a redirect loop from the Frankfurt mirror). The
> generator follows the published specification exactly (see `koerkel_ghosh.py`), but the
> random draws differ, so **objective values here are not comparable with published KG
> results** and literature best-known bounds do not apply. The arm-versus-arm comparison,
> which is what this benchmark exists for, is unaffected.

---

## 1. Why this family

| Property | Other families in this repo | Körkel-Ghosh |
|---|---|---|
| Allocation costs | Euclidean distances | uniform random [1,000, 2,000] |
| LP fractionality | 0.35% (California), 0% (cap134) | **27.0%** (mean here) |
| LP duality gap | 0% on cap134 (bound = optimum) | **1.50%** (mean, vs best found) |

Fixed-cost classes, per the published specification:

| Class | Fixed cost range | Effect |
|---|---|---|
| **a** | [100, 200] | many facilities open |
| **b** | [1,000, 2,000] | intermediate |
| **c** | [10,000, 20,000] | few facilities open |

## 2. Results

Instance size 250×250. Randomized arms run on 5 seeds (42, 7, 2024, 13, 99); values are means. Excess is percent above the best solution any arm found on that instance.

| Instance | LP fractional | LP gap | A · round | A+ · round+LS | B · LP-biased | C · α-GRASP | D · no LP |
|---|---|---|---|---|---|---|---|
| `gs250a-1` | 101/250 (40%) | 0.16% | 3.356% | 0.012% | 0.034% | 0.035% | 0.059% |
| `gs250a-2` | 107/250 (43%) | 0.16% | 2.868% | 0.028% | 0.037% | 0.018% | 0.031% |
| `gs250a-3` | 102/250 (41%) | 0.13% | 1.060% | 0.000% | 0.029% | 0.040% | 0.041% |
| `ga250a-1` | 90/250 (36%) | 0.13% | 2.048% | 0.000% | 0.032% | 0.051% | 0.006% |
| `ga250a-2` | 97/250 (39%) | 0.13% | 1.972% | 0.000% | 0.014% | 0.082% | 0.053% |
| `ga250a-3` | 109/250 (44%) | 0.16% | 3.026% | 0.049% | 0.027% | 0.046% | 0.053% |
| `gs250b-1` | 70/250 (28%) | 1.08% | 32.799% | 0.292% | 0.123% | 0.158% | 0.000% |
| `gs250b-2` | 70/250 (28%) | 1.14% | 33.553% | 0.080% | 0.090% | 0.114% | 0.371% |
| `gs250b-3` | 68/250 (27%) | 1.11% | 34.367% | 0.144% | 0.078% | 0.116% | 0.248% |
| `ga250b-1` | 57/250 (23%) | 0.88% | 34.298% | 0.000% | 0.129% | 0.061% | 0.426% |
| `ga250b-2` | 65/250 (26%) | 0.77% | 35.261% | 0.000% | 0.130% | 0.038% | 0.075% |
| `ga250b-3` | 63/250 (25%) | 1.09% | 35.306% | 0.179% | 0.124% | 0.201% | 0.206% |
| `gs250c-1` | 36/250 (14%) | 3.01% | 13.133% | 0.000% | 0.000% | 0.000% | 0.000% |
| `gs250c-2` | 36/250 (14%) | 3.32% | 14.483% | 0.029% | 0.075% | 0.172% | 0.029% |
| `gs250c-3` | 42/250 (17%) | 3.52% | 14.126% | 0.000% | 0.167% | 0.169% | 0.468% |
| `ga250c-1` | 32/250 (13%) | 3.42% | 13.716% | 0.000% | 0.071% | 0.200% | 0.000% |
| `ga250c-2` | 33/250 (13%) | 3.39% | 13.334% | 0.000% | 0.124% | 0.105% | 0.000% |
| `ga250c-3` | 39/250 (16%) | 3.47% | 14.922% | 0.000% | 0.112% | 0.000% | 0.000% |

| Arm | Mean excess over best found |
|---|---|
| A · LP rounding only | 16.868% |
| A+ · Rounding + local search | 0.045% |
| B · LP-biased GRASP | 0.078% |
| C · α-GRASP | 0.089% |
| D · Local search only (no LP) | 0.115% |

## 3. The ablation: does randomization help here?

On the California instance the randomized construction reached the identical solution
deterministic rounding did, so it contributed nothing. This family is the harder test.

Across 18 instances, arm B (randomized) beat arm A+
(deterministic) on **4**, lost on **13**, and tied on **1**.
Mean excess: **0.078%** for B against **0.045%** for A+.

**Randomization does not earn its place even here.** Arm A+ is better by 0.032 pp on average, so the California finding survives the move to
a family built to be hard for LP-guided methods.

Against the classical baseline, arm B beat arm C on **12** instances and lost on **5**.

## 4. Why rounding alone collapses here

| Instance | Σy | facilities with y ≥ 0.5 | A · round-only excess |
|---|---|---|---|
| `gs250a-1` | 29.1 | 13 | 3.36% |
| `gs250a-2` | 27.2 | 13 | 2.87% |
| `gs250a-3` | 28.7 | 20 | 1.06% |
| `ga250a-1` | 27.1 | 15 | 2.05% |
| `ga250a-2` | 27.2 | 15 | 1.97% |
| `ga250a-3` | 27.6 | 13 | 3.03% |
| `gs250b-1` | 9.3 | 0 | 32.80% |
| `gs250b-2` | 9.5 | 0 | 33.55% |
| `gs250b-3` | 9.7 | 0 | 34.37% |
| `ga250b-1` | 9.6 | 1 | 34.30% |
| `ga250b-2` | 9.7 | 1 | 35.26% |
| `ga250b-3` | 9.7 | 0 | 35.31% |
| `gs250c-1` | 3.3 | 0 | 13.13% |
| `gs250c-2` | 3.2 | 0 | 14.48% |
| `gs250c-3` | 3.3 | 0 | 14.13% |
| `ga250c-1` | 3.3 | 0 | 13.72% |
| `ga250c-2` | 3.3 | 0 | 13.33% |
| `ga250c-3` | 3.2 | 0 | 14.92% |

On **10** of 18 instances the relaxation puts *no* facility
at or above 0.5, so thresholding selects nothing at all and the construction falls back
to a single facility. The relaxation spreads its mass across many facilities rather than
concentrating it — which is exactly what a large duality gap looks like, and exactly the
condition under which a thresholding rule is the wrong way to read it.

## 5. Findings

**1. The relaxation really is weak here.** Mean fractionality 27.0% of facilities,
against 0.35% on the California instance; mean duality gap 1.50% against the
best solution found, where cap134's LP bound equalled its integer optimum exactly.

**2. The LP bias is not vindicated even here.** Arm A+ still matches or beats arm B
(0.045% against 0.078%), so the finding holds on the family specifically
chosen to give randomization its best chance.

**3. The LP does contribute here.** Arm A+ reaches 0.045% against
0.115% for arm D, which never reads the relaxation — so the LP is worth its
solve time on this family even though thresholding it is the wrong way to use it.

**4. Best arm overall: A+ · rounding + local search**, at 0.045% mean excess.

## 6. Reproduction

```bash
python run_koerkel_ghosh.py --size 250 --seeds 42 7 2024 13 99
```

| Run metadata | |
|---|---|
| Generated at | 2026-09-12 12:29 UTC |
| Generated by | `run_koerkel_ghosh.py` → `render_report` |
| Git commit | `f7fbf89 (working tree modified)` |
| Instances | 18 |
| Heuristic seeds | 42, 7, 2024, 13, 99 |
| Total wall-clock | 143.1 s |
| Python | 3.14.6 |
| NumPy / SciPy | 2.5.3 / 1.18.1 |

Raw measurements: [`koerkel_ghosh_results.json`](koerkel_ghosh_results.json).

