# Full Study — Hardness versus Provable Optima Across the Körkel-Ghosh Ladder

The same five arms run across every official Körkel-Ghosh size and then on the
2000×2000 California instance. The question this document answers is not only
*which arm wins*, but **where ground truth stops being available**, and whether the
ranking survives past that point.

| Arm | What it is | Reads the LP? | Randomized? |
|---|---|---|---|
| A | Round every `y ≥ 0.5`, no search | yes | no |
| A+ | A, then best-improvement local search | yes | no |
| D | Local search from the cheapest facility | **no** | no |
| **B · MS-LP-GRASP** | Multistart, construction sampled at `max(y, ε)` | yes | yes |
| C · α-GRASP | Multistart, savings-based RCL (α) | no | yes |

> **Provenance.** Every number was measured by the run described in §6 and written by
> `render_analysis` in `run_full_study.py`. No value is hardcoded; every comparative
> word is computed. Raw records: `full_study.json`, plus each stage's own sidecar.

---

## 1. Hardness versus provability

| Size | Instances | Restarts | LP fractional | Duality gap | Optima proven | Mean CBC time |
|---|---|---|---|---|---|---|
| 100×100 | 12 | 32 | 26.4% | 1.25% | **12 / 12** | 17.4 s |
| 150×150 | 6 | 32 | 26.6% | 1.19% | **6 / 6** | 254.9 s |
| 200×200 | 6 | 32 | 28.1% | 1.68% | **4 / 6** | 834.1 s |
| 250×250 | 12 | 32 | 26.5% | 3.01% | **1 / 12** | 1634.8 s |
| 500×500 | 6 | 32 | 23.3% | — | **0 / 6** | 2214.2 s |
| 750×750 | 6 | 32 | 22.3% | — | **0 / 6** | 3867.2 s |
| 2000×2000 (California) | 1 | 10 | 0.35% | — | **0 / 1** | not attempted |

Ground truth is available **in full up to 150×150**: every gap reported at that size and below is a true optimality gap.
At **200×200** CBC proved 4 of 6 optima within its budget; the rest are measured against the LP bound and therefore overstate the true gap.
At **250×250** CBC proved 1 of 12 optima within its budget; the rest are measured against the LP bound and therefore overstate the true gap.
At **500×500** CBC proved **none** within its budget — at this size the family is beyond exact solution on this hardware.
At **750×750** CBC proved **none** within its budget — at this size the family is beyond exact solution on this hardware.
On the 2000×2000 California instance the integer program is out of reach (not attempted): four million binary-linked assignment variables.

**This is the trade-off the ladder exists to expose.** Small instances give certainty and little difficulty; large ones give difficulty and no certainty. A result reported at one size only is reporting one half of it — which is precisely the criticism this run was built to answer.

## 2. Results, size by size

### Körkel-Ghosh 100×100

12 instances (classes a/b/c, symmetric and asymmetric), 32 restarts per
randomized arm.

> **Gaps here are true optimality gaps.** CBC proved all 12 integer optima (mean 17 s).

| Quantity | Value |
|---|---|
| Facilities × customers | 100 × 100 |
| Instances | 12 |
| Fractional facilities (mean) | **26.4%** |
| LP duality gap (mean, proven only) | **1.2547%** |
| Optima proven | 12 / 12 |
| Mean LP solve time | 0.11 s |
| Mean CBC exact-solve time | 17.4 s |

| Arm | Type | Mean optimality gap | Best | Worst | At reference | Mean time |
|---|---|---|---|---|---|---|
| A · LP rounding only | deterministic | 10.9032% | 0.2534% | 27.9188% | 0/12 | 0.1 s |
| A+ · LP rounding + local search | deterministic | 0.1129% | 0.0000% | 0.8713% | 6/12 | 0.1 s |
| D · Local search only (no LP) | deterministic | 0.0920% | 0.0000% | 0.3558% | 4/12 | 0.0 s |
| **B · MS-LP-GRASP (LP-biased)** | best of 32 | **0.0000%** | 0.0000% | 0.0000% | **12/12** | 0.9 s |
| **C · α-GRASP multistart** | best of 32 | **0.0030%** | 0.0000% | 0.0362% | **11/12** | 1.0 s |

Best mean gap at this size: **B · MS-LP-GRASP (LP-biased)** (0.0000%); worst: A · LP rounding only (10.9032%).

<details><summary>Per-instance detail</summary>

| Instance | Class | Reference | Proven | Fractional | A+ | D | **B** | C | B best at |
|---|---|---|---|---|---|---|---|---|---|
| `gs100a-1` | a/sym | 104,979 | yes | 19/100 | 0.0029% | 0.0619% | **0.0000%** | 0.0000% | 12/32 |
| `gs100a-2` | a/sym | 104,825 | yes | 37/100 | 0.0000% | 0.0820% | **0.0000%** | 0.0000% | 1/32 |
| `ga100a-1` | a/asym | 105,009 | yes | 34/100 | 0.0000% | 0.0638% | **0.0000%** | 0.0000% | 4/32 |
| `ga100a-2` | a/asym | 105,062 | yes | 47/100 | 0.0771% | 0.0276% | **0.0000%** | 0.0362% | 20/32 |
| `gs100b-1` | b/sym | 115,947 | yes | 32/100 | 0.0681% | 0.0000% | **0.0000%** | 0.0000% | 2/32 |
| `gs100b-2` | b/sym | 117,197 | yes | 35/100 | 0.0000% | 0.3558% | **0.0000%** | 0.0000% | 7/32 |
| `ga100b-1` | b/asym | 116,173 | yes | 26/100 | 0.0224% | 0.2987% | **0.0000%** | 0.0000% | 3/32 |
| `ga100b-2` | b/asym | 116,818 | yes | 33/100 | 0.3133% | 0.1986% | **0.0000%** | 0.0000% | 22/32 |
| `gs100c-1` | c/sym | 149,436 | yes | 17/100 | 0.8713% | 0.0000% | **0.0000%** | 0.0000% | 1/32 |
| `gs100c-2` | c/sym | 149,991 | yes | 10/100 | 0.0000% | 0.0153% | **0.0000%** | 0.0000% | 1/32 |
| `ga100c-1` | c/asym | 149,495 | yes | 13/100 | 0.0000% | 0.0000% | **0.0000%** | 0.0000% | 1/32 |
| `ga100c-2` | c/asym | 151,787 | yes | 14/100 | 0.0000% | 0.0000% | **0.0000%** | 0.0000% | 7/32 |

</details>

### Körkel-Ghosh 150×150

6 instances (classes a/b/c, symmetric and asymmetric), 32 restarts per
randomized arm.

> **Gaps here are true optimality gaps.** CBC proved all 6 integer optima (mean 255 s).

| Quantity | Value |
|---|---|
| Facilities × customers | 150 × 150 |
| Instances | 6 |
| Fractional facilities (mean) | **26.6%** |
| LP duality gap (mean, proven only) | **1.1868%** |
| Optima proven | 6 / 6 |
| Mean LP solve time | 0.39 s |
| Mean CBC exact-solve time | 254.9 s |

| Arm | Type | Mean optimality gap | Best | Worst | At reference | Mean time |
|---|---|---|---|---|---|---|
| A · LP rounding only | deterministic | 10.8556% | 0.8878% | 28.7025% | 0/6 | 0.4 s |
| A+ · LP rounding + local search | deterministic | 0.0460% | 0.0000% | 0.1979% | 3/6 | 0.5 s |
| D · Local search only (no LP) | deterministic | 0.1510% | 0.0000% | 0.3331% | 2/6 | 0.1 s |
| **B · MS-LP-GRASP (LP-biased)** | best of 32 | **0.0000%** | 0.0000% | 0.0000% | **6/6** | 3.0 s |
| **C · α-GRASP multistart** | best of 32 | **0.0000%** | 0.0000% | 0.0000% | **6/6** | 2.9 s |

Best mean gap at this size: **B · MS-LP-GRASP (LP-biased)** (0.0000%); worst: A · LP rounding only (10.8556%).

<details><summary>Per-instance detail</summary>

| Instance | Class | Reference | Proven | Fractional | A+ | D | **B** | C | B best at |
|---|---|---|---|---|---|---|---|---|---|
| `gs150a-1` | a/sym | 156,337 | yes | 59/150 | 0.0352% | 0.1452% | **0.0000%** | 0.0000% | 4/32 |
| `ga150a-1` | a/asym | 156,226 | yes | 63/150 | 0.0429% | 0.1223% | **0.0000%** | 0.0000% | 13/32 |
| `gs150b-1` | b/sym | 169,309 | yes | 36/150 | 0.1979% | 0.3331% | **0.0000%** | 0.0000% | 4/32 |
| `ga150b-1` | b/asym | 170,849 | yes | 43/150 | 0.0000% | 0.3055% | **0.0000%** | 0.0000% | 2/32 |
| `gs150c-1` | c/sym | 210,149 | yes | 17/150 | 0.0000% | 0.0000% | **0.0000%** | 0.0000% | 1/32 |
| `ga150c-1` | c/asym | 211,338 | yes | 21/150 | 0.0000% | 0.0000% | **0.0000%** | 0.0000% | 3/32 |

</details>

### Körkel-Ghosh 200×200

6 instances (classes a/b/c, symmetric and asymmetric), 32 restarts per
randomized arm.

> **Mixed reference.** CBC proved 4 of 6 optima; the remaining 2 are measured against the LP bound and therefore **overstate** the true optimality gap.

| Quantity | Value |
|---|---|
| Facilities × customers | 200 × 200 |
| Instances | 6 |
| Fractional facilities (mean) | **28.1%** |
| LP duality gap (mean, proven only) | **1.6777%** |
| Optima proven | 4 / 6 |
| Mean LP solve time | 1.01 s |
| Mean CBC exact-solve time | 834.1 s |

| Arm | Type | Mean gap vs reference | Best | Worst | At reference | Mean time |
|---|---|---|---|---|---|---|
| A · LP rounding only | deterministic | 13.5655% | 1.7409% | 34.4660% | 0/6 | 1.0 s |
| A+ · LP rounding + local search | deterministic | 0.4786% | 0.0128% | 1.4319% | 0/6 | 1.2 s |
| D · Local search only (no LP) | deterministic | 0.4865% | 0.0000% | 1.5654% | 1/6 | 0.3 s |
| **B · MS-LP-GRASP (LP-biased)** | best of 32 | **0.3625%** | 0.0000% | 1.1029% | **2/6** | 7.6 s |
| **C · α-GRASP multistart** | best of 32 | **0.3667%** | 0.0000% | 1.1029% | **2/6** | 7.2 s |

Best mean gap at this size: **B · MS-LP-GRASP (LP-biased)** (0.3625%); worst: A · LP rounding only (13.5655%).

<details><summary>Per-instance detail</summary>

| Instance | Class | Reference | Proven | Fractional | A+ | D | **B** | C | B best at |
|---|---|---|---|---|---|---|---|---|---|
| `gs200a-1` | a/sym | 207,136 | yes | 86/200 | 0.0492% | 0.0425% | **0.0034%** | 0.0058% | 13/32 |
| `ga200a-1` | a/asym | 207,224 | yes | 89/200 | 0.0878% | 0.1062% | **0.0140%** | 0.0372% | 30/32 |
| `gs200b-1` | b/sym | 220,956 | no (not proven (Solution Found)) | 53/200 | 1.4319% | 1.5654% | **1.1029%** | 1.1029% | 3/32 |
| `ga200b-1` | b/asym | 221,158 | no (not proven (Solution Found)) | 49/200 | 1.2341% | 1.1920% | **1.0546%** | 1.0546% | 3/32 |
| `gs200c-1` | c/sym | 273,768 | yes | 29/200 | 0.0555% | 0.0000% | **0.0000%** | 0.0000% | 1/32 |
| `ga200c-1` | c/asym | 273,713 | yes | 31/200 | 0.0128% | 0.0128% | **0.0000%** | 0.0000% | 2/32 |

</details>

### Körkel-Ghosh 250×250

12 instances (classes a/b/c, symmetric and asymmetric), 32 restarts per
randomized arm.

> **Mixed reference.** CBC proved 1 of 12 optima; the remaining 11 are measured against the LP bound and therefore **overstate** the true optimality gap.

| Quantity | Value |
|---|---|
| Facilities × customers | 250 × 250 |
| Instances | 12 |
| Fractional facilities (mean) | **26.5%** |
| LP duality gap (mean, proven only) | **3.0059%** |
| Optima proven | 1 / 12 |
| Mean LP solve time | 2.21 s |
| Mean CBC exact-solve time | 1634.8 s |

| Arm | Type | Mean gap vs reference | Best | Worst | At reference | Mean time |
|---|---|---|---|---|---|---|
| A · LP rounding only | deterministic | 18.1775% | 2.1049% | 36.3062% | 0/12 | 2.2 s |
| A+ · LP rounding + local search | deterministic | 1.2522% | 0.0000% | 3.4162% | 1/12 | 2.6 s |
| D · Local search only (no LP) | deterministic | 1.3034% | 0.0000% | 3.4162% | 1/12 | 0.6 s |
| **B · MS-LP-GRASP (LP-biased)** | best of 32 | **1.2030%** | 0.0000% | 3.4162% | **1/12** | 15.4 s |
| **C · α-GRASP multistart** | best of 32 | **1.2053%** | 0.0000% | 3.4162% | **1/12** | 13.0 s |

Best mean gap at this size: **B · MS-LP-GRASP (LP-biased)** (1.2030%); worst: A · LP rounding only (18.1775%).

<details><summary>Per-instance detail</summary>

| Instance | Class | Reference | Proven | Fractional | A+ | D | **B** | C | B best at |
|---|---|---|---|---|---|---|---|---|---|
| `gs250a-1` | a/sym | 257,536 | no (not proven (Solution Found)) | 101/250 | 0.1730% | 0.2196% | **0.1447%** | 0.1486% | 8/32 |
| `gs250a-2` | a/sym | 257,489 | no (not proven (Solution Found)) | 107/250 | 0.1914% | 0.1945% | **0.1611%** | 0.1440% | 15/32 |
| `ga250a-1` | a/asym | 257,202 | no (not proven (Solution Found)) | 90/250 | 0.1348% | 0.1410% | **0.1313%** | 0.1306% | 20/32 |
| `ga250a-2` | a/asym | 257,430 | no (not proven (Solution Found)) | 97/250 | 0.1304% | 0.1836% | **0.1304%** | 0.1304% | 7/32 |
| `gs250b-1` | b/sym | 274,234 | no (not proven (Solution Found)) | 70/250 | 1.3734% | 1.0784% | **1.0372%** | 1.0784% | 5/32 |
| `gs250b-2` | b/sym | 273,354 | no (not proven (Solution Found)) | 70/250 | 1.2211% | 1.5163% | **1.0590%** | 1.0590% | 23/32 |
| `ga250b-1` | b/asym | 272,991 | no (not proven (Solution Found)) | 57/250 | 0.8774% | 1.3074% | **0.8774%** | 0.8774% | 10/32 |
| `ga250b-2` | b/asym | 273,596 | no (not proven (Solution Found)) | 65/250 | 0.7727% | 0.8480% | **0.7727%** | 0.7727% | 2/32 |
| `gs250c-1` | c/sym | 331,825 | yes | 36/250 | 0.0000% | 0.0000% | **0.0000%** | 0.0000% | 1/32 |
| `gs250c-2` | c/sym | 321,472 | no (not proven (Solution Found)) | 36/250 | 3.3449% | 3.3449% | **3.3153%** | 3.3153% | 9/32 |
| `ga250c-1` | c/asym | 322,030 | no (not proven (Solution Found)) | 32/250 | 3.4162% | 3.4162% | **3.4162%** | 3.4162% | 6/32 |
| `ga250c-2` | c/asym | 322,261 | no (not proven (Solution Found)) | 33/250 | 3.3912% | 3.3912% | **3.3912%** | 3.3912% | 17/32 |

</details>

### Körkel-Ghosh 500×500

6 instances (classes a/b/c, symmetric and asymmetric), 32 restarts per
randomized arm.

> **No optimum proven** at this size within the budget. All 6 gaps are measured against the LP bound and **overstate** the true optimality gap.

| Quantity | Value |
|---|---|
| Facilities × customers | 500 × 500 |
| Instances | 6 |
| Fractional facilities (mean) | **23.3%** |
| LP duality gap | not computable — no optimum proven |
| Optima proven | 0 / 6 |
| Mean LP solve time | 28.28 s |
| Mean CBC exact-solve time | 2214.2 s |

| Arm | Type | Mean gap vs reference | Best | Worst | At reference | Mean time |
|---|---|---|---|---|---|---|
| A · LP rounding only | deterministic | 23.2521% | 3.4688% | 40.2995% | 0/6 | 28.4 s |
| A+ · LP rounding + local search | deterministic | 1.5099% | 0.1244% | 3.4804% | 0/6 | 34.5 s |
| D · Local search only (no LP) | deterministic | 1.5532% | 0.1507% | 3.7146% | 0/6 | 7.6 s |
| **B · MS-LP-GRASP (LP-biased)** | best of 32 | **1.4173%** | 0.1262% | 3.3895% | **0/6** | 182.5 s |
| **C · α-GRASP multistart** | best of 32 | **1.4180%** | 0.1217% | 3.4038% | **0/6** | 156.9 s |

Best mean gap at this size: **B · MS-LP-GRASP (LP-biased)** (1.4173%); worst: A · LP rounding only (23.2521%).

<details><summary>Per-instance detail</summary>

| Instance | Class | Reference | Proven | Fractional | A+ | D | **B** | C | B best at |
|---|---|---|---|---|---|---|---|---|---|
| `gs500a-1` | a/sym | 510,177 | no (not proven (Solution Found)) | 162/500 | 0.1244% | 0.1507% | **0.1262%** | 0.1217% | 5/32 |
| `ga500a-1` | a/asym | 510,281 | no (not proven (Solution Found)) | 180/500 | 0.1709% | 0.1991% | **0.1568%** | 0.1633% | 25/32 |
| `gs500b-1` | b/sym | 532,709 | no (not proven (Solution Found)) | 114/500 | 0.9217% | 1.0090% | **0.9221%** | 0.9136% | 10/32 |
| `ga500b-1` | b/asym | 532,626 | no (not proven (Solution Found)) | 111/500 | 1.1814% | 0.9221% | **0.8842%** | 0.8808% | 21/32 |
| `gs500c-1` | c/sym | 602,919 | no (not proven (Solution Found)) | 64/500 | 3.1808% | 3.3234% | **3.0249%** | 3.0249% | 2/32 |
| `ga500c-1` | c/asym | 602,629 | no (not proven (Solution Found)) | 69/500 | 3.4804% | 3.7146% | **3.3895%** | 3.4038% | 24/32 |

</details>

### Körkel-Ghosh 750×750

6 instances (classes a/b/c, symmetric and asymmetric), 32 restarts per
randomized arm.

> **No optimum proven** at this size within the budget. All 6 gaps are measured against the LP bound and **overstate** the true optimality gap.

| Quantity | Value |
|---|---|
| Facilities × customers | 750 × 750 |
| Instances | 6 |
| Fractional facilities (mean) | **22.3%** |
| LP duality gap | not computable — no optimum proven |
| Optima proven | 0 / 6 |
| Mean LP solve time | 205.58 s |
| Mean CBC exact-solve time | 3867.2 s |

| Arm | Type | Mean gap vs reference | Best | Worst | At reference | Mean time |
|---|---|---|---|---|---|---|
| A · LP rounding only | deterministic | 29.8614% | 17.2851% | 43.0174% | 0/6 | 206.3 s |
| A+ · LP rounding + local search | deterministic | 1.3425% | 0.2014% | 2.9514% | 0/6 | 230.2 s |
| D · Local search only (no LP) | deterministic | 1.3836% | 0.1874% | 3.2589% | 0/6 | 26.5 s |
| **B · MS-LP-GRASP (LP-biased)** | best of 32 | **1.2386%** | 0.1581% | 2.8574% | **0/6** | 834.0 s |
| **C · α-GRASP multistart** | best of 32 | **1.2416%** | 0.1620% | 2.8574% | **0/6** | 603.7 s |

Best mean gap at this size: **B · MS-LP-GRASP (LP-biased)** (1.2386%); worst: A · LP rounding only (29.8614%).

<details><summary>Per-instance detail</summary>

| Instance | Class | Reference | Proven | Fractional | A+ | D | **B** | C | B best at |
|---|---|---|---|---|---|---|---|---|---|
| `gs750a-1` | a/sym | 762,586 | no (not proven (Solution Found)) | 257/750 | 0.2014% | 0.2022% | **0.1581%** | 0.1812% | 11/32 |
| `ga750a-1` | a/asym | 762,606 | no (not proven (Solution Found)) | 274/750 | 0.2253% | 0.1874% | **0.1604%** | 0.1620% | 1/32 |
| `gs750b-1` | b/sym | 789,809 | no (not proven (Solution Found)) | 155/750 | 0.8852% | 0.8817% | **0.7729%** | 0.7729% | 29/32 |
| `ga750b-1` | b/asym | 789,811 | no (not proven (Solution Found)) | 153/750 | 0.9200% | 0.9758% | **0.8427%** | 0.8363% | 8/32 |
| `gs750c-1` | c/sym | 874,815 | no (not proven (Solution Found)) | 80/750 | 2.9514% | 3.2589% | **2.8574%** | 2.8574% | 4/32 |
| `ga750c-1` | c/asym | 874,968 | no (not proven (Solution Found)) | 85/750 | 2.8717% | 2.7954% | **2.6398%** | 2.6398% | 1/32 |

</details>

### California Housing 2000×2000

Block-group centroids from the California Housing dataset — a real geographic
instance with 4,000,000 service-cost entries, 10 restarts.

> **The integer optimum is not available** (not attempted). All gaps are measured
> against the LP bound and therefore **overstate** the true optimality gap by an
> unknown amount. The ranking between arms is still valid — every arm is measured
> against the same reference.

| Quantity | Value |
|---|---|
| Facilities × customers | 2000 × 2000 |
| LP bound | 397,600.48 |
| Reference value | 397,600.48 (not attempted) |
| Fractional facilities | **0.35%** |
| LP solve time | 125.9 s |

| Arm | Type | gap vs LP bound | Time |
|---|---|---|---|
| A · LP rounding only | deterministic | 1.0328% | 133.6 s |
| A+ · LP rounding + local search | deterministic | 0.0421% | 170.5 s |
| D · Local search only (no LP) | deterministic | 0.0421% | 422.1 s |
| **B · MS-LP-GRASP (LP-biased)** | best of 10 | **0.0358%** | 2098.7 s |
| **C · α-GRASP multistart** | best of 10 | **0.0358%** | 3627.6 s |

B reached its best at restart **2 of 10**; C at restart **4 of 10**.

B's **worst** restart is 0.0421%; C's **mean** restart is 0.1618% — B ahead even at its worst.

## 3. Does the ranking survive as instances get harder?

| Size | A+ (det.) | D (no LP) | **B · MS-LP-GRASP** | C · α-GRASP | B at reference | C at reference |
|---|---|---|---|---|---|---|
| 100×100 | 0.1129% | 0.0920% | **0.0000%** | 0.0030% | 12/12 | 11/12 |
| 150×150 | 0.0460% | 0.1510% | **0.0000%** | 0.0000% | 6/6 | 6/6 |
| 200×200 | 0.4786% | 0.4865% | **0.3625%** | 0.3667% | 2/6 | 2/6 |
| 250×250 | 1.2522% | 1.3034% | **1.2030%** | 1.2053% | 1/12 | 1/12 |
| 500×500 | 1.5099% | 1.5532% | **1.4173%** | 1.4180% | 0/6 | 0/6 |
| 750×750 | 1.3425% | 1.3836% | **1.2386%** | 1.2416% | 0/6 | 0/6 |
| 2000×2000 | 0.0421% | 0.0421% | **0.0358%** | 0.0358% | 0/1 | 0/1 |

Head-to-head on the Körkel-Ghosh ladder, per size:

| Size | B beats C | C beats B | B beats A+ | A+ beats B | Median winning restart (B) |
|---|---|---|---|---|---|
| 100×100 | 1 | 0 | 6 | 0 | 4 / 32 |
| 150×150 | 0 | 0 | 3 | 0 | 4 / 32 |
| 200×200 | 2 | 0 | 6 | 0 | 3 / 32 |
| 250×250 | 2 | 2 | 6 | 0 | 9 / 32 |
| 500×500 | 2 | 3 | 4 | 2 | 21 / 32 |
| 750×750 | 2 | 1 | 6 | 0 | 8 / 32 |

Across all 48 Körkel-Ghosh instances, **B beats C on 9, loses on 6, ties on 33**.
But those wins are **concentrated on the smaller instances**: B leads 3–0 on the smallest 3 sizes (100, 150, 200) and only 6–6 on the largest 3 (250, 500, 750). On this evidence the advantage **does not persist** at the sizes the family exists to test.

The B-versus-C margin **holds steady** with size: +0.0030 pp at 100×100, +0.0030 pp at 750×750.

## 4. What the LP contributes, by size

Arm D never reads the relaxation; A+ is the *same* local search seeded from it. The
difference between the two columns is the LP's entire contribution, isolated from
randomization.

| Size | A+ (LP-seeded) | D (no LP) | LP advantage |
|---|---|---|---|
| 100×100 | 0.1129% | 0.0920% | -0.0209 pp |
| 150×150 | 0.0460% | 0.1510% | +0.1050 pp |
| 200×200 | 0.4786% | 0.4865% | +0.0079 pp |
| 250×250 | 1.2522% | 1.3034% | +0.0512 pp |
| 500×500 | 1.5099% | 1.5532% | +0.0432 pp |
| 750×750 | 1.3425% | 1.3836% | +0.0411 pp |
| 2000×2000 | 0.0421% | 0.0421% | +0.0000 pp |

The LP advantage is **not consistent across sizes**; see the per-size figures above. Reporting a single headline number for it would misrepresent the evidence.

## 5. What the ladder cost

| Stage | Instances | Restarts | Wall-clock |
|---|---|---|---|
| Körkel-Ghosh 100×100 | 12 | 32 | 3.9 min |
| Körkel-Ghosh 150×150 | 6 | 32 | 26.1 min |
| Körkel-Ghosh 200×200 | 6 | 32 | 85.0 min |
| Körkel-Ghosh 250×250 | 12 | 32 | 332.9 min |
| Körkel-Ghosh 500×500 | 6 | 32 | 257.0 min |
| Körkel-Ghosh 750×750 | 6 | 32 | 536.4 min |
| California 2000×2000 | 1 | 10 | 103.5 min |
| **Total** | | | **0.0 min** |

**The restart budget is the same at every size** (best of 32), so the quality columns above are directly comparable across the ladder: a difference between two rows is a difference in difficulty, not in how much search each row was given.

## 6. Reproduction

```bash
python -m scripts.run_full_study
```

| Stage | Instances | Restarts | CBC budget |
|---|---|---|---|
| KG 100×100 | 12 | 32 | 1800 s |
| KG 150×150 | 6 | 32 | 1800 s |
| KG 200×200 | 6 | 32 | 1800 s |
| KG 250×250 | 12 | 32 | 1800 s |
| KG 500×500 | 6 | 32 | 1800 s |
| KG 750×750 | 6 | 32 | 1800 s |
| California 2000×2000 | 1 | 10 | not attempted |

| Run metadata | |
|---|---|
| Generated at | 2026-09-14 21:44 UTC |
| Generated by | `run_full_study.py` → `render_analysis` |
| Git commit | `ae61a4e (working tree modified)` |
| Total wall-clock | 0.0 min |
| Python | 3.14.6 |

Per-stage reports and raw data:
- `output/kg100/koerkel_ghosh_results.md` · `output/kg100/koerkel_ghosh_results.json`
- `output/kg150/koerkel_ghosh_results.md` · `output/kg150/koerkel_ghosh_results.json`
- `output/kg200/koerkel_ghosh_results.md` · `output/kg200/koerkel_ghosh_results.json`
- `output/kg250/koerkel_ghosh_results.md` · `output/kg250/koerkel_ghosh_results.json`
- `output/kg500/koerkel_ghosh_results.md` · `output/kg500/koerkel_ghosh_results.json`
- `output/kg750/koerkel_ghosh_results.md` · `output/kg750/koerkel_ghosh_results.json`
- `output/california_4M_results.md` · `output/california_4M_results.json`
- `output/full_study.md` · `output/full_study.json`

