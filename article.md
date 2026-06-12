# A hybrid multistart heuristic for the uncapacitated facility location problem

**Mauricio G.C. Resende**, **Renato F. Werneck** *European Journal of Operational Research 174 (2006) 54-68* *Internet and Networks Systems Research Center, AT&T Labs Research* *Department of Computer Science, Princeton University*

---

## Abstract
We present a multistart heuristic for the uncapacitated facility location problem, based on a very successful method we originally developed for the p-median problem. We show extensive empirical evidence to the effectiveness of our algorithm in practice. For most benchmarks instances in the literature, we obtain solutions that are either optimal or a fraction of a percentage point away from it. Even for pathological instances (created with the sole purpose of being hard to tackle), our algorithm can get very close to optimality if given enough time. It consistently outperforms other heuristics in the literature.

**Keywords:** Facility location; p-median problem; Heuristic; Local search; GRASP: Path-relinking

---

## 1. Introduction
Consider a set F of potential facilities, each with a setup cost $c(f)$, and let U be a set of users (or customers) that must be served by these facilities. The cost of serving user u with facility f is given by the distance $d(u,f)$ between them. The facility location problem consists in determining a set $S \subset F$ of facilities to open so as to minimize the total cost (including setup and service) of covering all customers:

$$cost(S) = \sum_{f \in S} c(f) + \sum_{u \in U} \min_{f \in S} d(u,f)$$

Note that we assume that each user is allocated to the closest open facility, and that this is the uncapacitated version of the problem: there is no limit to the number of users a facility can serve. Even with this assumption, the problem is NP-hard [8].

Exact algorithms for this problem do exist (some examples are [7,25]), but its NP-hard nature makes heuristics the natural choice for larger instances. Much progress has been made in terms of approximation algorithms for the metric version of this problem.

In this paper, we provide an alternative that can be even better in practice. It is a hybrid multistart heuristic akin to the one we developed for the p-median problem in [36]. A series of minor adaptations is enough to build a very robust algorithm for the facility location problem, capable of obtaining near-optimal solutions for a wide variety of instances.

## 2. The algorithm

In [36], we introduce a new hybrid metaheuristic and apply it to the p-median problem. The method works in two phases. The first is a multistart routine with intensification. In each iteration, it builds a randomized solution and applies local search to it. The resulting solution (S) is combined, through a process called path-relinking, with some other solution from a pool of elite solutions (which represents the best solutions found thus far). This results in a new solution S'. The algorithm then tries to insert both S' and S into the pool. The second phase is post-optimization, which combines the solutions in the pool with one another.

### 2.1. Constructive heuristic
In each iteration i, we first define the number of facilities $p_i$ that will be open. This number is $\lceil m/2 \rceil$ in the first iteration; for $i > 1$, we pick the average number of facilities in the solutions found. The algorithm chooses $\lceil \log_2(m/p_i) \rceil$ facilities uniformly at random and selects the one among those that reduces the total service cost the most.

### 2.2. Local search
The local search allows "pure" insertions, deletions, and swaps. All possible insertions, deletions, and swaps are considered, and the best among those is performed. The local search stops when no improving move exists, in which case the current solution is a local minimum.

### 2.3. Path-relinking
Path-relinking is an intensification procedure. It takes two solutions as input, $S_1$ and $S_2$. The algorithm starts from $S_1$ and gradually transforms it into $S_2$ using insertions, deletions, and swaps.

### 2.4. Elite solutions
A new solution will be inserted into the pool only if its symmetric difference to each cheaper solution already there is at least four.

### 2.5. Intensification
After each iteration, the solution S obtained by the local search procedure is combined (with path-relinking) with a solution S' obtained from the pool.

### 2.6. Post-optimization
Once the multistart phase is over, all elite solutions are combined with one another with path-relinking.

### 2.7. Parameters
The procedure takes only two input parameters: the number of iterations in the multistart phase and the size of the pool of elite solutions. We set those values to 32 and 10, respectively, for the standard version.

---

## 3. Empirical results

### 3.1. Experimental setup
The algorithm was implemented in C++ and compiled with the SGI MIPSPro C++ compiler. The algorithm was tested on all classes from the UflLib at the time of writing and on class GHOSH.
* **BK:** 220 instances with 30 to 100 users.
* **FPP:** Hard instances for algorithms based on flip + swap.
* **GAP:** Instances with large duality gaps.
* **GHOSH:** Created following guidelines by Körkel.
* **GR:** Graph-based instances.
* **M*:** Created with a generator introduced by Kratica et al.
* **MED:** Points picked uniformly at random in the unit square.
* **ORLIB:** Beasley's OR-Library instances.

### 3.2. Results

#### 3.2.1. Quality Assessment

**Table 1: Average deviation with respect to the best known upper bounds and mean running times of HYBRID**

| Class | Deviation (%) | Time (seconds) |
|---|---|---|
| BK | 0.002 | 0.28 |
| FPP | 33.375 | 7.66 |
| GAP | 5.953 | 1.64 |
| GHOSH | -0.039 | 34.31 |
| GR | 0.000 | 0.32 |
| M* | 0.000 | 7.86 |
| MED | -0.391 | 369.67 |
| ORLIB | 0.000 | 0.17 |

**Table 2: Results for M* instances**

| Name | n | Value | Time (seconds) |
|---|---|---|---|
| mo1 | 100 | 1305.95 | 0.988 |
| mo2 | 100 | 1432.36 | 1.030 |
| mo3 | 100 | 1516.77 | 0.960 |
| mo4 | 100 | 1442.24 | 0.892 |
| mo5 | 100 | 1408.77 | 0.815 |
| mp1 | 200 | 2686.48 | 3.695 |
| mt1 | 2000 | 10069.80 | 701.167 |

**Table 3: Results for ORLIB instances**

| Name | n | Optimum | Time (seconds) |
|---|---|---|---|
| cap101 | 50 | 796 648.44 | 0.055 |
| cap102 | 50 | 854 704.20 | 0.056 |
| cap103 | 50 | 893 782.11 | 0.072 |
| capa | 1000 | 17 156 454.48 | 7.380 |
| capb | 1000 | 12 979 071.58 | 6.245 |
| capc | 1000 | 11 505 594.33 | 6.148 |

#### 3.2.2. Comparative analysis
When compared to Michel and Van Hentenryck's tabu search (TABU), both algorithms find solutions very close to the optimal on five classes: BK, GHOSH, GR, M*, and ORLIB. On harder instances (GAP, FPP), both struggle initially, but HYBRID improves at a much faster rate when given more time, demonstrating the robustness introduced by path-relinking.

---

## 4. Concluding remarks
We have studied a simple adaptation to the facility location problem of Resende and Werneck's multistart heuristic for the p-median problem. The resulting algorithm is highly effective in practice, finding near-optimal or optimal solutions of a large and heterogeneous set of instances. The combination of fast local search and path-relinking within a multistart heuristic proves to be a very effective means of finding near-optimal solutions for an NP-hard problem.

## References
[1] S. Ahn, C. Cooper, G. Cornuéjols, A.M. Frieze, Probabilistic analysis of a relaxation for the k-median problem, Mathematics of Operations Research 13 (1998) 1-31.  
[8] G. Cornuéjols, G.L. Nemhauser, L.A. Wolsey, The uncapacitated facility location problem, Discrete Location Theory, 1990.  
[16] S. Guha, S. Khuller, Greedy strikes back: Improved facility location algorithms, Journal of Algorithms 31 (1999) 228-248.  
[36] M.G.C. Resende, R.F. Werneck, A hybrid heuristic for the p-median problem, Journal of Heuristics 10 (1) (2004) 59-88.
