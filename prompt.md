# Master Agent Prompt: Hybrid LP-GRASP for the Uncapacitated Facility Location Problem

**System Role:** You are an expert Operations Research developer specializing in Python and mathematical optimization.

**Task:** Implement a hybrid metaheuristic for the Uncapacitated Facility Location Problem (UFLP). You must combine an exact Linear Programming (LP) relaxation with a GRASP (Greedy Randomized Adaptive Search Procedure) based on the framework by Resende and Werneck (2006).

**Problem Definition:** Given a set of potential facilities $F$ with setup costs $c(f)$ and a set of users $U$, the goal is to open a subset of facilities $S \subset F$ to minimize the total cost:

$$cost(S) = \sum_{f \in S} c(f) + \sum_{u \in U} \min_{f \in S} d(u,f)$$

Each user is allocated to the closest open facility.

## Core Data Structures (Strict Implementation)

You must use the following data classes to manage the instance data and algorithmic state. Do not deviate from these types.

```python
from dataclasses import dataclass, field
from typing import List, Dict, Set, Tuple

@dataclass
class UFLPInstance:
    facilities: List[int]
    customers: List[int]
    setup_costs: Dict[int, float]
    service_costs: Dict[int, Dict[int, float]]

@dataclass
class SolutionState:
    open_facilities: Set[int] = field(default_factory=set)
    total_cost: float = float('inf')
    closest_facility: Dict[int, int] = field(default_factory=dict)
    second_closest_facility: Dict[int, int] = field(default_factory=dict)
    save: Dict[int, float] = field(default_factory=dict)
    loss: Dict[int, float] = field(default_factory=dict)
    extra: Dict[Tuple[int, int], float] = field(default_factory=dict)

```

## Phase 1: Simplex-Based LP Relaxation (The Improvement)

Before starting the metaheuristic, write a function using `PuLP` that models the UFLP mathematically.

1. **Variables:** Define facility opening variables $y_f$ and customer assignment variables $x_{u,f}$.
2. **Relaxation:** Relax the binary constraint on $y_f$. Define it as a continuous variable: $0 \le y_f \le 1$.
3. **Constraints:** Ensure every customer is fully assigned ($\sum x_{u,f} = 1$) and a customer can only be assigned to a facility if it is open ($x_{u,f} \le y_f$).
4. **Execution:** Solve the model using the default Simplex solver.
5. **Output:** Extract the fractional values of $y_f$ and return them as a probability dictionary: `Dict[int, float]`.

## Phase 2: Probabilistic Constructive Heuristic

The base algorithm builds a randomized solution by semi-greedily adding facilities. We will modify this to use the exact LP probabilities.

1. **Input:** The `UFLPInstance` and the probability dictionary from Phase 1.
2. **Logic:** Loop through the facilities. Instead of picking facilities uniformly at random, use the LP fractional values to bias the selection. If the LP assigned facility `A` a value of 0.85, it has an 85% chance of being added to the initial `SolutionState`.
3. **Completion:** Once the initial subset is selected, greedily assign every customer to their closest open facility and calculate the initial `total_cost`.

## Phase 3: Fast Local Search (Strict Delta Evaluation)

The local search improves the solution by considering all possible insertions, deletions, and swaps. **CRITICAL REQUIREMENT:** Do not recalculate the total cost from scratch for each move. You must update the delta using the three specific components defined by Resende and Werneck:

1. **`save`:** The decrease in solution value due to the insertion of a closed facility.
2. **`loss`:** The increase in solution value due to the removal of an open facility.
3. **`extra`:** A positive correction term used when swapping. It accounts for a user reassigned from a removed facility directly to the newly inserted one.
4. **The Profit Formula:** The profit of swapping a closed facility $f_i$ with an open facility $f_r$ must be calculated exactly as: $profit(f_i, f_r) = save(f_i) - loss(f_r) + extra(f_i, f_r)$.
5. **Sparsity:** You must implement `extra` as a sparse matrix (as provided in the `SolutionState` tuple dictionary) because it is only nonzero when facilities are close to each other.

## Phase 4: Main Execution Loop

Write a main function that orchestrates these modules:

1. Parse the problem data into `UFLPInstance`.
2. Run Phase 1 to get LP probabilities.
3. Execute Phase 2 to build the initial `SolutionState`.
4. Pass the state to Phase 3. Apply the move with the highest positive profit (whether insertion, deletion, or swap).
5. Update `save`, `loss`, and `extra` incrementally after each move.
6. Stop and return the solution when no improving move exists (local optimum reached).