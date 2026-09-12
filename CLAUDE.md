# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

This repo implements and benchmarks a **Hybrid LP-GRASP heuristic** for the Uncapacitated Facility
Location Problem (UFLP), following Resende & Werneck (2006). The core idea: bias the GRASP
constructive phase using fractional values from an LP relaxation (solved via PuLP/CBC) instead of
uniform random selection, then run a best-improvement local search with exact delta evaluation
(save/loss/extra) for insertions, deletions, and swaps.

The project includes a comprehensive testing gauntlet enforcing unit tests, mutation testing, high test coverage, and regression tests for all code changes.

### Prefer Existing Dependencies Over Custom Implementations

When given a task that requires implementing a technical feature, **always evaluate whether an existing solution should be reused before writing custom code**.

Follow this decision process:

1. **Inspect the current dependencies first.**
   Check the project's existing dependencies and determine whether any of them already provide the required functionality, either directly or through an appropriate API.

2. **If no existing dependency provides the functionality, research established libraries.**
   Search for well-known, reputable, actively maintained, and trustworthy libraries that can provide the required functionality and are compatible with the project's technology stack.

3. **Prefer established solutions over custom implementations.**
   If a suitable existing dependency is found, use it rather than implementing the functionality from scratch.

4. **Implement custom code only as a last resort.**
   Write a custom implementation **only when**:

   * no suitable existing dependency is already installed;
   * no reputable external library provides the required functionality; or
   * existing libraries were evaluated but do not satisfy the project's requirements.

5. **Do not add dependencies unnecessarily.**
   If an existing dependency already solves the problem, do not introduce another library. When considering a new dependency, evaluate its maintenance status, reputation, adoption, compatibility, security, license, and whether it is actively maintained.

The goal is to **reuse reliable existing solutions whenever possible and minimize unnecessary custom code and dependencies**.

## Setup & running

```bash
pip install -r requirements.txt   # pulp, matplotlib, pytest, pytest-cov, mutmut

# Run test suite & coverage (enforces >= 85% coverage gate)
pytest
pytest --cov --cov-report=term-missing --cov-report=xml:coverage.xml

# Run mutation testing gauntlet
mutmut run
mutmut results
mutmut show <mutant_id>

# Run SonarQube static & quality analysis
sonar-scanner

# Solve a single instance
# Entry points live in scripts/ and import each other as `scripts.<module>`,
# so run them as modules from the repository root (NOT `python scripts/x.py`).
python -m scripts.uflp_solver                                # random demo instance
python -m scripts.uflp_solver --file data/cap134.txt         # OR-Library (Beasley) instance
python -m scripts.uflp_solver --n-facilities 25 --n-customers 40 --seed 7
python -m scripts.uflp_solver --quiet                        # suppress verbose output

# Run the full comparative benchmark (LP-biased vs. alpha-GRASP baseline) against
# data/cap134.txt and the generated LP-duality-gap correlation suite
python -m scripts.run_experiments

# Run the large-instance scaling benchmark (imports helpers from run_experiments)
python -m scripts.run_scaling_benchmark

# Real-world California Housing benchmark (4 arms, 3 seeds) -> output/california_4M_results.{md,json}
python -m scripts.download_and_run_real_world
python -m scripts.download_and_run_real_world --from-json output/california_4M_results.json

# Koerkel-Ghosh benchmark, the hard family where the LP relaxation is weak
python -m scripts.run_koerkel_ghosh
```

`scripts/run_experiments.py` and `scripts/run_scaling_benchmark.py` use `ProcessPoolExecutor` with
`max_workers = os.cpu_count()`. All scripts resolve data and output paths relative to the working
directory, so **run them from the repository root**.

## Agent Quality & Testing Gauntlet (Unit, Mutation & Regression)

This is **non-negotiable** for any change to solver modules, optimization algorithms, data structures, or benchmark orchestration: **every code and function must be accompanied by unit tests, mutation tests, and regression tests (for bug fixes)**. Write tests alongside or before the code, and ensure all gates pass before declaring work complete.

### Layers

| Layer | Tooling here | Objective | Gate |
|---|---|---|---|
| 1 — Unit Tests & Invariants | `pytest tests/unit` | Boundaries, calculations, exact deltas, solution construction, invariants | 100% pass; zero hollow assertions |
| 2 — High Test Coverage | `pytest-cov` / `coverage` | Enforce deep code exercise across all branches and execution paths | Line coverage ≥ 85% (`--cov-fail-under=85`) |
| 3 — Mutation Gauntlet | `mutmut run` | Kill deliberate bugs (AST mutations: boolean flips, boundary operators, deleted logic) | Mutation score ≥ 85% on changed logic |
| 4 — SonarQube Quality Gate | `sonar-scanner` | Static analysis, code smells, maintainability, duplication, coverage ingestion | Quality gate OK; 0 open blocking issues |

### Golden Rules

- **Red-Green-Refactor.** Write the failing unit tests first, run them, confirm they fail *for the reason you expect*, then write the minimum code to pass. Refactor only with the suite green.
- **Assertion Rigor.** No hollow/tautological assertions — `assert result is not None` is not acceptable where a specific value, cost delta, state transition, or assignment is required. A test that still passes when you delete the function body is not a test.
- **Fast-to-Slow.** Run unit tests and coverage first; trigger the mutation gauntlet once the unit suite is 100% green.
- **Fail-Fast.** On any layer failure, stop, fix the root cause, and re-run *that* layer before moving on.
- Never mark a task complete with a failing or skipped test. If a test is genuinely wrong, say so and explain — do **not** silently delete it, weaken its assertion, or add `@pytest.mark.skip`.

### Regression Test Contract

Every bug fix — including issues introduced and caught mid-refactor — must add a dedicated regression test named `test_<symptom>_regression.py` under `tests/regression/`.

Requirements, all of them:
1. **Fail on the pre-fix code.** Prove it: revert the fix, run the spec, verify RED, restore, verify GREEN. A regression test that has never been seen red is an assumption, not a guard.
2. **Documented defect context.** Its file and function docblock states what broke, why it mattered, and which defect id / issue it belongs to.
3. **Counterweight case.** It carries a counterweight case asserting the fix did not overshoot or break complementary behavior (e.g., verifying a boundary fix on facility deletion still prohibits deleting the single remaining open facility).

### The Per-Change Gate

Run the **whole** checklist below after **every single change** — not once per task, and not only when something feels risky. A "change" is any edit that lands in solver logic, data structures, heuristics, or scripts:

1. `pytest` — ensure 100% pass rate.
2. `pytest --cov --cov-fail-under=85` — verify coverage meets or exceeds 85%.
3. `mutmut run` — ensure mutants on changed logic are killed.
4. `sonar-scanner` (when SonarQube server is running) — verify quality gate and coverage import.

## Architecture

**`scripts/uflp_solver.py`** is the canonical, self-contained solver module — other scripts import from it
rather than reimplementing solver logic. It is organized in numbered pipeline phases, and this
phase structure (not file layout) is the mental model to use when reasoning about the algorithm:

1. **Data structures**: `UFLPInstance` (facilities, customers, setup costs, service costs) and
   `SolutionState` (open facilities, total cost, closest/second-closest facility per customer, and
   the `save`/`loss`/`extra` dicts used for delta evaluation).
2. **Instance I/O**: `parse_orlib_instance` (Beasley OR-Library format) and
   `generate_random_instance` (Euclidean random instances).
3. **Phase 1 — LP relaxation** (`solve_lp_relaxation`): relaxes facility-opening variables to
   `[0,1]`, solves with PuLP's CBC, returns fractional `y[f]` values as opening probabilities.
4. **Phase 2 — Probabilistic construction** (`construct_solution`): opens each facility with
   probability `max(lp_prob[f], eps)` instead of uniform randomness; falls back to the single
   cheapest facility if none open.
5. **Phase 3 — Local search** (`local_search` + helpers): best-improvement search over
   insertions/deletions/swaps. Move profit is computed via **exact** delta formulas
   (`_compute_insert_profit`, `_compute_delete_profit`, `_compute_swap_profit`) to avoid numerical
   drift; the `save`/`loss`/`extra` structures are maintained on `SolutionState` per the
   Resende–Werneck formulation but are informational — acceptance always uses the exact deltas.
   Critically, `_apply_move_and_recompute` fully recomputes closest/second-closest assignments and
   auxiliary data after every move (no incremental delta bookkeeping across iterations), so the
   algorithm is correctness-first rather than maximally fast.
6. **Phase 4 — Orchestration** (`solve_uflp`): wires the three phases together and prints timing
   and a full customer-assignment report when `verbose=True`.

**`scripts/run_experiments.py`** imports solver internals from `scripts/uflp_solver.py` (`local_search`,
`_find_closest_two`, `_compute_total_cost`, `_compute_auxiliary_data`) and adds its own
LP-relaxation/exact-IP solving and constructors (`construct_lp_biased_solution`,
`construct_uniform_solution`) to run a head-to-head comparison: the LP-biased hybrid solver vs. a
classic savings-based α-GRASP baseline (α=0.2). It benchmarks both against `cap134.txt` (20 seeds)
and against a generated 10-instance "LP duality gap correlation suite" (`generate_suite_instance`)
that interpolates from fully Euclidean (k=1) to fully random/non-Euclidean (k=10) to study how LP
fractionality degrades the probabilistic construction. Exact IP optima are solved with PuLP/CBC as
the ground-truth reference for gap calculations. Work is parallelized across seeds/instances with
`ProcessPoolExecutor.map`, so task functions passed to the executor must be pickleable (module-level
functions taking plain tuples), not closures.

**`scripts/run_scaling_benchmark.py`** imports helpers from both `scripts/uflp_solver.py` and `scripts/run_experiments.py`
and sweeps instance size (20x30 up to 250x800 facilities×customers) to compare wall-clock time and
solution-quality scaling between the LP-biased and uniform-random constructors, since solving the
LP relaxation itself has growing overhead as instances scale up.

**`scripts/download_and_run_real_world.py`** builds a 2000x2000 UFLP instance from the California
Housing census blocks and compares five arms against the same LP bound: LP rounding, rounding +
local search (the deterministic ablation), LP-biased GRASP, α-GRASP, and local search with no LP at
all. `render_report` writes `output/california_4M_results.md` and a JSON sidecar; `--from-json`
re-renders the prose from the sidecar without re-running the benchmark.

**`scripts/koerkel_ghosh.py`** generates Körkel-Ghosh instances to the published specification (the
standard hard family, where the LP relaxation is genuinely fractional).
**`scripts/run_koerkel_ghosh.py`** runs the same arms over that family and writes
`output/koerkel_ghosh_results.{md,json}`.

**Generated artifacts live in `output/`** (`california_4M_results.md`, `koerkel_ghosh_results.md`,
`cap134.md`, `cap134.png`, `scaling_results.md`, `scaling_analysis.png`, and the JSON sidecars).
Every script resolves them through its own `OUTPUT_DIR` constant, and the path a script prints is
the path it wrote. They are regenerated by the `run_*` scripts and must not be hand-edited.

> **Reports are generated, never hand-written.** Every figure in a results file is a measured value
> carried in a typed record, and every comparative phrase ("lower", "faster", "tighter") is computed
> from those values rather than written into the template, so a report cannot state a conclusion its
> own table contradicts. Two regression suites enforce it:
> `test_report_narrative_regression.py` for the California and Körkel-Ghosh reports, and
> `test_scaling_discussion_regression.py` for the scaling report. Both feed the renderer
> mirror-image datasets and require the wording to flip, and both scan the generator source for
> hardcoded comparative verdicts — ignoring docstrings, so a fix may still explain the defect it
> fixed.
