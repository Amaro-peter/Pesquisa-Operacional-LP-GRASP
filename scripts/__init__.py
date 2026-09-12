"""
Solver, benchmark and reporting modules for the Hybrid LP-GRASP UFLP study.

Modules import one another as `scripts.<module>`, so the repository root must be
on `sys.path`. Run the entry points as modules from the repository root:

    python -m scripts.uflp_solver --file data/cap134.txt
    python -m scripts.run_experiments
    python -m scripts.run_scaling_benchmark
    python -m scripts.download_and_run_real_world
    python -m scripts.run_koerkel_ghosh

Data and output paths are resolved relative to the working directory, so run
them from the repository root.
"""
