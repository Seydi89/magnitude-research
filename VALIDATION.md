# Validation record

Validated on 2026-09-29 with Python 3.12, NumPy, SciPy, pandas, scikit-learn and matplotlib.

- All Python files compile successfully.
- Four core tests pass: magnitude limits with duplicates, exact leave-one-out identity, budget/uniqueness invariants, and profile-drift/effective-rank checks.
- All five lightweight mechanism runners complete successfully.
- The continual-learning pilot completes a fresh three-task, seven-checkpoint smoke run and its analysis step succeeds.
- The experiment registry CLI lists all six retained research tracks.

The environment did not provide the `pytest` executable, so the four test functions were invoked directly during packaging. The supplied Conda environment installs pytest, and `pytest` is the normal user-facing command.

This validates execution and mathematical invariants, not the scientific hypotheses. Full research conclusions remain those recorded in `docs/EXPERIMENT_LEDGER.md`.
