# Metric-Space Magnitude Research

A cleaned consolidation of the completed pilots on metric-space magnitude in conversational memory and continual learning. This repository preserves meaningful positive, negative, and inconclusive experiments without retaining duplicate generations of old scripts.

**[Full report](REPORT.md)**

## What is here

- `src/magnitude_research/`: one shared magnitude implementation plus selection, deduplication, history, and diagnostic utilities.
- `experiments/`: small executable reproductions of each experimental mechanism.
- `docs/EXPERIMENT_LEDGER.md`: what each full pilot tested and found.
- `tests/`: mathematical identities, numerical behavior, and selection invariants.
- `experiments/forgetting/`: the instrumented continual-learning pilot, which stores raw geometry at every checkpoint.
- `archive/authoritative_code/`: code-only snapshots of the final historical runners, isolated from the supported API for exact lineage.

Large datasets, embedding caches, and generated result directories are deliberately excluded. The ledger records the authoritative source artifacts and outcomes. New work should use `src/`; the archive exists for traceability, not as a second active implementation.

## Setup

```bash
conda env create -f environment.yml
conda activate magnitude-research
pytest
magresearch list
```

## Run smoke experiments

```bash
python experiments/memory_selection/run_smoke.py
python experiments/history/run_smoke.py
python experiments/scope/run_smoke.py
python experiments/profile/run_smoke.py
python experiments/dedup/run_smoke.py
python experiments/forgetting/run_pilot.py --output runs/forgetting --epochs-per-task 3 --checkpoint-every 1 --reference-per-class 8 --scales 13
python experiments/forgetting/analyze_run.py runs/forgetting
```

## Central conclusion so far

Magnitude is a rigorous whole-set multiscale geometry statistic, but it has not yet shown unique practical value over simpler similarity, relevance, or displacement baselines in these pilots. History-aware retrieval itself worked; semantic aspect coverage worked in controlled redundancy; magnitude-based memory selection did not. The forgetting result remains an instrumentation-validated hypothesis that requires multiple seeds and held-out evaluation.
