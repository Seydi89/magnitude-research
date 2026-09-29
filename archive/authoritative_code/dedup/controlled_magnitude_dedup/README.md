# Controlled multiscale magnitude deduplication

This experiment tests one narrow question:

> Can multiscale leave-one-out magnitude remove interchangeable memory copies
> without deleting semantically close but complementary or updated evidence?

It does **not** ask magnitude to predict relevance. Relevance retrieves and
packs memories; magnitude is used only as a pool-cleaning signal.

## Controlled dataset

Each fictional deployment profile is rendered with `1`, `2`, `4`, and `8`
copies of four facts. One group uses exact text copies and three use genuine
paraphrases. Across the four conditions, everything else is fixed:

- the underlying facts and query;
- ten distractors;
- a required primary/backup-region pair with deliberately similar wording;
- an obsolete/current platform pair;
- the first copy of every duplicated fact.

Consequently, differences between copy conditions are attributable only to
the added copies. Profiles, rather than cases, are assigned to calibration,
validation, and test, preventing cross-condition leakage.

## Methods

- `relevance`: no pool cleaning; raw cosine relevance packs the budget.
- `exact-text-dedup`: remove repeated byte-identical payloads before selection;
  this establishes whether magnitude does more than ordinary hashing.
- `cosine-dedup`: suppress a candidate when it exceeds a validation-frozen
  cosine threshold against an already retained, more relevant candidate.
- `magnitude-loo-dedup`: suppress only when a candidate is both close to an
  already retained candidate and has a small multiscale leave-one-out
  magnitude contribution.
- `MMR`: direct selection from the uncleaned pool with validation-frozen
  lambda.
- `oracle-dedup`: retain one relevance-best member from each labelled
  equivalence group. This is an upper bound, not a deployable method.

For one scale, with kernel inverse `Q = K^-1` and weighting vector
`w = Q1`, the exact leave-one-out contribution is

```text
Mag(C) - Mag(C without i) = w_i^2 / Q_ii.
```

The runner evaluates this identity at all 129 calibrated scales and integrates
the contributions uniformly in log scale. The self-test checks the formula
against explicit matrix deletion.

## Placement

Place this directory inside `synthetic_memory_pilot/` in the existing project:

```text
history_magnitude_pilot/
├── trajectory_memory/
└── synthetic_memory_pilot/
    └── controlled_magnitude_dedup/
```

## Run

The package already includes the generated 80-profile dataset. To reproduce or
replace it deterministically:

```bash
python controlled_magnitude_dedup/generate_controlled.py \
  --out controlled_magnitude_dedup/data/controlled_magnitude_dedup.json
```

Run the main MiniLM experiment:

```bash
python controlled_magnitude_dedup/run_dedup.py \
  --data controlled_magnitude_dedup/data/controlled_magnitude_dedup.json \
  --encoder minilm \
  --onnx-threads 10 \
  --report results/controlled_magnitude_dedup.json
```

Embeddings use the existing persistent SQLite cache. A quick lexical smoke run
is available with `--encoder lsa` and should not be treated as the main result.

Print an existing report without encoding or recomputation:

```bash
python controlled_magnitude_dedup/run_dedup.py \
  --report results/controlled_magnitude_dedup.json \
  --summary-only
```

## Self-tests

```bash
python controlled_magnitude_dedup/generate_controlled.py --self-test
python controlled_magnitude_dedup/run_dedup.py --self-test
```

## Reading the report

Primary safeguards:

- `gold_pool_recall_after_cleaning`: whether cleaning erased required facts;
- `complement_preservation`: whether both close, complementary memories remain;
- `current_update_preservation`: whether the current fact survives;
- `obsolete_removal`: whether the outdated fact is removed.

Deduplication metrics:

- precision: removed memories that were replaceable copies;
- recall: labelled redundant extras successfully removed;
- F1: their harmonic mean;
- pool-token reduction: storage passed to the final selector that was removed.
- exact-copy and paraphrase-copy recall: which kind of redundancy was removed.

Final utility metrics:

- evidence recall and complete evidence coverage under budgets 80 and 120;
- selected duplicate extras and obsolete memories;
- paired profile-bootstrap differences against relevance, cosine and MMR;
- results separately for 1, 2, 4 and 8 copies.

Thresholds and MMR lambda are selected on validation profiles and frozen for
test. Tuning protects evidence recall lexicographically before rewarding
deduplication quality or token savings. The oracle labels never enter a
deployable selector.

The final-check runner includes magnitude leave-one-out thresholds through
`1.0`, an exact-text hashing baseline, separate exact/paraphrase removal recall,
and current-update preservation in the terminal table. When rerunning after an
earlier report, choose a new `--report` filename because reports are never
overwritten silently.

## Semantic aspect-coverage experiment

`run_aspect_coverage.py` implements information diversity rather than geometric
diversity. Each query is decomposed into seven answer-free aspect questions.
For every aspect `a` and memory `m`, MiniLM supplies a relative support value
`p_am`. Selection maximizes probabilistic saturated coverage

```text
C_a(S) = 1 - product_{m in S}(1 - p_am)
```

mixed with ordinary query relevance. Repeated support for an already-covered
aspect therefore has diminishing value, while two geometrically close memories
can both be selected when they answer different aspects.

Run:

```bash
python controlled_magnitude_dedup/run_aspect_coverage.py \
  --data controlled_magnitude_dedup/data/controlled_magnitude_dedup.json \
  --encoder minilm \
  --onnx-threads 10 \
  --report results/controlled_aspect_coverage.json
```

The methods are relevance, MMR, embedding aspect coverage, embedding aspect
coverage with explicit textual temporal cues, and a label-based aspect oracle.
Only the oracle reads fact/validity labels. Temperature, relevance mixture and
MMR lambda are frozen on validation profiles before test.
