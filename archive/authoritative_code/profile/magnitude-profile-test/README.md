# Multiscale magnitude-profile test

This code tests the prerequisite hypothesis before using magnitude for memory selection:

> Does an **ordered multiscale marginal-magnitude profile** predict residual factual information beyond fixed-scale magnitude and conventional candidate-to-set similarity summaries?

It deliberately does **not** claim that magnitude contains information absent from the complete pairwise geometry: magnitude is computed from that geometry. The strongest included comparison is therefore:

`complete canonicalized pairwise geometry + profile` versus `complete geometry`,
under both linear and random-forest models.

## What the experiment includes

- Seven controlled relationships: duplicate, paraphrase, incremental fact, temporal update, contradiction, related-but-irrelevant, and novel topic.
- Explicit atomic facts and an automatically derived `InfoDelta` target.
- Base sets of 4, 8, and 16 memories. Singleton sets are intentionally avoided.
- Log-spaced kernel scales and the complete marginal profile
  `M_scale(S ∪ {x}) - M_scale(S)`.
- Ten independently worded template families, held out as complete groups by default.
- Complete pairwise-geometry controls: candidate-to-set and within-set relations.
- Surface controls for length, punctuation, clauses, negation and digits.
- Linear, polynomial, spline and random-forest similarity baselines.
- Sigmoid decomposition into amplitude, slope, transition location and residual profile.
- Similarity, single-magnitude, full-profile, and interpretable profile-shape baselines.
- Scale-order controls: sorted profile values and independent per-example shuffling.
- Saved raw data, features, fold metrics, paired comparisons, plots, and a conservative verdict.

## Quick smoke run

The hashing backend checks that the pipeline works, but it is **not evidence about semantic embeddings**.

### Conda (recommended)

Run these commands from the project directory:

```bash
conda env create -f environment.yml
conda activate magnitude-profile
pytest
python run_experiment.py --backend hashing --groups 30 --folds 10 --output outputs/smoke
```

To update an already-created environment after the file changes:

```bash
conda env update -f environment.yml --prune
conda activate magnitude-profile
```

### `venv` alternative

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'
python run_experiment.py --backend hashing --groups 30 --folds 10 --output outputs/smoke
pytest
```

## Recommended semantic run

```bash
conda activate magnitude-profile
python run_experiment.py \
  --backend sentence-transformer \
  --model sentence-transformers/all-mpnet-base-v2 \
  --groups 300 \
  --folds 10 \
  --split-by template_family \
  --scales 32 \
  --output outputs/mpnet
```

For a transfer check, repeat with a second embedding model:

```bash
python run_experiment.py \
  --backend sentence-transformer \
  --model sentence-transformers/all-MiniLM-L6-v2 \
  --groups 300 \
  --folds 10 \
  --split-by template_family \
  --scales 32 \
  --output outputs/minilm
```

## Read these results first

- `model_results.csv`: grouped-CV aggregate performance.
- `paired_comparisons.json`: fold-paired bootstrap differences.
- `profiles_by_type.png`: class-average profile shapes.
- `verdict.json`: directional checks, not significance claims.

The key comparisons are:

1. `sim_stats + profile` versus `sim_stats + single magnitude`.
2. `complete geometry + profile` versus `complete geometry`, for linear models.
3. The same complete-geometry comparison with random forests.
4. Full profile versus its fitted sigmoid parameters.
5. Sigmoid parameters plus residual profile versus sigmoid parameters alone.
6. Profile versus nonlinear polynomial/spline similarity baselines.
7. Profile versus independently shuffled profile values.
8. Similarity + surface controls + profile versus the corresponding single-scale model.

Comparisons 2–3 test whether magnitude helps after the model receives the geometry
from which magnitude itself is calculated. Comparisons 4–5 test whether the
profile contains structure beyond a simple shifted/squeezed sigmoid. Because
marginal profiles are often monotonic, sorting may leave them literally
unchanged; `verdict.json` explicitly reports whether that ablation was
informative.

The results also report `type_macro_f1` and `type_balanced_accuracy` for the
seven-way semantic-relationship task. This is the direct diagnostic for H3:
whether different scale regimes distinguish duplicates, extensions, updates,
contradictions, irrelevant related facts, and novel topics.

## Important interpretation limits

This synthetic dataset is a **controlled diagnostic**, not final thesis evidence. It can reveal whether the pipeline sees a signal and whether that signal survives strong geometric controls. A positive result should next be tested on:

- human-reviewed examples;
- unseen semantic domains beyond the included template-held-out test;
- at least two embedding models;
- query-conditioned useful complementarity, separately from factual novelty;
- a real conversational-memory dataset before building a selector.

The labels are intentionally separate: `info_delta`, `compatible`, and `subject_relevant`. A contradiction, an update, and irrelevant trivia may all contain new atomic information, but they should not be treated as equivalent by a later memory selector.
