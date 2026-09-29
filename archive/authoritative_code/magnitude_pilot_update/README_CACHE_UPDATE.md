# Persistent embedding cache and diagnostic update

This package contains all Python source files from the supplied project, the
updated LongMemEval diagnostic, a shared persistent cache, and tests. Supplied
historical result files are retained in the archive as reference outputs; they
are not results of the new code. The installer updates code only.

## Install into your existing project

Let any old process finish, or stop it with Ctrl-C, before replacing code.
The old LongMemEval process never persisted embeddings, so its completed encoding
cannot be recovered by this update. The updated process saves each completed batch.

Extract the archive into a separate directory. From that extracted directory run:

```bash
python install_update.py "/path/to/history_magnitude_pilot"
```

Use the existing parent directory containing BOTH `trajectory_memory` and
`synthetic_memory_pilot`. The installer backs up replaced files under
`update_backups/` and leaves your dataset, existing cache and results untouched.
Do not replace just `longmemeval.py`: it imports the new shared cache module.
Your existing magnitude-pilot environment is sufficient; SQLite is in Python's
standard library. No new dependency installation is required.

## Run one full diagnostic

From your existing `synthetic_memory_pilot` directory:

```bash
python longmemeval.py --data longmemeval_s_cleaned.json --check --encoder minilm --report results/longmemeval_minilm_diagnostics.json
```

This prints progress per question and saved batch. It writes both overall and
per-question-type results, per-case candidate/selected IDs, cache statistics,
the dataset hash and run arguments to the report. With no `--report` argument,
it automatically writes a timestamped JSON under `results/`.

The first updated run still needs to encode uncached text. Identical text shared
by different questions is reused immediately. The first model download/loading
and LSA fitting are not skipped by the cache.

## Summarize the three types without a second model run

```bash
python summarize_longmemeval.py results/longmemeval_minilm_diagnostics.json --types multi-session knowledge-update temporal-reasoning
```

This uses the saved validation cases, needs no model or dataset, and does not
encode anything. It combines those types and lists the question IDs used.
Optionally add `--out results/longmemeval_selected_types_summary.json`.

The old second `longmemeval.py --check --types ...` command still works and reuses
cached texts. However, filtering before split assignment can change validation
membership, as in the original code; newly encountered text must be encoded.
Use the summary command when you want the same cases as the full diagnostic.

## Pause and resume

Press Ctrl-C, then repeat the same command when ready. Completed batches stay on
disk; an unfinished batch may need to be recomputed. Selection and reporting are
recomputed from the beginning, but completed text embeddings are reused. The
final report is written only after the check finishes.

The default cache lives at `synthetic_memory_pilot/cache/embeddings.sqlite3`,
resolved relative to the installed code. To share a cache between project copies:

```bash
python longmemeval.py --data longmemeval_s_cleaned.json --check --encoder minilm --cache-dir "/path/to/shared/cache" --report results/longmemeval_minilm_diagnostics.json
```

Keep that cache directory. When moving/backing it up, first stop running encoders
and copy the whole directory, including any SQLite sidecar files.

By default, each batch contains up to 64 unique texts. `--cache-batch-size 16`
saves more frequently. Batches are measured in texts, not model windows: a very
long turn may still take time inside one batch. Smaller batches do not necessarily
make inference faster. MiniLM retains the existing CPU implementation.

## Additional diagnostics in this version (report schema 2)

This version preserves the encoder and cache files byte-for-byte, so the cache
created by the previous update remains valid. Install into the same project and
keep the same cache directory. Rerun the diagnostic to generate the new fields;
old reports alone do not contain enough information to reconstruct them. The
report-only summary command continues to read old reports but cannot invent
missing diagnostics.

Each budget now reports an exact **budget-feasible oracle recall**, using only
labelled evidence turns already in the candidate pool. Every labelled turn is
worth one unit of evidence recall, so sorting their costs and taking the longest
affordable prefix gives the exact maximum. Whole turns are indivisible. This
oracle is an evaluation device and does not influence retrieval or selection.

The report separates the recall loss into:

- `oracle_retrieval_loss`: gold evidence absent from the pool.
- `oracle_unavoidable_budget_loss`: evidence in the pool that cannot all fit.
- `oracle_selection_gap`: oracle recall minus actual selected recall.

Their sum is `1 - evidence_recall`, both per question and after equal-question
averaging. If the last term is close to zero, there is little room for a different
selector to improve labelled-turn recall under that exact pool and budget.
This is not an answer-quality or minimum-semantic-evidence oracle.

**Cosine distributions** include p50, p75, p90, p95 and p99 for pairs of selected
turns, all gold evidence turns of the same question, and gold evidence inside the
pool. Aggregate percentiles pool within-question pair values; questions with
more pairs get more weight. No cross-question pairs are formed. The existing
0.8/0.9 threshold fractions retain their original equal-question averaging.
Raw pair values are saved so type-subset summaries recompute exact percentiles
instead of averaging per-question percentiles. A single turn has no pair, so
its undefined pair statistics are null/n/a. Cosine similarity does not establish
that evidence is interchangeable rather than complementary.

**Session diagnostics** report distinct selected sessions, the largest session's
share of the selection, and gold-evidence session coverage. A session counts as
covered only if a selected turn in it is evidence-labelled. Visiting unrelated
text in that session is recorded separately and does not count as evidence.
For questions with gold evidence spanning multiple sessions, the report gives
fractions covering all/some/no gold sessions and the fraction selecting all gold
turns. These are different: one turn per session need not cover all evidence.

A separate session oracle chooses the cheapest available evidence turn per gold
session and maximizes the number of sessions represented under the budget. It
may choose different turns from the turn-recall oracle. Both ceilings use the
same candidate pool and cost units. Session concentration alone is not evidence
that a diversification method will improve performance.

New implementation: `synthetic_memory_pilot/diagnostics.py`. The installer
includes it automatically. Selection rules and validation splits are unchanged.

## What changed

- `trajectory_memory/embedding_cache.py`: SQLite cache, exact-text deduplication,
  model-specific namespaces, batch transactions and interruption recovery.
- `trajectory_memory/encoders.py`: cache identities include actual model and
  tokenizer hashes, revision, pooling, encoder source hash, or fitted LSA state.
- `synthetic_memory_pilot/longmemeval.py`: cached encoding, progress, saved JSON,
  per-type summaries, empty-evidence handling and explicit undefined pair metrics.
- `synthetic_memory_pilot/summarize_longmemeval.py`: summaries of saved checks.
- `evaluate.py`, `detail.py`, `run_all.sh`: shared cache support. `detail.py` also
  accepts cache/model path overrides instead of relying on another machine's
  stored cache path. `summarize.py` ignores unrelated diagnostic JSON files.
- `t2c.py`: compatibility entry point for maintained `detail.py` diagnostics.
- `trajectory_memory/data.py` and `__main__.py`: the conversational preparation
  command supports the same cache and batch options.

The cache stores exact text, its SHA-256 key and float32 vectors. Cache identity
excludes question types, evidence labels, retrieval budgets and selection settings.
Changing the model, tokenizer, pooling, encoder source or fitted LSA coordinate
system creates a separate namespace. Different LSA fits cannot share vectors.

Old experiment-wide `emb_*.npz` files are left untouched. They lack the new full
encoder identity and are not imported automatically. The new SQLite cache is
built on its first use. Subsequent synthetic `detail.py` runs use it too.

## Interpretation boundaries

The LongMemEval check still evaluates relevance retrieval and pairwise similarity;
it does not run magnitude selection or generate LLM answers. Its export format
is not yet connected to synthetic `evaluate.py`. Low near-duplicate percentages
alone are not a reason to reject magnitude. Costs remain regex word/punctuation
units, not the answer model's tokenizer.

The magnitude objective, V0–V5 selection rules, synthetic generator and split
rules were not changed by this cache update. Existing experimental limitations
identified in the review still apply.

## Verify the installation

From the project parent directory:

```bash
python -m unittest discover -s tests -v
```

Tests use deterministic substitute encoders and a small real LSA fit; they do not
download MiniLM or run a full LongMemEval benchmark. See `VALIDATION.md` for the
checks actually performed while preparing this update.
