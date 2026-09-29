# Additional diagnostic validation

- All 17 tests pass (the nine cache tests plus eight new diagnostic tests).
- The turn and session budget oracles match exhaustive subset enumeration on
  30 seeded seven-candidate problems (128 subsets each), including missing gold
  turns and budgets that cannot accommodate any evidence.
- Checked retrieval/budget/selection loss decomposition, partial evidence within
  one session, unrelated turns from gold sessions, empty selections, undefined
  pair statistics, pooled percentile weighting, and invalid budget violations.
- The updated LongMemEval CLI was run using the real LSA cache from the preceding
  update. It encoded zero new texts, preserved every old per-case metric and
  candidate/selected ID, and saved a strict-JSON schema-2 report. The report-only
  type filter was checked on the resulting report.
- Encoder, embedding-cache and magnitude source files are byte-for-byte identical
  to the preceding update. No cache namespace invalidation was introduced.

Full MiniLM inference and the user's cleaned LongMemEval dataset were not run
here. Model-route integration tests use deterministic substitute encoders; CLI
checks use real LSA on a small fixture. No new scientific performance claim is
made by these tests.

## Previous cache-update verification

# Validation performed for the cache update

- Nine automated tests passed: exact-text deduplication and order restoration;
  reuse after reopening; interrupted batch recovery; recovery after abrupt process
  exit; separate model/text identities; visibility across two open connections;
  rejection of invalid vectors; real LSA embedding agreement and fit isolation;
  cached LongMemEval checks and type subsets without additional inference; and
  empty-evidence handling. Some tests cover more than one behavior.
- A full synthetic LSA run with six copies, the original 15/50/100 split, and the
  complete original validation grid reproduced every saved test-summary value
  and frozen parameter exactly. Its 14,598 requested texts required 926 unique
  vectors (919 memory texts and seven questions).
- The updated LongMemEval CLI was run twice on a 24-question, three-type fixture
  using real LSA. Cold and warm runs had identical case records and aggregate/
  per-type metrics. The warm run encoded zero new texts. Recall, pool statistics,
  selected counts and defined pair-similarity metrics matched the original CLI
  on that fixture. Undefined pair statistics are now null/n/a instead of NaN.
- The report-only summary command was checked on a subset of types.
- Python sources were compiled, CLI help checked, and the installer was checked
  for code backup and preservation of datasets, caches and existing results.
- The geometric magnitude implementation was compared byte-for-byte with the
  supplied file and is unchanged.

Environment: Python 3.12, NumPy 2.3.5, SciPy 1.17.0, scikit-learn 1.8.0.

MiniLM model inference and the full cleaned LongMemEval dataset were not run in
this environment. MiniLM's inference/pooling implementation was retained; the
new wrapper was tested with deterministic substitute encoders, including the
MiniLM route through the LongMemEval diagnostic. These checks establish cache
behavior and the exercised regressions, not full-benchmark scientific results.
