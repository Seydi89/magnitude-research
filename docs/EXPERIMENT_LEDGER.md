# Experiment ledger

This ledger separates completed evidence from ideas and prevents negative results from disappearing during cleanup.

| Experiment | Target | Outcome | Interpretation |
|---|---|---|---|
| Synthetic/LongMemEval selection | Improve evidence recall under token budgets | Negative | Magnitude never beat relevance; MMR matched or beat it. |
| TopiOCQA history retrieval | Use history, then add magnitude selection | Mixed | History helped strongly; magnitude added zero over history-aware relevance. |
| LoCoMo scope controller | Adapt scale weights to narrowing/broadening | Inconclusive | Corrected alignment produced no clear advantage over fixed scales. |
| Profile-information diagnostic | Predict semantic marginal information | Negative prerequisite | Magnitude profiles did not decisively improve on similarity/geometry controls. |
| Controlled deduplication | Remove exact/paraphrased redundancy | Mixed | Worked in redundant controlled pools; semantic aspect coverage performed much better. |
| Catastrophic-forgetting diagnostic | Predict future old-task loss | Feasibility only | Hidden-1 profile drift correlated with forgetting, but displacement was stronger in one synthetic run. |

## Source lineage

The cleanup was based on these latest authoritative artifacts:

- `magnitude_pilot_handoff_diagnostics.tar.gz` — final numerical and cache safeguards for memory selection.
- `History_Dependent_Magnitude_Pilot.zip` — verified TopiOCQA history experiment.
- `LoCoMo_Aligned_Scope_Feasibility(1).zip` — alignment-corrected scope experiment.
- `magnitude-profile-test.zip` — strongest geometry-controlled profile diagnostic.
- `controlled_magnitude_dedup.zip` — controlled redundancy and aspect-coverage experiments.
- `magnitude_forgetting_pilot.zip` — checkpointed continual-learning feasibility run.

Earlier numbered copies, intermediate one-off scripts, caches, and large result dumps are intentionally not duplicated. Their meaningful hypotheses and outcomes are represented above.

## Rules for future work

1. Add an experiment entry before adding a new runner.
2. Keep relevance/semantics separate from geometric diversity.
3. Tune on validation data and freeze before testing.
4. Compare magnitude against the strongest simpler statistic derived from the same distances.
5. Store generated outputs outside source control.
6. Mark conclusions as positive, negative, inconclusive, or feasibility-only.

