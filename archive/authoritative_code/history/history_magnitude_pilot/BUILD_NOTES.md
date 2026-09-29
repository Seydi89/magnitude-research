# Included run: what was and was not established

The included run uses genuine TopiOCQA conversational questions, preserved chronological history and released supporting passages. There are 815 scored questions across 72 conversations: 12 calibration, 20 validation and 40 test conversations. The test split has 454 scored turns; primary comparisons exclude first turns.

## Results

Conversation-weighted evidence recall, averaged over the two budgets:

| Method | Recall |
|---|---:|
| Current-question relevance | 16.85% |
| History-dependent relevance | 39.09% |
| Concatenated recent history | 29.78% |
| Reversed-history control | 34.31% |
| Validation-selected magnitude profile | 39.09% |
| Explicit nonzero magnitude diagnostic (lambda=.5) | 22.85% |

The history-dependent representation improved over current-only retrieval by 22.25 percentage points; the exploratory conversation bootstrap interval was [18.16, 26.63]. Reversing prior-turn weighting reduced performance, which supports sensitivity to recency in this setup. It does not establish a sophisticated trajectory representation or semantic scope detection.

Both the fixed-scale and multiscale selectors selected relevance weight 1.0 on validation. Therefore their geometric contributions were zero, and their primary results equal history-relevance. The .5 diagnostic exercises real multiscale geometry and performed worse. The experiment does not support a magnitude quality benefit in this encoder/corpus/objective setting.

## Numerical profile feasibility

The first 33-node cap missed the calibration interpolation tolerance. The cap was increased to 65 in response to that geometry-only calibration diagnostic, before inspecting held-out evidence results. The final grid used 43 nodes and met the .002 calibration tolerance.

At a nonzero geometric weight, the 43-node and 129-node selectors agreed on 99.89% of 908 test case/budget subsets and had identical annotated evidence recall. The maximum held-out selected-profile ratio interpolation error was .001846, and maximum integral error was .000297.

The 129-versus-257-node convergence check, on the first follow-up of twelve test conversations and both budgets, produced the same selected subsets in all 24 comparisons. Maximum integral difference was 2.27e-7. These finite-range empirical checks do not prove uniform accuracy for every unseen geometry or every scale.

The nonzero-weight diagnostic is explicitly separate from the validation-selected method: a numerical selection comparison at relevance weight 1.0 would not test the effect of the magnitude approximation.

## Validation performed

- Thirteen unit tests passed, including analytic two-point magnitude, exact Schur gains with duplicate embeddings, independent numerical integration, unequal-grid quadrature, budget/objective recomputation, label separation, prefix alignment and history-dependent retrieval.
- All 815 prepared queries passed source-prefix and source-question alignment checks.
- The independent artifact audit verified all 454 test candidate-pool records and 9,080 selected subsets: hashes, independent rankings, shared pools, costs, evidence scores, trace IDs, profile errors and aggregate calculations.
- The standalone profile figure was visually inspected.
- The offline explorer passed JavaScript syntax and sixteen simulated DOM interactions across questions, methods and budgets, including the relevance-only disclosure and 43-node curve. A full browser render was unavailable because the Chromium executable was absent.
- Example answer prompts were exported without a reader call. Answer parsing/scoring utilities have unit tests; no live reader or LLM quality evaluation was run.

## Limits

This run uses TF-IDF/SVD, not a pretrained semantic encoder. The pretrained download encountered a network approval cancellation; it was not retried through an alternate route. MiniLM support is included for an environment where its weights are available, but that adapter was not exercised end to end here.

The 1,940-passage corpus is annotation-derived and restricted. The scores are not official full-Wikipedia TopiOCQA results. Recorded prior answers are available, so this is teacher-forced retrieval. No paid model service was used.

This package establishes a runnable, auditable first-stage experiment with a useful history signal and an accurate numerical magnitude-profile approximation on the tested data. It does not establish a retrieval-quality benefit from magnitude.
