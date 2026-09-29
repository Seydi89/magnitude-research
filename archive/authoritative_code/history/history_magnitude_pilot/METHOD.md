# Exact method and experimental contract

## Information available at turn t

The current utterance q_t and recorded pairs (q_i, a_i) for i < t are public. Each candidate memory is an original Wikipedia passage with stable ID, source title and full text. Answers a_t, gold passage IDs, rationales and topic annotations are evaluation data only.

TopiOCQA is an information-retrieval test bed. Calling its document passages memory does not turn it into a personal conversational-memory benchmark.

## History-dependent representation

Let E be the fixed encoder and N(v) be L2 normalization, leaving a zero vector zero. With up to six previous turns and decay d=0.8:

    h_t = N(sum_i d^(t-1-i) E("Question: q_i Answer: a_i"))
    u_t = N(E(q_t) + beta h_t)

The normalization of the history sum makes its total scale independent of the number of available prior turns. Beta is selected from {0, .25, .5, 1, 2} on validation only. Current-question-only retrieval uses beta=0. The order-control reverses the history vectors before applying the same recency weights; it is a corruption diagnostic, not a semantically valid alternative-history benchmark.

The concatenation baseline encodes the same six prior turns plus the current question as one text. No query rewrite is generated and no future conversational goal is inferred.

Candidates are the highest-cosine-scoring 32 passages for each representation. Equal scores break by content-derived ID. History selectors share exactly the same ordered pool C and scores.

## Metric-space magnitude

For candidate embeddings x_i and a distance unit u frozen from calibration:

    D_ij = ||x_i - x_j||_2 / u
    K_s = exp(-s D) + epsilon I, epsilon = 1e-8
    Mag_s(S) = 1^T (K_s[S,S])^-1 1
    Mag_s(empty) = 0

Distances are Euclidean, not squared Euclidean distances or 1-cosine values presented as a metric. Float64 Cholesky solves evaluate magnitude. The ridge means this is regularized magnitude; duplicates are numerically permitted. It is not a relevance or correctness probability.

For a new candidate j, write A=K[S,S], b=K[S,j], c=K[j,j]. Its exact regularized magnitude gain is:

    Delta Mag(j | S) = (1 - b^T A^-1 1)^2 / (c - b^T A^-1 b).

The implementation maintains the Schur complement and residual for all remaining candidates simultaneously. Each selected item applies rank-one updates across scales; no inverse is explicitly formed. Nonpositive pivots fail visibly instead of being silently clipped. Tests compare sequential gains against independent Cholesky solves, including duplicate embeddings.

## Multiscale objective

Relevance r_i is min-max normalized within C. The proposed finite profile objective is:

    F(S) = lambda sum_{i in S} r_i
         + (1-lambda) |C| / log(s_max/s_min)
           * integral_{log(s_min)}^{log(s_max)}
             Mag_exp(z)(S) / Mag_exp(z)(C) dz.

Dividing by full-pool magnitude aligns the scale-wise terms. Multiplying by |C| makes the fine-scale cardinality limit comparable to an item-wise relevance sum. These are modeling choices, not mathematical necessities. Uniform weighting in log-scale is an explicit prior; other measures would be different objectives.

The implemented integral is trapezoidal quadrature. At unequal log-scale nodes, each weight equals half of its adjacent interval widths, normalized by the total interval length. We do not weight nodes equally merely because they were sampled.

The code also evaluates fixed scales. Each method tunes its relevance coefficient on validation. Lambda=1 is included and wins ties before nonzero geometric variants, preventing forced magnitude effects.

## Subset search

Greedy selection chooses the largest marginal objective gain per cost unit that fits the remaining budget. For relevance and magnitude objectives, its final objective value is compared against the best fitting singleton. MMR stops when its best fitting incremental score is nonpositive. All methods obey the same memory budget.

This is a transparent heuristic. Magnitude is not asserted to be submodular here; there is no claimed knapsack approximation ratio or global optimum. The method spends a budget; it does not learn the minimum necessary number of memories.

LSA-run budgets use regex word-or-punctuation counts. MiniLM runs use its tokenizer's WordPiece counts on the exact `[ID] title + text` payload. Neither is claimed to be the total prompt token count of an arbitrary downstream reader. Recent history, instructions and generation allowance are separate.

## Scale calibration

The median positive pairwise distance in geometry-only calibration pools sets u. Up to three evenly spaced turns per calibration conversation generate probe pools with a fixed beta=.5; evidence labels are not used.

Bounds start at 1/64 and 16. They expand until calibration full-pool curves approach their coarse and fine limits within the requested tolerance, or reach explicit caps. Exactly duplicate distance rows use their regularized duplicate-group cardinality limit at the fine end.

The reference grid has 129 log-spaced points. Monitors include full-pool magnitude divided by pool size and Mag(S)/Mag(C) for ranked prefixes of sizes 1,2,4,8,16 plus a seeded random subset. Starting at the range endpoints, the grid adds the reference-grid position with the largest error under piecewise-linear interpolation of all monitors. It stops at tolerance .002 or the node cap of 65. Achieved errors and cap failures are recorded.

This monitors many calibration shapes; it does not prove coverage of every subset or unseen profile. Numerical held-out audits use a fixed lambda=.5 even if the primary validation choice is lambda=1. For every test question and budget, compare the adaptive and dense selected sets, recalls and the selected-subset ratio's integral. A 257-point convergence audit checks a fixed subset of test geometries. These diagnostics do not feed back into the frozen policy.

## Dataset and evaluation

The default shared corpus pools the published dev reference passages into one content-deduplicated store. This uses answer annotations to define a restricted universe, so it cannot support full-corpus claims. It does not insert a query's gold passage into its retrieved candidates. Whole passages are retained; a passage that exceeds the memory budget is not silently cropped and may be impossible to select.

Source conversation IDs and complete histories are checked against the raw release. Scored cases exclude source unanswerable questions and missing-evidence cases. Full source histories remain intact, including unanswerable earlier turns.

Conversation splits: 12 calibration, 20 validation, 40 test by seed-hashed conversation ID. All are internal splits of the official dev release, so this is exploratory evaluation, not an official test score. Corpus documents are shared among splits, as in a static retrieval index; conversations are disjoint. The LSA encoder fits only the unlabeled shared passage corpus, not test questions or labels.

Primary evidence recall = fraction of the annotated supporting passage IDs selected. In this release each eligible query has one positive passage, so recall equals annotated passage hit. Candidate recall is measured before subset selection. Conditional recall and the budget-feasible candidate ceiling distinguish retrieval failure from selection failure. Other passages may also answer correctly despite lacking an annotation; annotated-match fraction is not exhaustive relevance precision.

Primary summaries exclude first turns and give each conversation equal weight. For paired comparisons, average per-question/budget differences within a conversation, then bootstrap conversations. First turns and source document-topic continue/switch/return tags are retained as diagnostics. Reference-answer occurrence in prior answers is a crude string-based tag only, not a human judgment of answerability from history.

The optional reader gets identical recent history and a method's selected passages. A no-memory control is exported alongside it. Reference answers are never in exported evaluation prompts except when they legitimately already occur in recorded prior history. Pilot answer scores and citation-ID checks cannot establish entailment or causal reliance; no LLM answer result is included in the retrieval run.
