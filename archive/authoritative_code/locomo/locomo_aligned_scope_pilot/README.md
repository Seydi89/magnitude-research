# Aligned scope-to-scale feasibility: LoCoMo-grounded controlled episodes

This fixes the two specific mismatches in the prefix pilot:

1. The scope signal belongs to the **same questions being answered**. It no longer comes from an unrelated archive ending.
2. Both the detector and annotation sheet compare **A→B and B→C separately**. Recency weights combine those exact transitions, not a different A→C question.

We keep real LoCoMo question text, original memory records, and evidence IDs. The sequence of requests is **constructed for this experiment**: it is not a recorded natural conversation, and the results are not official LoCoMo QA scores.

## Run

Python 3.10–3.12. Use a working PyTorch environment on RunPod, or install PyTorch for your local CPU first: https://pytorch.org/get-started/locally/ .

```bash
unzip LoCoMo_Aligned_Scope_Feasibility.zip
cd locomo_aligned_scope_pilot
python -m pip install -r requirements.txt
python -m unittest -v test_alignment.py
python pilot.py --device cuda --out runs/my_run
```

For CPU replace `cuda` with `cpu`. The raw LoCoMo data and its license are included. MiniLM weights download on first use; the revision is pinned. Every output directory must be new. `--cache` permits reuse of embeddings.

Default budgets are 256 and 512 MiniLM-tokenizer tokens; 32 candidates; at most four topic bundles per conversation. You can change these before a new experiment using `--budgets`, `--candidates`, `--bundles`. Do not choose configurations from test results.

## What an episode means

Each state is an explicit revision of the user's **currently active request**, containing three real LoCoMo questions. Earlier states are superseded; we are not expected to answer every earlier question as well. No assistant answers or evaluation references are inserted into the trajectory.

A narrowing episode:
- A: one question from the focus topic plus questions from two other topics.
- B: two questions from the focus topic plus one from another topic.
- C: three different questions from the focus topic.

Broadening reverses these states. Stable controls repeat C at all three stages, or repeat A at all three stages. Thus all states contain three distinct questions: simply changing the number of points cannot produce the intended direction.

The final request explicitly asks exactly the three C questions. Its evidence target is the union of their original supporting dialogue IDs. In a matched stable control, final questions, gold evidence, candidate pool and relevance scores are identical to the changing-path case. Only the earlier scope states differ. This tests whether trajectory contributes beyond the endpoint.

Question topics are assigned by predefined keyword patterns, independently of embeddings, magnitude or retrieval results. Questions matching multiple topics are excluded. This gives transparent **constructed labels**, not guaranteed semantic labels. Some bundles may be unnatural or topic labels wrong; the blind sheet asks humans to check both scope transitions and current-request coherence.

## Scope computation

Each active question is one MiniLM embedding point. For each state:

`mu = mean(question embeddings)`

`spread = mean_i ||embedding_i - mu||²`.

The mean locates the topic; spread estimates its breadth. A mean vector alone is not a scope measurement.

Calculate `d1 = spread(B)-spread(A)` and `d2 = spread(C)-spread(B)`.

`trend = beta1*d1 + beta2*d2`, with `beta ∝ [decay, 1]` (default decay .8).

The newer change B→C receives greater weight. `--decay 1` uses equal weights; an unweighted controller is also compared automatically. The annotation sheet has a separate column for each of these exact two transitions. Float64 spread calculation and a nonzero calibration-derived deadband prevent tiny numerical drift from being labeled a scope change. Predictions are not shown in the blind sheet.

This is now a question-scope representation, not a rolling window of unrelated archive messages. It is an explicit controlled simplification of conversational request evolution, and must be labeled that way in a thesis.

## Retrieval and subset selection

All original conversation records are available as memory before the constructed episode begins. Original LoCoMo QA records are annotations, not part of stored memory text.

Each of the three current questions retrieves memory candidates independently. A deterministic round-robin union of their rankings creates one bounded candidate pool. This avoids collapsing a broad multi-question request into a single mean retrieval vector that can miss its subtopics. Relevance is maximum cosine similarity to any of the three current questions. Every scale policy receives the exact same pool and relevance scores; no gold is inserted into retrieval.

The shared objective is:

`F(S) = lambda * sum relevance(m) + (1-lambda) * |C| * sum_l omega_l * Mag_l(S)/Mag_l(C)`.

Relevance is min-max normalized within C. `lambda` is chosen on validation with uniform scales, then frozen for every scale policy. Greedy selection starts with the highest-relevance fitting record and repeatedly adds the largest objective gain that fits the remaining budget. It returns original text chunks and source IDs. It does not compress or rewrite memory, or claim a global optimum.

Scales are .25/.5/1/2/4/8 on calibration-normalized Euclidean distances. Magnitude uses an exponential distance kernel, Cholesky solves and exact Schur-complement gains, with a fixed 1e-8 ridge for numerical stability. The quantity is therefore ridge-regularized magnitude. Weights are not probabilities of evidence correctness.

Long source records are split without dropping text. A gold turn is counted recovered only when all its chunks are selected; candidate recall uses the same conservative convention. Images/captions are not encoded. These are text evidence-retrieval metrics, not official QA or sentence-level evidence accuracy.

## Comparisons

- Six fixed scales and uniform scale weights.
- Embedding-trajectory weighting, testing **both mapping directions** and strengths .5/1/2.
- Absolute final spread (does the endpoint alone suffice?).
- Unweighted trajectory (does recency help?).
- Shuffled trajectory weights (does the correct trajectory matter?).
- Relevance and fixed MMR-style baselines.
- A constructed-label controller using the independent keyword-topic directions. This separates construction-label signal from measured geometric signal, but is NOT a human oracle.
- Optional human-label controller after annotation.

Small/large-scale mapping is a hypothesis to test, not an assumed law. If validation chooses lambda=1, the objective is relevance-only and scale weights cannot affect it; the report keeps that result.

## Splits and interpretation

Conversations are sorted by sample_id: first three calibration, next three validation, remaining four test. No conversation crosses splits. The included run has 32 calibration, 44 validation and 64 test episodes (half test episodes are matched stable controls).

The primary table and validation selection use changing-scope cases. Stable cases remain in metrics and traces as matched controls. Bootstrap resamples independent conversations, not correlated episode variants. Four independent test conversations are not enough for a definitive go/stop conclusion.

Primary narrow/broad cases differ in final scope, so the absolute-scope baseline is essential. Passing a comparison against uniform alone does not establish trajectory usefulness. Inspect candidate recall: selection cannot recover evidence not retrieved.

Constructed-label agreement is a diagnostic only. Report changing transitions separately from stable controls, which are deliberately identical and easy. A larger percentage here is not natural-dialogue scope-detection accuracy.

## Files to open

- `REPORT.md`: primary selection results and paired comparisons.
- `metrics.csv`, `selections.json`: all episodes including stable controls, selected IDs and gains.
- `cases.json`: the exact three request states, source question indices, final request, candidate text, gold and relevance scores.
- `scope_features.json`: spreads, two differences, beta and predictions.
- `scope_annotation_blind.csv`: A/B/C questions and separate transition labels, with a coherence check.
- `scope_agreement.csv`: agreement with construction labels; not human ground truth.
- `frozen_validation.json`, `manifest.json`: frozen decisions, hashes, versions.

### Human validation

Independently label A→B and B→C as `narrowing`, `stable`, or `broadening` in a copy of the blind sheet. Leave uncertain rows blank and note ambiguity. Judge topic scope, not whether topic names changed. Mark whether the current three-question request is coherent. Prefer two annotators with adjudication; do not inspect model predictions first.

```bash
python score_scope.py --run runs/my_run --labels my_labels.csv --out runs/human_agreement.json
```

This scores only annotated held-out transitions and reports coverage. Empty labels give unknown accuracy.

With complete adjudicated validation/test labels, run the optional human-label scale controller:

```bash
python pilot.py --device cpu --labels my_labels.csv --out runs/human_controller
```

That controller tunes direction/strength on validation scope labels plus evidence scores, then applies the frozen mapping to held-out human scope labels. It is a diagnostic upper-level control, not deployable without a scope detector. Every other policy and split stays the same. Coherence labels are for review and do not silently filter test cases; any exclusion rule must be specified independently before another experiment.

## Attribution

LoCoMo: Adyasha Maharana, Dong-Ho Lee, Sergey Tulyakov, Mohit Bansal, Francesco Barbieri, Yuwei Fang. *Evaluating Very Long-Term Conversational Memory of LLM Agents*, ACL 2024.
https://github.com/snap-research/locomo

Raw data is included unmodified from the local release copy; hashes are recorded. Source license: CC BY-NC 4.0, in `data/LICENSE_LOCOMO.txt`. Question bundles, transitions, chunking, retrieval and scores are additions by this pilot, not tasks supplied by the dataset authors.

Development note: the first aligned development run used mean-query retrieval. Its candidate coverage was poor for broad requests. The delivered revision uses per-question round-robin retrieval; the earlier development run is not included as a confirmatory result. All results remain exploratory.
