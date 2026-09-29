# History-dependent retrieval with magnitude-profile memory selection

A runnable first-stage feasibility study on **real, unrewritten TopiOCQA questions and recorded conversational prefixes**. It measures whether the annotated supporting passage enters a token-budgeted memory subset.

The package includes a completed CPU run, cached embeddings, original source files, an offline evidence browser, mathematical tests, and an optional answer-generation stage.

## Start here

1. Open `runs/verified/REPORT.md` for results and limitations.
2. Open `runs/verified/explorer.html` in a browser. Choose a question, method and budget. Inspect its actual history, selected passages, missing gold evidence, greedy decisions and magnitude profile.
3. Read `METHOD.md` for the exact objective and evaluation protocol.

The included run uses **TF-IDF + truncated SVD (LSA)**, a local lexical/latent encoder. It is **not a pretrained semantic embedding result**. A pretrained-model download was blocked in the build environment. The optional MiniLM adapter is provided but was not exercised in that environment.

## Rerun the included experiment — no model download or API key

Python 3.10+ is recommended. From this directory:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
python -m trajectory_memory run --data data/prepared --out runs/my_run
python verify_run.py --data data/prepared --run runs/my_run
```

On Windows activate with `.venv\Scripts\activate` instead. Every output directory must be new. The cached-data run requires no network access and no GPU. It generates all reports again.

## Rebuild embeddings or use more conversations

Original sources are in `data/raw`. If using a source-only checkout, download them first:

```bash
python -m trajectory_memory fetch --out data/raw
```

Reproduce the included LSA preparation:

```bash
python -m trajectory_memory prepare --encoder lsa --out data/lsa_rebuilt
python -m trajectory_memory run --data data/lsa_rebuilt --out runs/lsa_rebuilt
```

The default splits are 12 calibration, 20 validation, 40 test conversations, selected by a seeded hash of conversation ID. The source dev release contains more conversations; for a separately specified larger exploratory experiment:

```bash
python -m trajectory_memory prepare --encoder lsa --splits 20 40 100 --out data/larger
python -m trajectory_memory run --data data/larger --out runs/larger
```

Do not choose new settings because they improve the already inspected test results. The supplied results are exploratory; freezing a new protocol and reserving new conversations is necessary for confirmation.

## Use pretrained semantic embeddings

```bash
python -m pip install -r requirements-minilm.txt
python -m trajectory_memory prepare --encoder minilm --out data/minilm
python -m trajectory_memory run --data data/minilm --out runs/minilm
```

The adapter downloads pinned `sentence-transformers/all-MiniLM-L6-v2` ONNX weights and its tokenizer. It uses CPU inference, masked mean pooling, and L2 normalization. Inputs longer than 254 content tokens are embedded in nonoverlapping windows and pooled; full memory text is retained. Model and tokenizer file hashes are recorded.

For already-downloaded weights, supply `--model-dir /path/to/model`, containing `tokenizer.json` and `onnx/model.onnx`. Run the tests and inspect this new run separately: the pretrained adapter was **not end-to-end validated here**. No pretrained result is implied by the included LSA scores.

## What is compared

| Method | Public retrieval input | Subset selection |
|---|---|---|
| Current question only | Current question | Relevance |
| History + relevance | Current question + recency-weighted prior Q/A vectors | Relevance |
| Concatenated recent history | The same recent history as one text | Relevance |
| Reversed-history control | Same current question; reversed prior-turn weighting | Relevance |
| History + MMR | Identical history candidate pool | Validation-tuned MMR |
| History + fixed magnitude | Identical history candidate pool | Validation-tuned fixed scale and relevance mixture |
| History + magnitude profile | Identical history candidate pool | Validation-tuned relevance plus integrated magnitude |
| History + dense profile | Identical history candidate pool | Same mixture, dense numerical reference |
| Profile/dense diagnostics | Identical history candidate pool | Explicit fixed relevance weight 0.5, so geometry is active |

The profile method may legitimately select relevance weight 1.0 on validation, making its geometric contribution zero. The code preserves that outcome. The separately labeled 0.5 diagnostics let you inspect actual magnitude-based selections and test numerical approximation even when the selected method uses no geometry.

History weighting is an explicit, modest baseline. There is no LLM query rewriter, semantic topic detector, learned switch gate, or narrowing/broadening scale controller in this first stage.

## Continuous profile versus a finite scale grid

The target is a **finite integral over log(scale)**. A dense 129-point grid supplies a numerical reference. Calibration selects a smaller nonuniform grid by refining the largest interpolation errors in:

- Full candidate-pool magnitude curves.
- Ranked-prefix subset-to-pool magnitude ratios.
- Seeded random-subset ratios.

Integration weights account for the unequal log-scale spacing. Equal weights at adaptively spaced nodes would change the objective and are not used.

Scale bounds expand using calibration geometry until coarse and fine tail criteria are met, subject to explicit numerical caps. The range, distance unit and grid freeze before test evaluation. `scale_calibration.json` records achieved errors and whether the tolerance was met.

Every held-out question compares adaptive and dense selections at a nonzero geometric weight. `profile_audit.csv` reports interpolation error, integral error, selected-set agreement and recall differences. `dense_convergence.json` compares the 129-node reference with 257 nodes on fixed first follow-ups from up to twelve test conversations. The code never silently changes scales in response to test evidence scores.

The profile is a numerical approximation, **not an exact continuous computation or a guarantee for all unseen geometries**. A global calibrated profile measure is used for every question; adaptive sampling here concerns numerical resolution, not conversation-driven scale preference.

## What the data supports

- Real TopiOCQA questions, original wording and original chronological prefixes.
- Correct current-question/evidence alignment via conversation ID, turn ID and independently checked question text.
- Recorded prior answers available as history; no current/future answers available to retrieval.
- Exact source passage identities based on title and full text, matching the released evaluator's identity convention.
- Equal candidate pools, relevance scores and budgets for all history-based subset selectors.
- Strictly separate public inputs and evaluation labels.
- Conversation-level calibration/validation/test separation and paired conversation bootstrap intervals.

**Corpus limitation:** the bundled 1,940-passage universe is formed by pooling the dev release's reference passages globally. These are independent of each individual retrieval call, and gold is not inserted into its candidate pool. Nevertheless, corpus construction uses the benchmark's annotations and is easier than searching the full Wikipedia collection. These are restricted-corpus feasibility scores, not official TopiOCQA benchmark scores or evidence of general-world retrieval performance.

To index an externally assembled corpus, use `prepare --corpus corpus.jsonl`, with one `{ "id": "...", "title": "...", "text": "..." }` object per line. It replaces the corpus completely; missing gold passages remain misses. Passage identity uses exact title+text, so use the published passage boundaries to reproduce annotation-based scoring.

The source includes unanswerable turns. They remain in later turns' recorded history, but are excluded as scored retrieval questions; exclusions are recorded. Switch/return breakdowns use document-topic annotations only during evaluation. They do not establish semantic narrowing/broadening accuracy.

## Optional LLM answer stage

Retrieval scores directly answer whether annotated evidence was selected. For answer generation, export a matched set of prompts, including a no-memory baseline:

```bash
python -m trajectory_memory export-prompts --run runs/verified --out reader/prompts.jsonl --limit 30
```

This does not call an LLM. Each prompt has only recorded recent history, the current question and the selected passages with source IDs. No reference answer or gold marker is exposed.

If you have a local OpenAI-compatible inference server, make explicit model calls:

```bash
python -m trajectory_memory answer --prompts reader/prompts.jsonl --out reader/answers.jsonl --base-url http://localhost:8000/v1 --model YOUR_MODEL
python -m trajectory_memory score-answers --prompts reader/prompts.jsonl --answers reader/answers.jsonl --out reader/scores.json
```

An authenticated endpoint can use the `MEMORY_READER_API_KEY` environment variable. The `answer` command calls only the endpoint you supply and may incur its charges. It is not called automatically. The endpoint integration was not run against a live reader during this build.

The scorer reports normalized exact match, token F1, citation-ID validity, gold citation recall and missing-answer coverage. It is a pilot scorer, not the official TopiOCQA reader evaluator. Missing outputs count as zero text scores. ID validity does not establish that a passage entails the answer; citation support requires inspection or independent entailment judgments. Correctness alone also does not establish causal use of retrieved memory.

## Main outputs

| File | Purpose |
|---|---|
| `REPORT.md` | Results, comparisons, numerical checks and limits |
| `explorer.html` | Offline question/history/evidence/profile browser |
| `magnitude_profiles.png` | Standalone scientific profile figure |
| `metrics.csv`, `summary.csv` | Per-question and conversation-weighted metrics |
| `selections.json` | Selected IDs, scores, costs and greedy traces |
| `candidate_pools.json` | Independent retrieval variants and shared selection pool |
| `scale_calibration.json` | Scale bounds, nodes, weights, tolerance and monitor error |
| `profile_audit.csv`, `dense_convergence.json` | Held-out approximation diagnostics |
| `frozen_validation.json` | Validation choices, including relevance-only outcomes |
| `manifest.json` | Configuration, versions, source hashes and data provenance |

## Attribution

TopiOCQA: Vaibhav Adlakha, Shehzaad Dhuliawala, Kaheer Suleman, Harm de Vries and Siva Reddy. *TopiOCQA: Open-domain Conversational Question Answering with Topic Switching*, TACL 2022. [Paper](https://arxiv.org/abs/2110.00768), [official repository](https://github.com/McGill-NLP/topiocqa). Original data is distributed under CC BY-NC-SA 4.0; see `LICENSE_TOPIOCQA.txt`. The bundled derivative data retains that license. Wikipedia material retains its applicable attribution and licensing; source article titles and passage IDs are preserved.

Magnitude: Tom Leinster, *The magnitude of metric spaces*. [Paper](https://arxiv.org/abs/1012.5857). The selection objective, scale quadrature and experiment in this package are an implementation proposal; no novelty or optimality claim is made.

MiniLM: [official model card](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2). Optional model assets are not bundled.
