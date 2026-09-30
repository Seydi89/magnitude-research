# Metric-Space Magnitude for Conversational Memory Selection

This report documents a series of experiments on a simple question: can metric-space magnitude improve which memories a conversational system keeps under a token budget?

The answer from these pilots is mostly no. Conversational history helped retrieval, and explicit semantic coverage helped when several pieces of evidence were required. Magnitude itself did not improve selection over strong, simpler baselines. 

## Starting point

Long-running assistants may retrieve more memories than fit in the active context. Relevance ranking is necessary, but it can return repeated evidence or concentrate on only one part of a multi-part request. I wanted to test whether a whole-set geometric measure could help choose a compact but varied subset.

Metric-space magnitude was a plausible candidate. It measures the effective size of a finite metric space across different distance scales and has been used to evaluate diversity in latent representations. I found no previous work directly applying it to conversational-memory selection, although the general ideas of diversity-aware retrieval and geometric subset selection are well established.

## Magnitude in brief

For a set of embedded memories `X` with pairwise distances `d(i,j)`, define the similarity matrix

```math
Z_{ij}(t) = e^{-t d(i,j)}.
```

The magnitude at scale `t` is

```math
\mathrm{Mag}_t(X)
= \mathbf{1}^{T} Z(t)^{-1} \mathbf{1}.
```

The scale parameter `t` controls the resolution at which distances are considered. At small values of `t`, more points appear similar and the magnitude approaches one. As `t` increases, only nearby points remain similar and distinct points contribute more independently. Evaluating magnitude across a range of scales produces a **magnitude profile**, rather than a single diversity score.

This is geometric information, not semantic utility. Two memories can be close in embedding space but provide different facts; an unusual memory can also be irrelevant. That distinction became central to the experiments.

## What I tested

The work developed through several related but separate pilots.

### 1. Magnitude as a selection objective

After relevance-based candidate retrieval, I selected memories greedily under a token budget. A candidate's score combined query relevance with its marginal contribution to the magnitude of the selected set. I compared this with relevance-only selection and MMR.

The final implementation used regularized kernels, cost-aware selection, validation-frozen hyperparameters, dense scale audits and identical candidate pools across methods.

### 2. History-aware retrieval

On TopiOCQA, I tested whether recorded conversation history improved retrieval for the current question. I then compared history-aware relevance with fixed-scale and multiscale magnitude selection.

This experiment used previous recorded answers, so it is teacher-forced conversational retrieval rather than an autonomous long-term-memory system.

### 3. Trajectory-conditioned scales

Using real LoCoMo questions arranged into controlled sequences, I tested whether narrowing or broadening between question bundles could change the weighting of magnitude scales. The final version aligned the constructed scope labels, transition boundaries and detector inputs.

The sequences were controlled constructions from real questions, not naturally observed topic trajectories, and only four independent test conversations were available.

### 4. Magnitude-profile diagnostic

Before building further selectors, I tested whether a candidate's full marginal-magnitude profile predicted factual novelty beyond simpler features. The controls included candidate-to-set similarity summaries and a canonicalized representation of the full pairwise geometry from which magnitude is computed.

This was a prerequisite test: if the profile did not add information beyond simpler geometry, there was little reason to expect it to improve selection.

### 5. Controlled deduplication

I generated pools with one, two, four or eight exact and paraphrased copies of the same facts. Leave-one-out magnitude contributions were used to remove geometrically redundant candidates before relevance-based packing.

The main safety checks were whether the method preserved complementary facts, retained current information and avoided keeping obsolete facts. I also implemented semantic aspect coverage, which decomposed each request into answer-free subquestions and rewarded new coverage directly.

## Results

| Experiment | Question | Result |
| --- | --- | --- |
| Synthetic and LongMemEval selection | Does magnitude improve evidence recall under token budgets? | **Negative.** Magnitude did not beat relevance; MMR matched or beat it. |
| TopiOCQA history retrieval | Does history help, and does magnitude add anything afterward? | **Mixed.** History-aware relevance clearly improved over current-question-only retrieval. Magnitude added no measurable gain. |
| LoCoMo scope controller | Can narrowing or broadening guide scale weighting? | **Inconclusive.** The corrected pilot showed no clear advantage over a fixed scale. |
| Profile diagnostic | Does the full profile add predictive information beyond simpler geometry? | **Negative prerequisite.** It was predictive, but did not decisively improve on strong similarity and geometry controls. |
| Controlled deduplication | Can magnitude remove copies without deleting useful evidence? | **Mixed.** It worked when redundancy was deliberately common, but semantic aspect coverage was substantially better. |

Full experiment notes are in [`docs/EXPERIMENT_LEDGER.md`](docs/EXPERIMENT_LEDGER.md).

## Why magnitude did not help enough

The experiments point to several limitations.

First, magnitude is order-invariant. It can describe the geometry of a moving conversation window, but it cannot recover the order of turns inside that window. Comparing consecutive windows captures geometric change, not conversational direction by itself.

Second, geometric distinctiveness is not the same as usefulness. Magnitude can identify a point that adds a new direction to an embedding set, but it cannot tell whether that direction supplies a required fact. Relevance gating reduced obvious failures, but it did not solve this semantic gap.

Third, the pure geometric objective has a strong distance preference. Starting from one selected memory, the unregularized two-point marginal gain is

```math
\tanh\left(\frac{td}{2}\right),
```

which increases with distance at every scale. With equal costs and no relevance term, the next choice is therefore the farthest eligible candidate. Scale weighting changes the strength of this preference, not its direction. The complete hybrid selector also includes relevance and token cost, so this observation does not determine every selection it makes.

Finally, magnitude is computed from the same pairwise geometry available to simpler baselines. The profile experiment found no clear evidence that its nonlinear compression added useful semantic information once strong geometric controls were included.

## What did work

Two results were more encouraging.

History-aware relevance substantially improved evidence recall over retrieval based only on the current question. This supports using recent conversational context when forming the retrieval request, independently of magnitude.

Semantic aspect coverage performed best in the controlled redundancy experiment. Instead of asking whether two memories looked different, it asked whether a candidate contributed evidence for a still-uncovered part of the request. This matched the actual evaluation target more directly.

Both findings point in the same direction: useful complementarity should be defined semantically first. Geometry can then be tested as a possible predictor of that target, rather than treated as the target itself.

## Limitations

- Several experiments were controlled feasibility studies rather than natural deployments.
- The LoCoMo trajectory study had few independent test conversations.
- The TopiOCQA run used recorded previous answers and an easier annotation-derived corpus.
- Evidence recall does not guarantee final answer quality.
- The controlled deduplication data intentionally contained much more redundancy than LongMemEval.
- Hyperparameters were frozen on validation data, but the overall research direction evolved across repeated pilots.

These limitations mean the project supports a narrow conclusion: magnitude did not improve the tested memory-selection pipelines. It does not show that magnitude can never be useful in retrieval or representation analysis.

## Related exploratory extension

I also ran a small continual-learning pilot that tracked magnitude-profile drift while a model forgot earlier tasks. Early-layer drift correlated with later forgetting in one synthetic run, but ordinary representation displacement was slightly stronger. This remains a feasibility observation and is not part of the conversational-memory conclusion.

## Conclusion

The original idea was that multiscale geometry might provide a cheap, principled way to balance relevance and diversity as a conversation changes. The implementation worked as intended, but the experiments did not show a selection benefit. In the datasets with little redundancy, diversity was not the main problem. In the controlled dataset with substantial redundancy, explicit semantic coverage was better aligned with the task.

For me, the main outcome is methodological: define the missing semantic utility first, then ask whether geometry predicts it. Starting from a geometric mechanism and hoping that it corresponds to useful complementarity put the steps in the wrong order.

## References

- Leinster, T. (2013). *The magnitude of metric spaces*. Documenta Mathematica, 18, 857–905.
- Limbeck, K., Andreeva, R., Sarkar, R., & Rieck, B. (2024). *Metric Space Magnitude for Evaluating the Diversity of Latent Representations*. NeurIPS 2024. [arXiv:2311.16054](https://arxiv.org/abs/2311.16054)
- Andreeva, R., Ward, J., Skraba, P., Gao, J., & Sarkar, R. (2025). *Approximating Metric Magnitude of Point Sets*. AAAI 2025. [DOI](https://doi.org/10.1609/aaai.v39i15.33687)
- Emmerich, M. T. M., Pereverdieva, K., & Deutz, A. H. (2026). *Selecting a Maximum Solow–Polasky Diversity Subset in General Metric Spaces Is NP-hard*. [arXiv:2604.05495](https://arxiv.org/abs/2604.05495)
