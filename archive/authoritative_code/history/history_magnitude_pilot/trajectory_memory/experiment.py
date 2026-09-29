"""Frozen validation choices, paired selection controls, and held-out evidence scoring."""
import csv
import json
import time
import importlib.metadata
from collections import defaultdict
from pathlib import Path
import numpy as np
from threadpoolctl import threadpool_limits
from .data import dump, digest, new_directory
from .encoders import normalized
from .magnitude import (distances, Geometry, calibrate_profiles, log_weights,
                        magnitude, select, RIDGE)


def write_csv(path, rows):
    if rows:
        with Path(path).open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


class Retrieval:
    """Receives only public passages and embeddings, never evidence annotations."""
    def __init__(self, corpus, arrays, candidates):
        self.corpus, self.arrays = corpus, arrays
        self.candidates = min(candidates, len(corpus))
        self.emb = np.asarray(arrays["corpus"], dtype=np.float64)

    def vector(self, i, mode="history", beta=0.5):
        current = self.arrays["current"][i]
        if mode == "current":
            return current
        if mode == "concat":
            return self.arrays["concat"][i]
        history = self.arrays["permuted" if mode == "permuted" else "history"][i]
        return normalized(current + beta * history)

    def pool(self, i, mode="history", beta=0.5):
        scores = self.emb @ self.vector(i, mode, beta)
        # Corpus was sorted by content ID; stable ties do not reveal gold order.
        order = np.argsort(-scores, kind="stable")[:self.candidates]
        return dict(indices=order, relevance=scores[order],
                    costs=np.array([self.corpus[j]["cost"] for j in order]),
                    ids=[self.corpus[j]["id"] for j in order])


def selection_metrics(pool, selected, gold_ids):
    gold, available = set(gold_ids), set(pool["ids"])
    ids = [pool["ids"][j] for j in selected["indices"]]
    hit = gold & set(ids)
    reachable = gold & available
    return dict(selected_ids=ids, evidence_recall=len(hit) / len(gold),
                gold_hit=int(bool(hit)), complete=int(gold <= set(ids)),
                candidate_recall=len(reachable) / len(gold),
                conditional_recall=len(hit) / len(reachable) if reachable else None,
                # Exact annotated passage matches, not exhaustive semantic precision.
                annotated_fraction=len(hit) / len(ids) if ids else 0.0,
                tokens=selected["tokens"], selected_count=len(ids))


def paired_comparison(rows, a, b, bootstrap_samples=2000):
    pairs = defaultdict(dict)
    for row in rows:
        if row["turn"] > 1:
            pairs[(row["case"], row["budget"])][row["method"]] = row
    grouped = defaultdict(list)
    for pair in pairs.values():
        grouped[pair[a]["conversation"]].append(pair[a]["evidence_recall"] - pair[b]["evidence_recall"])
    values = np.array([np.mean(v) for v in grouped.values()])
    rng = np.random.default_rng(71)
    sample = rng.choice(values, size=(bootstrap_samples, len(values)), replace=True).mean(axis=1)
    return dict(a=a, b=b, mean_difference=float(values.mean()),
                ci95=np.quantile(sample, [.025, .975]).tolist(), conversations=len(values),
                unit="equal conversation weights; average turns and budgets within conversation")


def summarize(rows):
    groups = defaultdict(list)
    for row in rows:
        if row["turn"] > 1:
            groups[(row["method"], row["budget"])].append(row)
    summary = []
    for (method, budget), rr in groups.items():
        by_conv = defaultdict(list)
        for r in rr:
            by_conv[r["conversation"]].append(r)
        result = dict(method=method, budget=budget, turns=len(rr), conversations=len(by_conv))
        for field in ["evidence_recall", "candidate_recall", "complete", "tokens", "selected_count"]:
            result[field] = float(np.mean([np.mean([r[field] for r in group]) for group in by_conv.values()]))
        result["turn_weighted_recall"] = float(np.mean([r["evidence_recall"] for r in rr]))
        summary.append(result)
    return summary


def run(args):
    with threadpool_limits(limits=args.threads):
        return _run(args)


def _run(args):
    start = time.perf_counter()
    out = new_directory(args.out)
    data = Path(args.data)
    meta = json.loads((data / "manifest.json").read_text())
    for f, expected in meta["files_sha256"].items():
        if digest(data / f) != expected:
            raise ValueError(f"Prepared file changed: {f}")
    corpus = json.loads((data / "corpus.json").read_text())
    queries = json.loads((data / "queries.json").read_text())
    labels = json.loads((data / "labels.json").read_text())
    arrays = dict(np.load(data / "embeddings.npz", allow_pickle=False))
    if len(arrays["current"]) != len(queries) or len(arrays["corpus"]) != len(corpus):
        raise ValueError("Embedding rows do not match data")
    retrieval = Retrieval(corpus, arrays, args.candidates)
    split = {name: [i for i, q in enumerate(queries) if q["split"] == name]
             for name in ["calibration", "validation", "test"]}
    val = [i for i in split["validation"] if queries[i]["turn"] > 1]
    if not val or not split["test"] or not split["calibration"]:
        raise ValueError("Nonempty conversation splits and validation follow-ups required")
    # Choose calibration pools without label access, balanced across conversations.
    by_conv = defaultdict(list)
    for i in split["calibration"]:
        by_conv[queries[i]["conversation"]].append(i)
    probes = []
    for indices in by_conv.values():
        positions = np.linspace(0, len(indices) - 1, min(3, len(indices))).astype(int)
        probes.extend(indices[p] for p in positions)
    raw_distances = [distances(retrieval.emb[retrieval.pool(i, beta=.5)["indices"]]) for i in probes]
    positive = np.concatenate([d[np.triu_indices(len(d), 1)] for d in raw_distances])
    positive = positive[positive > 1e-10]
    if not len(positive):
        raise ValueError("All calibration embeddings coincide")
    unit = float(np.median(positive))
    print(f"Calibrating profiles on {len(probes)} unlabeled pools", flush=True)
    scale_config = calibrate_profiles([d / unit for d in raw_distances], args.dense_nodes,
                                      args.max_scales, args.profile_tolerance)
    dense_scales = np.array(scale_config["dense_scales"])
    nodes = np.array(scale_config["indices"])
    adaptive_weights = np.array(scale_config["weights"])
    dense_weights = log_weights(dense_scales)
    fixed_indices = np.linspace(0, len(dense_scales) - 1, 9).round().astype(int).tolist()
    dump(out / "scale_calibration.json", scale_config | {"distance_unit": unit, "probe_ids": [queries[i]["id"] for i in probes]})
    print(f"Profile grid: {len(nodes)}/{len(dense_scales)} nodes; error {scale_config['calibration_max_interpolation_error']:.4g}", flush=True)
    # Tune history strength with the relevance-only selector. Zero is allowed to win.
    beta_grid, lambda_grid = [0.0, .25, .5, 1.0, 2.0], [1.0, .75, .5, .25, 0.0]
    def relevance_score(i, beta, budget):
        pool = retrieval.pool(i, beta=beta)
        result = select(None, pool["relevance"], pool["costs"], budget, method="relevance")
        return selection_metrics(pool, result, labels[queries[i]["id"]]["gold_ids"])["evidence_recall"]
    def validation_mean(items):
        groups = defaultdict(list)
        for i, score in items:
            groups[queries[i]["conversation"]].append(score)
        return float(np.mean([np.mean(v) for v in groups.values()]))
    beta_scores = {b: validation_mean((i, relevance_score(i, b, budget)) for i in val for budget in args.budgets)
                   for b in beta_grid}
    beta = max(beta_grid, key=lambda b: beta_scores[b])
    # Geometry-only policies get separate validation choices. No test selection.
    scores = defaultdict(list)
    for i in val:
        pool = retrieval.pool(i, beta=beta)
        dense = Geometry.make(distances(retrieval.emb[pool["indices"]], unit), dense_scales)
        adaptive = Geometry(dense.bank[nodes], dense.full[nodes], dense.scales[nodes])
        similarity = retrieval.emb[pool["indices"]] @ retrieval.emb[pool["indices"]].T
        gold = labels[queries[i]["id"]]["gold_ids"]
        for budget in args.budgets:
            for lam in lambda_grid:
                result = select(adaptive, pool["relevance"], pool["costs"], budget, lam, adaptive_weights)
                scores[("profile", lam, -1)].append((i, selection_metrics(pool, result, gold)["evidence_recall"]))
                for j in fixed_indices:
                    fixed = Geometry(dense.bank[j:j + 1], dense.full[j:j + 1], dense.scales[j:j + 1])
                    result = select(fixed, pool["relevance"], pool["costs"], budget, lam, np.ones(1))
                    scores[("fixed", lam, j)].append((i, selection_metrics(pool, result, gold)["evidence_recall"]))
                result = select(None, pool["relevance"], pool["costs"], budget, method="mmr",
                                mmr_lambda=lam, similarity=similarity)
                scores[("mmr", lam, -1)].append((i, selection_metrics(pool, result, gold)["evidence_recall"]))
    val_scores = {key: validation_mean(items) for key, items in scores.items()}
    chosen = {kind: max([k for k in scores if k[0] == kind], key=lambda k: val_scores[k])
              for kind in ["profile", "fixed", "mmr"]}
    frozen = dict(history_beta=beta, beta_validation=beta_scores,
                  profile_lambda=chosen["profile"][1], fixed_lambda=chosen["fixed"][1],
                  fixed_index=chosen["fixed"][2], fixed_scale=float(dense_scales[chosen["fixed"][2]]),
                  mmr_lambda=chosen["mmr"][1], selection_validation={str(k): v for k, v in val_scores.items()},
                  validation_unit="equal conversation weights; follow-up turns only",
                  data_manifest_sha256=digest(data / "manifest.json"))
    dump(out / "frozen_validation.json", frozen)
    print(f"Frozen history weight {beta}; profile lambda {frozen['profile_lambda']}", flush=True)
    rows, details, audits, pool_records, convergence = [], [], [], [], []
    convergence_groups = set()
    for progress, i in enumerate(split["test"]):
        q = queries[i]
        label = labels[q["id"]]
        pool = retrieval.pool(i, beta=beta)
        dense = Geometry.make(distances(retrieval.emb[pool["indices"]], unit), dense_scales)
        adaptive = Geometry(dense.bank[nodes], dense.full[nodes], dense.scales[nodes])
        j = frozen["fixed_index"]
        fixed = Geometry(dense.bank[j:j + 1], dense.full[j:j + 1], dense.scales[j:j + 1])
        similarity = retrieval.emb[pool["indices"]] @ retrieval.emb[pool["indices"]].T
        variants = {mode: retrieval.pool(i, mode=mode, beta=beta) for mode in ["current", "concat", "permuted"]}
        refined = None
        if q["turn"] > 1 and q["conversation"] not in convergence_groups and len(convergence_groups) < 12:
            convergence_groups.add(q["conversation"])
            refined = Geometry.make(distances(retrieval.emb[pool["indices"]], unit),
                                    np.geomspace(dense_scales[0], dense_scales[-1], 2 * len(dense_scales) - 1))
        pool_records.append(dict(case=q["id"], **{mode: dict(ids=p["ids"], relevance=p["relevance"].tolist())
                                                 for mode, p in {"history": pool, **variants}.items()}))
        for budget in args.budgets:
            methods = {}
            for mode, p in {"current": variants["current"], "history": pool,
                            "concat": variants["concat"], "permuted_history": variants["permuted"]}.items():
                methods[mode + "_relevance"] = (p, select(None, p["relevance"], p["costs"], budget, method="relevance"))
            methods["history_mmr"] = (pool, select(None, pool["relevance"], pool["costs"], budget,
                                                  method="mmr", mmr_lambda=frozen["mmr_lambda"], similarity=similarity))
            methods["history_fixed_magnitude"] = (pool, select(fixed, pool["relevance"], pool["costs"], budget,
                                                              frozen["fixed_lambda"], np.ones(1)))
            methods["history_profile_magnitude"] = (pool, select(adaptive, pool["relevance"], pool["costs"], budget,
                                                                frozen["profile_lambda"], adaptive_weights))
            methods["history_dense_magnitude"] = (pool, select(dense, pool["relevance"], pool["costs"], budget,
                                                              frozen["profile_lambda"], dense_weights))
            # A numerical audit must still exercise geometry when validation selects
            # relevance-only. This fixed .5 control is not a validation-selected winner.
            methods["history_profile_diagnostic"] = (pool, select(adaptive, pool["relevance"], pool["costs"], budget,
                                                                  .5, adaptive_weights))
            methods["history_dense_diagnostic"] = (pool, select(dense, pool["relevance"], pool["costs"], budget,
                                                                .5, dense_weights))
            for method, (p, result) in methods.items():
                metric = selection_metrics(p, result, label["gold_ids"])
                selected_ids = metric.pop("selected_ids")
                row = dict(case=q["id"], conversation=q["conversation"], turn=q["turn"],
                           transition=label["transition"], reference_answer_in_history=label["reference_answer_in_history"],
                           method=method, budget=budget, **metric)
                rows.append(row)
                details.append(row | dict(selected_ids=selected_ids, candidate_ids=p["ids"],
                                          relevance=p["relevance"].tolist(), trace=result["trace"]))
            picked = methods["history_profile_diagnostic"][1]["indices"]
            reference = methods["history_dense_diagnostic"][1]["indices"]
            subset = np.array([magnitude(k[np.ix_(picked, picked)]) for k in dense.bank])
            ratio = subset / dense.full
            reconstructed = np.interp(np.log(dense_scales), np.log(dense_scales[nodes]), ratio[nodes])
            if refined is not None:
                fine_result = select(refined, pool["relevance"], pool["costs"], budget, .5, log_weights(refined.scales))
                fine_ratio = np.array([magnitude(k[np.ix_(picked, picked)]) for k in refined.bank]) / refined.full
                convergence.append(dict(case=q["id"], budget=budget,
                                        dense_nodes=len(dense_scales), refined_nodes=len(refined.scales),
                                        integral_gap=float(abs(dense_weights @ ratio - log_weights(refined.scales) @ fine_ratio)),
                                        same_selected_set=set(reference) == set(fine_result["indices"])))
            oracle_reachable_costs = sorted(corpus[k]["cost"] for k in pool["indices"] if corpus[k]["id"] in label["gold_ids"])
            remaining, reachable_count = budget, 0
            for cost in oracle_reachable_costs:
                if cost <= remaining:
                    remaining -= cost
                    reachable_count += 1
            audits.append(dict(case=q["id"], budget=budget, audit_lambda=.5,
                               selected_profile_interpolation_error=float(np.max(np.abs(ratio - reconstructed))),
                               integral_error=float(abs(adaptive_weights @ ratio[nodes] - dense_weights @ ratio)),
                               same_selected_set=set(picked) == set(reference),
                               recall_difference=selection_metrics(pool, methods["history_profile_diagnostic"][1], label["gold_ids"])["evidence_recall"] -
                                                 selection_metrics(pool, methods["history_dense_diagnostic"][1], label["gold_ids"])["evidence_recall"],
                               budget_feasible_candidate_ceiling=reachable_count / len(label["gold_ids"]),
                               full_profile=dense.full.tolist(), selected_profile=subset.tolist()))
        if (progress + 1) % 50 == 0:
            print(f"Evaluated {progress + 1}/{len(split['test'])} real turns", flush=True)
    summary = summarize(rows)
    comparisons = [paired_comparison(rows, a, b) for a, b in [
        ("history_relevance", "current_relevance"),
        ("history_relevance", "concat_relevance"),
        ("history_relevance", "permuted_history_relevance"),
        ("history_profile_magnitude", "history_relevance"),
        ("history_profile_magnitude", "history_mmr"),
        ("history_profile_magnitude", "history_fixed_magnitude"),
        ("history_profile_magnitude", "history_dense_magnitude"),
        ("history_profile_diagnostic", "history_relevance"),
        ("history_profile_diagnostic", "history_dense_diagnostic")]]
    write_csv(out / "metrics.csv", rows)
    write_csv(out / "summary.csv", summary)
    write_csv(out / "profile_audit.csv", [{k: v for k, v in r.items() if not k.endswith("_profile")} for r in audits])
    dump(out / "selections.json", details)
    dump(out / "candidate_pools.json", pool_records)
    dump(out / "profiles.json", audits)
    dump(out / "dense_convergence.json", convergence)
    dump(out / "comparisons.json", comparisons)
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["method"], row["transition"])].append(row)
    breakdown = [dict(method=method, transition=transition, turn_budget_pairs=len(rr),
                      evidence_recall=float(np.mean([r["evidence_recall"] for r in rr])))
                 for (method, transition), rr in grouped.items()]
    write_csv(out / "transition_breakdown.csv", breakdown)
    source_files = sorted(Path(__file__).parent.glob("*.py"))
    run_meta = dict(arguments=vars(args), elapsed_seconds=time.perf_counter() - start,
                    data_manifest=meta, ridge=RIDGE,
                    source_sha256={f.name: digest(f) for f in source_files},
                    versions={n: importlib.metadata.version(n) for n in ["numpy", "scipy", "scikit-learn"]},
                    test_conversations=len({queries[i]["conversation"] for i in split["test"]}),
                    test_turns=len(split["test"]), llm_answers_evaluated=False)
    dump(out / "manifest.json", run_meta)
    from .report import make_report
    make_report(out, data, queries, labels, corpus, summary, details, audits, comparisons, scale_config, frozen, run_meta)
    print(f"Finished in {time.perf_counter() - start:.1f}s: {out / 'REPORT.md'}", flush=True)
