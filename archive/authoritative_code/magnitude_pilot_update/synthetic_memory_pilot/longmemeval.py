"""LongMemEval loader and cached relevance/redundancy diagnostics.

Exports retain real turn evidence labels; evaluate.py is still the synthetic experiment.

  python longmemeval.py --data /path/to/longmemeval_s.json --check
  python longmemeval.py --data /path/to/longmemeval_s.json --out cases_s.json.gz

Memory unit = one turn (evidence labels are per turn). One "case" = one question, with its own
candidate pool drawn only from that question's haystack.

Expected input (LongMemEval_S / _M, one JSON list):
  question_id, question_type, question, answer, question_date,
  haystack_session_ids: [str], haystack_sessions: [[{role, content, has_answer?}, ...], ...],
  haystack_dates: [str], answer_session_ids: [str]
Questions whose question_id ends with "_abs" are abstention questions: no evidence to find, so
they are skipped (the benchmark's own retrieval evaluation does the same).
"""
import argparse, gzip, json, re, sys, os, tempfile, time
from datetime import datetime, timezone
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
from diagnostics import budget_diagnostics, cosine_pairs, distribution, summarize_extended

POOL_DEFAULT = 32
BUDGETS_DEFAULT = [256, 512, 1024]
SPLITS = ("calibration", "validation", "test")


# ------------------------------------------------------------------ loading
def load_raw(path):
    op = gzip.open if str(path).endswith(".gz") else open
    with op(path, "rt", encoding="utf-8") as stream:
        data = json.load(stream)
    if isinstance(data, dict):
        data = list(data.values())
    return data


def turns_of(item):
    """Flatten a question's haystack into turns, keeping session id, date, position and evidence flag."""
    out = []
    sessions = item["haystack_sessions"]
    sids = item.get("haystack_session_ids") or [f"s{i}" for i in range(len(sessions))]
    dates = item.get("haystack_dates") or [""] * len(sessions)
    gold_sessions = set(item.get("answer_session_ids") or [])
    for si, (sid, sess) in enumerate(zip(sids, sessions)):
        for ti, turn in enumerate(sess):
            if not isinstance(turn, dict):
                continue
            out.append(dict(id=f"{sid}#{ti}", session=sid, session_index=si, turn_index=ti,
                            date=dates[si] if si < len(dates) else "", role=turn.get("role", ""),
                            text=turn.get("content", "") or "",
                            evidence=bool(turn.get("has_answer", False)),
                            gold_session=sid in gold_sessions))
    return out


def assign_splits(items, seed=2026, shares=(.15, .25, .60)):
    """Split questions (not sessions) into calibration/validation/test, balanced within question type."""
    rng = np.random.default_rng(seed)
    by_type = defaultdict(list)
    for i, it in enumerate(items):
        by_type[it.get("question_type", "unknown")].append(i)
    split = {}
    for t, idx in sorted(by_type.items()):
        idx = list(idx); rng.shuffle(idx)
        n_cal = max(1, round(len(idx) * shares[0]))
        n_val = max(1, round(len(idx) * shares[1]))
        for k, i in enumerate(idx):
            split[i] = SPLITS[0] if k < n_cal else SPLITS[1] if k < n_cal + n_val else SPLITS[2]
    return split


# ------------------------------------------------------------------ checks
def evidence_report(items, user_only=False):
    per_q, per_type = [], defaultdict(list)
    sess_reuse = Counter()
    for it in items:
        ts = [t for t in turns_of(it) if not user_only or t["role"] == "user"]
        ev = [t for t in ts if t["evidence"]]
        row = dict(turns=len(ts), evidence_turns=len(ev),
                   evidence_sessions=len({t["session"] for t in ev}),
                   gold_sessions=len({t["session"] for t in ts if t["gold_session"]}),
                   labelled=bool(ev))
        per_q.append(row); per_type[it.get("question_type", "unknown")].append(row)
        for sid in {t["session"] for t in ts}:
            sess_reuse[sid] += 1
    return per_q, per_type, sess_reuse


def print_check(items, user_only):
    per_q, per_type, reuse = evidence_report(items, user_only)
    labelled = [r for r in per_q if r["labelled"]]
    ev = np.array([r["evidence_turns"] for r in labelled])
    print(f"\n=== GO / NO-GO CHECK ({'user turns only' if user_only else 'all turns'}) ===")
    print(f"questions: {len(per_q)}  with turn-level evidence labels: {len(labelled)} ({len(labelled)/max(len(per_q),1):.0%})")
    if not len(ev):
        print("  NO turn-level has_answer labels found -- this file cannot be used for turn-level evidence.")
        return
    print(f"turns per question: median {np.median([r['turns'] for r in per_q]):.0f}")
    print(f"evidence turns per question: median {np.median(ev):.0f}  mean {ev.mean():.1f}  "
          f"share needing >1: {np.mean(ev > 1):.0%}  >2: {np.mean(ev > 2):.0%}")
    print("  (CHECK 1: multiple labelled turns make multi-evidence retrieval measurable; one turn can contain multiple facts)")
    print(f"\n{'question type':28s} {'n':>4} {'ev turns (median)':>18} {'need >1':>8} {'ev sessions (median)':>21}")
    for t, rows in sorted(per_type.items()):
        e = np.array([r["evidence_turns"] for r in rows if r["labelled"]])
        if not len(e):
            print(f"{t:28s} {len(rows):4d} {'no labels':>18}")
            continue
        print(f"{t:28s} {len(rows):4d} {np.median(e):18.0f} {np.mean(e > 1):8.0%} "
              f"{np.median([r['evidence_sessions'] for r in rows if r['labelled']]):21.0f}")
    multi = sum(v > 1 for v in reuse.values())
    print(f"\nsessions appearing in more than one question's haystack: {multi} of {len(reuse)} "
          f"(shared filler; splits are by question, so a session can appear in two splits)")


def mean_or_none(values):
    return float(np.mean(values)) if len(values) else None


def summarize_checks(rows, budgets):
    out = dict(questions=len(rows),
               median_turns=float(np.median([r["turns"] for r in rows])) if rows else None,
               candidate_evidence_recall=mean_or_none([r["candidate_evidence_recall"] for r in rows]),
               candidate_evidence_fraction=mean_or_none([r["candidate_evidence_fraction"] for r in rows]),
               budgets={})
    for field in ["evidence_pairwise_cosines", "candidate_evidence_pairwise_cosines"]:
        known = [r for r in rows if field in r]
        out[field + "_distribution"] = distribution([x for r in known for x in r[field]])
        out[field + "_questions_with_pairs"] = sum(bool(r[field]) for r in known)
    out["cosine_aggregation"] = "Pooled within-question pairs; questions with more pairs have more weight. No cross-question pairs."
    for b in budgets:
        rb = [r["budgets"][str(b)] for r in rows]
        out["budgets"][str(b)] = {
            k: mean_or_none([r[k] for r in rb if r[k] is not None])
            for k in ["selected_count", "evidence_recall", "cost", "pairs_cosine_gt_09", "pairs_cosine_gt_08"]}
        out["budgets"][str(b)]["questions_with_selected_pairs"] = sum(r["pairs_cosine_gt_09"] is not None for r in rb)
        out["budgets"][str(b)].update(summarize_extended(rows, b))
    return out


def duplicate_check(items, split, encoder, budgets, pool, user_only, relevance, model_dir, revision, threads,
                    cache_dir=None, cache_batch_size=64):
    """CHECK 2: relevance selection only. Cache inference; retain original scoring rules."""
    from evaluate import keyword_tokens, bm25_scores, word_cost
    from trajectory_memory.magnitude import select
    from trajectory_memory.embedding_cache import CachedEncoder, DEFAULT_CACHE_DIR
    started = time.monotonic()
    val = [it for i, it in enumerate(items) if split[i] == "validation"]
    # Skip unusable cases before model loading, but retain the original LSA fit corpus.
    usable = []
    skipped = []
    for it in val:
        ts = [t for t in turns_of(it) if (not user_only or t["role"] == "user") and t["text"].strip()]
        if not ts or not any(t["evidence"] for t in ts):
            skipped.append(dict(question_id=it["question_id"], reason="no nonempty labelled evidence turns after role filtering"))
        else:
            usable.append((it, ts))
    rows, ev_cost, all_cost, cache_stats, encoder_meta = [], [], [], None, None
    if usable:
        if encoder == "minilm":
            from trajectory_memory.encoders import MiniLMEncoder
            enc = MiniLMEncoder(model_dir, revision, threads)
        else:
            from trajectory_memory.encoders import LSAEncoder
            enc = LSAEncoder(128, 2026)
            pool_texts = [t["text"] for it in val for t in turns_of(it) if t["text"].strip()]
            enc.fit(pool_texts[::max(1, len(pool_texts) // 20000)])
        encoder_meta = enc.metadata
        with CachedEncoder(enc, cache_dir or DEFAULT_CACHE_DIR, cache_batch_size) as cached:
            print(f"Persistent embedding cache: {cached.path}", flush=True)
            for progress, (it, ts) in enumerate(usable, 1):
                question_start = time.monotonic()
                print(f"\nQuestion {progress}/{len(usable)}: {it['question_id']} "
                      f"({it.get('question_type', 'unknown')}); {len(ts)} turns", flush=True)
                texts = [t["text"] for t in ts]
                E = cached.encode(texts).astype(np.float64)
                if relevance == "bm25":
                    docs = [keyword_tokens(x) for x in texts]
                    s = bm25_scores(it["question"], docs)
                    s = s / s.max() if s.max() > 0 else s
                else:
                    q = cached.encode([it["question"]])[0].astype(np.float64)
                    s = E @ q
                order = np.argsort(-s, kind="stable")[:pool]
                cost = np.array([word_cost(texts[i]) for i in order])
                sim = E[order] @ E[order].T
                gold = {t["id"] for t in ts if t["evidence"]}
                in_pool = {ts[i]["id"] for i in order}
                gold_indices = [i for i, t in enumerate(ts) if t["evidence"]]
                gold_turns = [ts[i] for i in gold_indices]
                candidate_turns = [ts[i] for i in order]
                row = dict(question_id=it["question_id"], question_type=it.get("question_type", "unknown"),
                           turns=len(ts), evidence_turns=len(gold),
                           candidate_ids=[ts[i]["id"] for i in order],
                           evidence_pairwise_cosines=cosine_pairs(E[gold_indices]),
                           candidate_evidence_pairwise_cosines=cosine_pairs(E[[i for i in order if ts[i]["evidence"]]]),
                           candidate_evidence_recall=float(np.mean([g in in_pool for g in gold])),
                           candidate_evidence_fraction=float(np.mean([ts[i]["id"] in gold for i in order])), budgets={})
                ev_cost += [word_cost(t["text"]) for t in ts if t["id"] in gold]
                all_cost += [word_cost(t["text"]) for t in ts]
                for b in budgets:
                    S = select(None, s[order], cost, b, method="relevance")["indices"]
                    pairs = [sim[S[a], S[c]] for a in range(len(S)) for c in range(a + 1, len(S))]
                    picked = [ts[order[i]]["id"] for i in S]
                    row["budgets"][str(b)] = dict(selected_ids=picked, selected_count=len(S),
                        cost=int(cost[S].sum()) if S else 0,
                        evidence_recall=float(np.mean([g in set(picked) for g in gold])),
                        pairs_cosine_gt_09=mean_or_none([x > .9 for x in pairs]),
                        pairs_cosine_gt_08=mean_or_none([x > .8 for x in pairs]))
                    # These annotations are consulted only after the existing
                    # relevance selector has produced its unchanged selection.
                    oracle, sessions = budget_diagnostics(gold_turns, candidate_turns, cost, S, b)
                    pair_values = np.clip(pairs, -1.0, 1.0).tolist()
                    row["budgets"][str(b)].update(oracle=oracle, sessions=sessions,
                        selected_pairwise_cosines=pair_values, selected_pairwise_cosine=distribution(pair_values))
                rows.append(row)
                print(f"Finished question {progress}/{len(usable)} in {time.monotonic() - question_start:.1f}s; "
                      f"{cached.encoded} new text embeddings saved this run", flush=True)
            cache_stats = cached.stats()
    overall = summarize_checks(rows, budgets)
    per_type = {t: summarize_checks([r for r in rows if r["question_type"] == t], budgets)
                for t in sorted({r["question_type"] for r in rows})}
    print(f"\n=== CHECK 2: redundancy among selected turns (validation, {encoder}+{relevance}, pool {pool}) ===")
    if rows:
        print(f"questions used: {len(rows)}  turns per question: median {overall['median_turns']:.0f}")
        print(f"turn length (cost units): all turns median {np.median(all_cost):.0f}, evidence turns median {np.median(ev_cost):.0f}")
        print(f"RETRIEVAL: gold turns reaching the pool: {overall['candidate_evidence_recall']:.1%}  "
              f"(share of pool that is evidence: {overall['candidate_evidence_fraction']:.1%})")
        pct = lambda x: "n/a" if x is None else f"{x:.1%}"
        for name, result in [("all types", overall)] + list(per_type.items()):
            print(f"  {name}: {result['questions']} questions")
            def show_quantiles(stats):
                return " ".join(f"p{p}=" + ("n/a" if stats[f"p{p}"] is None else f"{stats[f'p{p}']:.3f}")
                                for p in [50, 75, 90, 95, 99])
            gold_stats = result["evidence_pairwise_cosines_distribution"]
            print(f"    gold cosine pairs (pooled n={gold_stats['pairs']}): {show_quantiles(gold_stats)}")
            for b, r in result["budgets"].items():
                print(f"    budget {b}: selected {r['selected_count']:.1f}, recall {r['evidence_recall']:.3f}, "
                      f"oracle {r['oracle_recall']:.3f}, selection gap {r['oracle_selection_gap']:.3f}")
                print(f"      recall loss: retrieval {r['oracle_retrieval_loss']:.3f}, "
                      f"budget {r['oracle_unavoidable_budget_loss']:.3f}, selection {r['oracle_selection_gap']:.3f}")
                ps = r["selected_pairwise_cosine"]
                print(f"      selected cosine pairs (pooled n={ps['pairs']}): {show_quantiles(ps)}")
                print(f"      mean per-question pair fractions: >0.9 {pct(r['pairs_cosine_gt_09'])}, >0.8 {pct(r['pairs_cosine_gt_08'])}")
                ms = r["multi_session_evidence"]
                print(f"      gold spans multiple sessions: n={ms['questions']}; evidence-session coverage "
                      f"all {pct(ms['all_sessions_share'])}, some {pct(ms['some_sessions_share'])}, "
                      f"none {pct(ms['no_sessions_share'])}; all gold turns {pct(ms['all_gold_turns_share'])}")
                print(f"      multi-session mean coverage {pct(ms['mean_session_recall'])}, "
                      f"session oracle {pct(ms['mean_oracle_session_recall'])}; "
                      f"all sessions budget-feasible {pct(ms['all_sessions_budget_feasible_share'])}")
                print(f"      distinct selected sessions {r['selected_sessions']:.1f}; "
                      f"largest session share {pct(r['largest_selected_session_share'])}")
    else:
        print("No usable validation questions. No model inference was performed.")
    print("Near-duplicate thresholds are descriptive; low pair similarity alone is not a no-go decision for magnitude.")
    return dict(overall=overall, by_type=per_type, cases=rows, skipped=skipped,
                validation_question_ids=[it["question_id"] for it in val],
                median_turn_cost=float(np.median(all_cost)) if all_cost else None,
                median_evidence_turn_cost=float(np.median(ev_cost)) if ev_cost else None,
                embedding_cache=cache_stats, encoder=encoder_meta,
                elapsed_seconds=round(time.monotonic() - started, 2))


def write_report(path, value):
    """Replace reports atomically; embeddings are already committed independently."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent, delete=False) as f:
            temp = f.name
            json.dump(value, f, indent=2, ensure_ascii=False, allow_nan=False)
            f.flush()
            os.fsync(f.fileno())
        os.replace(temp, path)
    finally:
        if temp is not None and os.path.exists(temp):
            os.unlink(temp)
    print(f"Saved diagnostic report: {path}", flush=True)


# ------------------------------------------------------------------ export
def export(items, split, path, pool, user_only, drop_empty=True):
    out = []
    for i, it in enumerate(items):
        ts = [t for t in turns_of(it) if (not user_only or t["role"] == "user")]
        if drop_empty:
            ts = [t for t in ts if t["text"].strip()]
        if not any(t["evidence"] for t in ts):
            continue
        out.append(dict(id=it["question_id"], question=it["question"], answer=it.get("answer"),
                        type=it.get("question_type", "unknown"), date=it.get("question_date", ""),
                        split=split[i], pool_size=pool,
                        memories=[dict(id=t["id"], text=t["text"], role=t["role"], session=t["session"],
                                       session_index=t["session_index"], turn_index=t["turn_index"],
                                       date=t["date"], evidence=t["evidence"]) for t in ts]))
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    op = gzip.open if str(path).endswith(".gz") else open
    with op(path, "wt") as f:
        json.dump(out, f)
    n = Counter(c["split"] for c in out)
    print(f"\nwrote {path}: {len(out)} questions ({dict(n)}), "
          f"{sum(len(c['memories']) for c in out)} turns, "
          f"{sum(sum(m['evidence'] for m in c['memories']) for c in out)} evidence turns")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True, help="longmemeval_s.json / longmemeval_m.json / oracle file")
    p.add_argument("--check", action="store_true", help="run the go/no-go checks and stop")
    p.add_argument("--out", help="export raw cases (.json or .json.gz); not consumed by synthetic evaluate.py")
    from trajectory_memory.embedding_cache import DEFAULT_CACHE_DIR
    p.add_argument("--cache-dir", default=str(DEFAULT_CACHE_DIR), help="persistent cache shared across runs")
    p.add_argument("--cache-batch-size", type=int, default=64, help="unique texts per committed cache batch")
    p.add_argument("--report", help="diagnostic JSON path; default results/longmemeval_check_<timestamp>.json")
    p.add_argument("--pool", type=int, default=POOL_DEFAULT)
    p.add_argument("--budgets", type=int, nargs="+", default=BUDGETS_DEFAULT)
    p.add_argument("--types", nargs="+", help="keep only these question types (e.g. multi-session knowledge-update)")
    p.add_argument("--user-only", action="store_true", help="index user turns only (assistant turns dropped)")
    p.add_argument("--encoder", choices=["lsa", "minilm"], default="lsa", help="encoder for check 2")
    p.add_argument("--relevance", choices=["cosine", "bm25"], default="cosine", help="relevance for check 2")
    p.add_argument("--model-dir"); p.add_argument("--revision", default=None); p.add_argument("--onnx-threads", type=int, default=2)
    p.add_argument("--limit", type=int, help="use only the first N questions (quick trial)")
    a = p.parse_args()
    if a.pool < 1 or min(a.budgets) < 1 or a.cache_batch_size < 1 or a.onnx_threads < 1 or (a.limit is not None and a.limit < 1):
        p.error("pool, budgets, cache batch size, threads and limit must be positive")
    if a.report and not a.check:
        p.error("--report requires --check")
    for output in [a.report, a.out]:
        if output and Path(output).resolve() == Path(a.data).resolve():
            p.error("output must not overwrite the input dataset")

    items = load_raw(a.data)
    items = [it for it in items if not str(it.get("question_id", "")).endswith("_abs")]
    if a.types:
        items = [it for it in items if it.get("question_type") in a.types]
    if a.limit:
        items = items[:a.limit]
    print(f"loaded {len(items)} questions from {a.data} (abstention questions excluded)")
    print("types:", dict(Counter(it.get("question_type", "unknown") for it in items)))

    if not items:
        p.error("no questions remain after filtering")
    print_check(items, a.user_only)
    split = assign_splits(items)
    if a.check:
        from evaluate import DEFAULT_REVISION
        from trajectory_memory.data import digest
        from threadpoolctl import threadpool_limits
        with threadpool_limits(limits=1):
            result = duplicate_check(items, split, a.encoder, a.budgets, a.pool, a.user_only, a.relevance,
                                     a.model_dir, a.revision or DEFAULT_REVISION, a.onnx_threads,
                                     a.cache_dir, a.cache_batch_size)
        per_q, per_type, reuse = evidence_report(items, a.user_only)
        result.update(schema_version=2, experiment="longmemeval_relevance_diagnostic",
                      arguments=vars(a), data_sha256=digest(a.data),
                      token_unit="regex word-or-punctuation units; not LLM tokens",
                      evidence_check=dict(questions=per_q, by_type=dict(per_type),
                                          reused_sessions=sum(v > 1 for v in reuse.values())),
                      diagnostic_definitions={
                          "oracle": "Exact maximum labelled-turn recall within the existing candidate pool and whole-turn cost budget; all labelled turns have equal value.",
                          "loss_decomposition": "1 - actual recall = retrieval loss + unavoidable budget loss + selection gap. Means use equal question weights.",
                          "session_coverage": "A gold session counts as covered only when a selected turn in it has an evidence label; visiting a non-evidence turn does not count.",
                          "session_oracle": "Separate maximum number of gold sessions covered by one or more labelled turns under the budget; may choose a different set from the turn oracle.",
                          "cosine_percentiles": "Exact pooled within-question pair percentiles; no cross-question pairs. Raw pair values saved for report-only subset summaries.",
                          "undefined": "Empty pair sets and empty multi-session groups are null, not zero. Single-turn evidence has no pair distribution."},
                      limitations=["No magnitude selection or answer generation in this check.",
                                   "The oracle uses annotated turns, not semantic fact equivalence or answer correctness.",
                                   "Session concentration and cosine similarity are diagnostics, not evidence that diversity will improve retrieval.",
                                   "Question splits can share source sessions.",
                                   "Type filtering happens before split assignment, as in the original script.",
                                   "Pair similarities are averaged only over questions with at least two selected turns."])
        target = a.report or HERE / "results" / ("longmemeval_check_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ") + ".json")
        write_report(target, result)
    if a.out:
        export(items, split, a.out, a.pool, a.user_only)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted. Completed embedding batches remain on disk; repeat the command to reuse them.", flush=True)
        raise SystemExit(130)
