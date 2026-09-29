"""Evaluation-only diagnostics. Gold labels never enter retrieval or selection."""
from collections import Counter, defaultdict

import numpy as np


PERCENTILES = (50, 75, 90, 95, 99)


def cosine_pairs(embeddings):
    """All distinct within-question pairs; inputs use the existing normalized encoder."""
    x = np.asarray(embeddings, dtype=np.float64)
    if len(x) < 2:
        return []
    values = (x @ x.T)[np.triu_indices(len(x), 1)]
    return np.clip(values, -1.0, 1.0).tolist()


def distribution(values):
    values = np.asarray(values, dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError("Nonfinite similarity in diagnostic")
    return dict(pairs=len(values), mean=float(values.mean()) if len(values) else None,
                **{f"p{p}": float(np.percentile(values, p)) if len(values) else None for p in PERCENTILES})


def cheapest_prefix(options, budget):
    """Exact maximum count with positive costs and one unit of value per option."""
    used, chosen = 0, []
    for cost, key in sorted(options, key=lambda v: (v[0], str(v[1]))):
        if cost <= 0:
            raise ValueError("Positive diagnostic costs required")
        if used + cost > budget:
            break  # every later option costs at least as much
        used += int(cost)
        chosen.append(key)
    return chosen, used


def budget_diagnostics(gold_turns, candidates, costs, selected, budget):
    """Exact labelled-turn ceiling and evidence-backed session coverage.

    The turn oracle maximizes the number of labelled turns, not answer quality.
    The session oracle separately maximizes the number of gold sessions with at
    least one selected labelled turn; these objectives can choose different sets.
    """
    gold = {t["id"]: t for t in gold_turns}
    if not gold or len(candidates) != len(costs):
        raise ValueError("Gold turns and aligned candidate costs are required")
    if any(c <= 0 for c in costs) or budget < 0 or len(selected) != len(set(selected)):
        raise ValueError("Positive costs, nonnegative budget and unique selected indices required")
    if any(i < 0 or i >= len(candidates) for i in selected) or sum(costs[i] for i in selected) > budget:
        raise ValueError("Selected turns must fit the candidate pool and budget")
    ids = [t["id"] for t in candidates]
    if len(ids) != len(set(ids)) or len(gold) != len(gold_turns):
        raise ValueError("Turn IDs must be unique within each question")
    picked = {ids[i] for i in selected}
    hit = picked & gold.keys()
    reachable = set(ids) & gold.keys()
    best, oracle_cost = cheapest_prefix([(int(costs[i]), mid) for i, mid in enumerate(ids) if mid in gold], budget)
    n_gold = len(gold)
    recall = len(hit) / n_gold
    candidate_recall = len(reachable) / n_gold
    oracle_recall = len(best) / n_gold
    if len(hit) > len(best):
        raise ValueError("Selection exceeds oracle; check costs, budget and selected indices")
    oracle = dict(recall=oracle_recall, selected_gold_ids=best, cost=oracle_cost,
                  gold_turns_reachable=len(reachable), gold_turns_affordable=len(best),
                  all_gold_turns_fit=(len(best) == n_gold),
                  retrieval_loss=1.0 - candidate_recall,
                  unavoidable_budget_loss=candidate_recall - oracle_recall,
                  selection_gap=oracle_recall - recall)

    by_session = defaultdict(set)
    for mid, t in gold.items():
        by_session[str(t["session"])].add(mid)
    gold_sessions = set(by_session)
    covered_sessions = {str(gold[mid]["session"]) for mid in hit}
    reachable_sessions = {str(gold[mid]["session"]) for mid in reachable}
    fully_covered = {sid for sid, mids in by_session.items() if mids <= picked}
    selected_counts = Counter(str(candidates[i]["session"]) for i in selected)
    minimum_cost = {}
    for mid, cost in zip(ids, costs):
        if mid in gold:
            sid = str(gold[mid]["session"])
            minimum_cost[sid] = min(minimum_cost.get(sid, float("inf")), int(cost))
    session_best, _ = cheapest_prefix([(c, sid) for sid, c in minimum_cost.items()], budget)
    n_sessions = len(gold_sessions)
    session = dict(gold_sessions=n_sessions, gold_session_ids=sorted(gold_sessions),
                   selected_sessions=len(selected_counts), selected_session_counts=dict(selected_counts),
                   largest_selected_session_share=max(selected_counts.values()) / len(selected) if selected else None,
                   selected_gold_sessions_any_turn=len(gold_sessions & selected_counts.keys()),
                   evidence_sessions_reachable=len(reachable_sessions),
                   evidence_sessions_covered=len(covered_sessions),
                   evidence_session_recall=len(covered_sessions) / n_sessions,
                   evidence_sessions_fully_covered=len(fully_covered),
                   all_evidence_sessions_covered=(covered_sessions == gold_sessions),
                   all_gold_turns_covered=(len(hit) == n_gold),
                   evidence_session_state="all" if covered_sessions == gold_sessions else "some" if covered_sessions else "none",
                   oracle_evidence_session_recall=len(session_best) / n_sessions,
                   evidence_session_selection_gap=(len(session_best) - len(covered_sessions)) / n_sessions,
                   per_gold_session=[dict(session=sid, gold_turns=len(mids),
                                          candidate_gold_turns=len(mids & reachable),
                                          selected_gold_turns=len(mids & hit))
                                     for sid, mids in sorted(by_session.items())])
    return oracle, session


def summarize_extended(rows, budget):
    """Question-macro losses and pooled within-question cosine distributions."""
    entries = [r["budgets"][str(budget)] for r in rows]
    available = [e for e in entries if "oracle" in e and "sessions" in e]
    mean = lambda values: float(np.mean(values)) if values else None
    out = dict(questions_with_diagnostics=len(available))
    for key in ["recall", "retrieval_loss", "unavoidable_budget_loss", "selection_gap", "all_gold_turns_fit"]:
        out["oracle_" + key] = mean([e["oracle"][key] for e in available])
    pair_rows = [e for e in entries if "selected_pairwise_cosines" in e]
    out["selected_pairwise_cosine"] = distribution([x for e in pair_rows for x in e["selected_pairwise_cosines"]])
    out["questions_with_selected_cosine_pairs"] = sum(bool(e["selected_pairwise_cosines"]) for e in pair_rows)
    for key in ["selected_sessions", "largest_selected_session_share", "evidence_session_recall",
                "oracle_evidence_session_recall", "evidence_session_selection_gap"]:
        out[key] = mean([e["sessions"][key] for e in available if e["sessions"][key] is not None])
    multi = [e["sessions"] for e in available if e["sessions"]["gold_sessions"] > 1]
    out["multi_session_evidence"] = dict(
        questions=len(multi),
        all_sessions_share=mean([s["evidence_session_state"] == "all" for s in multi]),
        some_sessions_share=mean([s["evidence_session_state"] == "some" for s in multi]),
        no_sessions_share=mean([s["evidence_session_state"] == "none" for s in multi]),
        all_gold_turns_share=mean([s["all_gold_turns_covered"] for s in multi]),
        mean_session_recall=mean([s["evidence_session_recall"] for s in multi]),
        mean_oracle_session_recall=mean([s["oracle_evidence_session_recall"] for s in multi]),
        all_sessions_budget_feasible_share=mean([s["oracle_evidence_session_recall"] == 1.0 for s in multi]))
    return out
