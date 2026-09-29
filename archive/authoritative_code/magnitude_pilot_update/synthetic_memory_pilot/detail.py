"""Breakdown for one result file.

  python detail.py results/minilm_c6.json
    -> results/minilm_c6.detail.json   per-type coverage, paired tokens-to-cover-all, diagnostics
    -> results/minilm_c6.picks.txt     readable picks where a method's outcome differs from relevance

Uses the persistent text cache and frozen lambdas from the result file; cache misses are encoded.
Budget-limited numbers are unchanged. Only the token comparison for MMR is changed: its early stop
is disabled, so every method is compared on the same questions.
"""
import argparse, json, sys
from collections import defaultdict
from pathlib import Path
import numpy as np
from threadpoolctl import threadpool_limits
from evaluate import build, embed, prepare_geometry, choose, metrics, BUDGETS, POOL, view, pool_size
from gen import make_dataset

QTYPE = {0: "1 fact: city", 1: "2 facts: major+uni", 2: "3 facts: best uni in my city?", 3: "2 facts: job+commute",
         4: "5 facts: profile", 5: "2 facts: advisor+year", 6: "1 fact: current uni (transfer)"}
qnum = lambda c: int(c["q"]["id"].split("_q")[1])


# ------------------------------------------------------------ pool texts
def attach_texts(cases, a):
    """Pools from evaluate.build already carry memory texts; older builds are rebuilt here (cosine only)."""
    if all("texts" in c for c in cases):
        return
    users = make_dataset(a.copies, tuple(a.users))
    M, Q, _ = embed(users, a)
    it, mi, qi = iter(cases), 0, 0
    for u in users:
        E = M[mi:mi + len(u["memories"])].astype(np.float64); QV = Q[qi:qi + len(u["questions"])].astype(np.float64)
        mi += len(u["memories"]); qi += len(u["questions"])
        city = u["questions"][0]["required"][0][1]
        for q, qv in zip(u["questions"], QV):
            c = next(it); s = E @ qv; order = np.argsort(-s, kind="stable")[:POOL]
            assert c["q"]["id"] == q["id"] and np.array_equal(c["rel"], s[order]), "pools differ from evaluate.py"
            c["texts"] = [u["memories"][k]["text"] for k in order]
            c["city"] = city


def label(c, i):
    parts = []
    for slot, val, cur in sorted(c["tags"][i]):
        if slot == "top": parts.append("best-uni ranking (" + ("my city" if val == c["city"] else "other city") + ")")
        elif slot == "university": parts.append("university (" + ("current" if cur else "old") + ")")
        else: parts.append(slot)
    return " + ".join(parts) if parts else "distractor"


# ------------------------------------------------------------ fair MMR
def mmr_full_order(c, lam, continuation):
    """Project MMR rule, but never stops. After scores reach <= 0 the next pick is chosen by
    'score' (highest MMR score) or 'per_token' (the project's rule, which then favours long memories)."""
    rel, cost, n, sim = c["rel"], c["cost"], c["n"], c["sim"]
    r = (rel - rel.min()) / max(np.ptp(rel), 1e-12)
    chosen, stop = [], None
    while len(chosen) < n:
        fits = [i for i in range(n) if i not in chosen]
        gains = lam * r - (1 - lam) * np.maximum(sim[:, chosen].max(axis=1), 0) if chosen else r.copy()
        j = max(fits, key=lambda i: (gains[i] / cost[i], rel[i], -i))
        if chosen and gains[j] <= 0:
            stop = len(chosen) if stop is None else stop
            if continuation == "score":
                j = max(fits, key=lambda i: (gains[i], rel[i], -i))
        chosen.append(j)
    return chosen, stop


def check_mmr_matches_project(test, lam, W):
    """With no budget limit, the project's MMR picks must be exactly the first picks of the no-stop MMR."""
    bad = 0
    for c in test:
        proj = choose("MMR", c, int(c["cost"].sum()), lam, W)
        full, stop = mmr_full_order(c, lam, "score")
        bad += proj != full[:len(proj)] or (stop is not None and len(proj) != stop)
    return bad


def tokens_until_covered(c, order):
    req = [tuple(x) + (True,) for x in c["q"]["required"]]
    if not all(any(t in tg for tg in c["tags"]) for t in req): return None, None
    got, spent = set(), 0
    for k, s in enumerate(order):
        got |= c["tags"][s]; spent += int(c["cost"][s])
        if all(t in got for t in req): return spent, k + 1
    return None, None


def paired(test, base, other, rng):
    both = [(c, x, y) for c, x, y in zip(test, base, other) if x is not None and y is not None]
    per_user, per_type = defaultdict(list), defaultdict(list)
    for c, x, y in both:
        per_user[c["user"]].append(y - x); per_type[QTYPE[qnum(c)]].append(y - x)
    d = np.array([np.mean(v) for v in per_user.values()]); boot = rng.choice(d, (2000, len(d))).mean(1)
    raw = np.array([y - x for _, x, y in both])
    return dict(n_questions=len(both), mean_relevance=float(np.mean([x for _, x, _ in both])),
                mean_method=float(np.mean([y for _, _, y in both])), diff=float(d.mean()),
                ci95=[float(np.quantile(boot, .025)), float(np.quantile(boot, .975))],
                share_fewer=float(np.mean(raw < 0)), share_same=float(np.mean(raw == 0)), share_more=float(np.mean(raw > 0)),
                never_complete=int(sum(v is None for v in other)),
                mean_diff_by_type={t: float(np.mean(v)) for t, v in per_type.items()})


# ------------------------------------------------------------ main
def main(res_path, cache_dir=None, cache_batch_size=64, model_dir=None):
    from evaluate import parser, HERE
    from trajectory_memory.embedding_cache import DEFAULT_CACHE_DIR
    res = json.load(open(res_path))
    saved = res.get("args", {})
    defaults = vars(parser().parse_args(["--copies", str(saved.get("copies", res.get("copies", 6))),
                                        "--encoder", saved.get("encoder", "lsa"), "--out", str(res_path)]))
    defaults.update(saved)
    # Saved absolute paths can belong to a different machine. Use the current
    # project's cache unless an explicit override is supplied.
    defaults.update(cache_dir=str(cache_dir or DEFAULT_CACHE_DIR), cache_batch_size=cache_batch_size)
    if model_dir is not None:
        defaults["model_dir"] = model_dir
    a = argparse.Namespace(**defaults)
    frozen = res["frozen"]
    cases, _ = build(a); W = prepare_geometry(cases)["W"]
    test = [c for c in cases if c["split"] == "test"]
    attach_texts(cases, a)
    rule, ftau = res["args"].get("pool_rule", "topk"), res.get("frozen_tau", {})
    V = lambda c, m: view(c, pool_size(c, rule, ftau.get(m)))       # each method's own candidate pool
    K = lambda c, m: pool_size(c, rule, ftau.get(m))
    rng = np.random.default_rng(71)
    out = dict(source=str(res_path), n_test_questions=len(test), tokens_to_complete={}, by_type={}, diagnostics={})

    # 1. tokens to cover all required facts, same questions for every method
    mismatches = check_mmr_matches_project([V(c, "MMR") for c in test], frozen["MMR"], W)
    assert mismatches == 0, f"no-stop MMR diverges from project MMR in {mismatches} selections"
    base = [tokens_until_covered(v, choose("relevance", v, int(v["cost"].sum()), 1.0, W))[0]
            for v in [V(c, "relevance") for c in test]]
    for m, l in frozen.items():
        if m == "MMR":
            vm = [V(c, "MMR") for c in test]
            orders = {k: [mmr_full_order(v, l, k) for v in vm] for k in ["score", "per_token"]}
            main_t = [tokens_until_covered(v, o)[0] for v, (o, _) in zip(vm, orders["score"])]
            sens_t = [tokens_until_covered(v, o)[0] for v, (o, _) in zip(vm, orders["per_token"])]
            before = [(k is not None) and (stop is None or k <= stop)
                      for v, (o, stop) in zip(vm, orders["score"]) for k in [tokens_until_covered(v, o)[1]]]
            entry = paired(test, base, main_t, rng)
            entry.update(no_early_stop=True, continuation="score",
                         covered_before_mmr_would_stop=float(np.mean([b for b, t in zip(before, main_t) if t is not None])),
                         sensitivity_per_token_continuation=paired(test, base, sens_t, np.random.default_rng(72)))  # separate stream keeps other CIs unchanged
        else:
            entry = paired(test, base, [tokens_until_covered(v, choose(m, v, int(v["cost"].sum()), l, W))[0]
                                        for v in [V(c, m) for c in test]], rng)
        out["tokens_to_complete"][m] = entry

    # 2. budget-limited coverage by question type (unchanged definition), keep picks for diagnostics
    picks = {}
    for m, l in frozen.items():
        acc = defaultdict(lambda: defaultdict(list))
        for c in test:
            v = V(c, m)
            for b in BUDGETS:
                S = choose(m, v, b, l, W); x = metrics(v, S)
                acc[QTYPE[qnum(c)]][f"coverage@{b}"].append(x["coverage"]); acc[QTYPE[qnum(c)]][f"complete@{b}"].append(x["complete"])
                if qnum(c) in (2, 6): picks[(c["q"]["id"], m, b)] = S
        out["by_type"][m] = {t: {k: float(np.mean(v)) for k, v in d.items()} | {"n_questions": len(d[f"coverage@{BUDGETS[0]}"])}
                             for t, d in acc.items()}

    # 3. diagnostics for the transfer and best-uni questions
    def nearest_pick(c, i, S, m=None):
        if m is not None and i >= K(c, m): return "filtered out by threshold", None
        if not S: return None, None
        k = max(S, key=lambda s: c["sim"][i, s]); return label(c, k), float(c["sim"][i, k])

    def fact_rows(c, fact):
        return [i for i in range(c["n"]) if fact in c["tags"][i]]

    diag = {"transfer": {}, "best_uni": {}}
    txt = []
    for m in frozen:
        for b in BUDGETS:
            T = dict(questions=0, current_covered=0, old_picked=0, stale_only=0, lost_vs_relevance=0,
                     lost_and_old_picked=0, lost_nearest_pick_labels=defaultdict(int), lost_nearest_pick_cosine=[])
            B = dict(questions=0, covered={"city": 0, "university": 0, "top": 0}, top_lost_vs_relevance=0,
                     top_lost_nearest_pick_labels=defaultdict(int), top_lost_nearest_pick_cosine=[], top_rank_in_pool=[])
            for c in test:
                k = qnum(c)
                if k not in (2, 6): continue
                S, R = picks[(c["q"]["id"], m, b)], picks[(c["q"]["id"], "relevance", b)]
                got = lambda X: set().union(*[c["tags"][s] for s in X]) if X else set()
                if k == 6:
                    cur = tuple(c["q"]["required"][0]) + (True,); old = ("university", c["old"], False)
                    hit, rhit = cur in got(S), cur in got(R)
                    T["questions"] += 1; T["current_covered"] += hit; T["old_picked"] += old in got(S)
                    T["stale_only"] += (not hit) and old in got(S)
                    if rhit and not hit:
                        T["lost_vs_relevance"] += 1; T["lost_and_old_picked"] += old in got(S)
                        i = min(fact_rows(c, cur)); lab, cos = nearest_pick(c, i, S, m)
                        T["lost_nearest_pick_labels"][lab] += 1
                        if cos is not None: T["lost_nearest_pick_cosine"].append(cos)
                else:
                    req = {x[0]: tuple(x) + (True,) for x in c["q"]["required"]}
                    B["questions"] += 1
                    for f in B["covered"]: B["covered"][f] += req[f] in got(S)
                    tops = fact_rows(c, req["top"])
                    if tops: B["top_rank_in_pool"].append(min(tops) + 1)
                    if req["top"] in got(R) and req["top"] not in got(S):
                        B["top_lost_vs_relevance"] += 1
                        lab, cos = nearest_pick(c, min(tops), S, m)
                        B["top_lost_nearest_pick_labels"][lab] += 1
                        if cos is not None: B["top_lost_nearest_pick_cosine"].append(cos)
                # readable picks, only where coverage differs from relevance
                if m != "relevance":
                    req = [tuple(x) + (True,) for x in c["q"]["required"]]
                    cs, cr = sum(t in got(S) for t in req), sum(t in got(R) for t in req)
                    if cs != cr:
                        need = ", ".join(f"{x[0]}={x[1]}" for x in c["q"]["required"])
                        txt.append(f"\n[{c['q']['id']}] {c['q']['question']}\n  needs: {need}   budget {b}")
                        for name, X, cov in [("relevance", R, cr), (m, S, cs)]:
                            txt.append(f"  {name}  covered {cov}/{len(req)}   (candidate pool: {K(c, name)} memories)")
                            for s in X:
                                txt.append(f"     #{s+1:<2} {label(c, s):38s} {c['texts'][s]}")
                        missed = [t for t in req if t in got(R) and t not in got(S)]
                        for t in missed:
                            i = min(fact_rows(c, t)); lab, cos = nearest_pick(c, i, S, m)
                            why = lab if cos is None else f"its nearest pick: {lab}, cosine {cos:.2f}"
                            txt.append(f"  -> {m} skipped #{i+1} ({label(c, i)}); {why}")
            for D, keys in [(T, ["lost_nearest_pick_cosine"]), (B, ["top_lost_nearest_pick_cosine", "top_rank_in_pool"])]:
                for key in keys:
                    v = D[key]; D[key] = dict(n=len(v), median=float(np.median(v)) if v else None)
            T["lost_nearest_pick_labels"] = dict(T["lost_nearest_pick_labels"])
            B["top_lost_nearest_pick_labels"] = dict(B["top_lost_nearest_pick_labels"])
            diag["transfer"][f"{m}@{b}"] = T; diag["best_uni"][f"{m}@{b}"] = B
    out["diagnostics"] = diag

    target = Path(res_path).with_suffix(".detail.json")
    json.dump(out, open(target, "w"), indent=1)
    Path(res_path).with_suffix(".picks.txt").write_text(
        "Picks where a method covers a different number of required facts than relevance-only.\n"
        "#N = relevance rank in the 32-memory pool.\n" + "\n".join(txt) + "\n")
    print(f"wrote {target} and {Path(res_path).with_suffix('.picks.txt')}")
    report(out)


def fmt(x):
    return '-' if x is None else f'{x:.2f}'


def report(out):
    t = out["tokens_to_complete"]
    ns = {e["n_questions"] for m, e in t.items() if m != "relevance"}
    print(f"\nExtra tokens to cover all required facts vs relevance (n = questions both can fully cover, of {out['n_test_questions']}):")
    if len(ns) > 1:
        print("  note: methods use different candidate pools here, so n differs and rows are not on identical question sets")
    for m, e in t.items():
        if m == "relevance": continue
        extra = ""
        if e.get("no_early_stop"):
            s = e["sensitivity_per_token_continuation"]
            extra = (f"   [MMR without early stop; covered before it would have stopped: {e['covered_before_mmr_would_stop']:.0%}; "
                     f"with per-token continuation instead: {s['diff']:+.1f}]")
        print(f"  {m:24s} {e['diff']:+.1f} [{e['ci95'][0]:+.1f}, {e['ci95'][1]:+.1f}]  n={e['n_questions']}{extra}")
    for b in BUDGETS:
        print(f"\nTransfer question @ {b} tokens: current uni covered / old uni picked / lost vs relevance (of those, old uni picked; nearest pick to the skipped current memory)")
        for key, T in out["diagnostics"]["transfer"].items():
            m, bb = key.rsplit("@", 1)
            if int(bb) != b: continue
            print(f"  {m:24s} {T['current_covered']}/{T['questions']}  old {T['old_picked']}  lost {T['lost_vs_relevance']} "
                  f"(old picked {T['lost_and_old_picked']}; nearest {T['lost_nearest_pick_labels']}, median cos {fmt(T['lost_nearest_pick_cosine']['median'])})")
        print(f"Best-uni question @ {b} tokens: city / university / ranking covered; ranking lost vs relevance (nearest pick to the skipped ranking memory)")
        for key, B in out["diagnostics"]["best_uni"].items():
            m, bb = key.rsplit("@", 1)
            if int(bb) != b: continue
            cv = B["covered"]
            print(f"  {m:24s} {cv['city']}/{cv['university']}/{cv['top']} of {B['questions']}  lost {B['top_lost_vs_relevance']} "
                  f"(nearest {B['top_lost_nearest_pick_labels']}, median cos {fmt(B['top_lost_nearest_pick_cosine']['median'])}; "
                  f"ranking memory median pool rank {B['top_rank_in_pool']['median']})")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Detail a saved synthetic result with persistent embedding reuse")
    p.add_argument("result")
    p.add_argument("--cache-dir")
    p.add_argument("--cache-batch-size", type=int, default=64)
    p.add_argument("--model-dir")
    args = p.parse_args()
    if args.cache_batch_size < 1:
        p.error("cache batch size must be positive")
    with threadpool_limits(limits=1):
        main(args.result, args.cache_dir, args.cache_batch_size, args.model_dir)
