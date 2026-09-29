"""Synthetic student-memory experiment: relevance, MMR and magnitude variants V0-V5.

Usage (from inside this folder):
  python evaluate.py --copies 6 --encoder lsa    --out results/lsa_c6.json
  python evaluate.py --copies 6 --encoder minilm --out results/minilm_c6.json

Only the encoder changes between runs. Token costs are always counted in regex
word units, so budgets (32 / 64) mean the same thing for both encoders.
"""
import argparse, hashlib, json, os, re, sys, time
from collections import defaultdict
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))  # this folder must sit inside history_magnitude_pilot/
from threadpoolctl import threadpool_limits
from trajectory_memory.magnitude import distances, Geometry, SchurState, log_weights, calibrate_profiles, select, magnitude
from gen import make_dataset

POOL, BUDGETS = 32, [32, 64]
VARIANTS = ["V0 as run", "V1 minus single credit", "V2 excess ratio", "V3 slope-weighted", "V4 one scale per pool", "V5 per-step rescale"]
LAMS = [.95, .9, .8, .7, .5, .3]
DEFAULT_REVISION = "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"
POOL_RULES = ["topk", "absolute", "margin"]
DEFAULT_TAUS = {"absolute": [.1, .2, .3, .4, .5, .6], "margin": [.05, .1, .15, .2, .3, .4]}


def keyword_tokens(text):
    from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
    return [w for w in re.findall(r"[a-z0-9']+", text.lower()) if w not in ENGLISH_STOP_WORDS]


def bm25_scores(question, docs, k1=1.2, b=.75):
    """BM25 of one question against one student's memory store (IDF from that store, no labels)."""
    import math
    from collections import Counter
    q = set(keyword_tokens(question))
    avg = max(np.mean([len(d) for d in docs]), 1e-9)
    df = Counter(w for d in docs for w in set(d))
    idf = {w: math.log(1 + (len(docs) - n + .5) / (n + .5)) for w, n in df.items()}
    out = np.zeros(len(docs))
    for i, d in enumerate(docs):
        tf = Counter(d)
        out[i] = sum(idf[w] * tf[w] * (k1 + 1) / (tf[w] + k1 * (1 - b + b * len(d) / avg)) for w in q if w in tf)
    return out


def word_cost(text):
    return max(1, len(re.findall(r"\w+|[^\w\s]", text)))


# ---------------------------------------------------------------- embeddings
def embed(users, a):
    """Persist unique texts across runs; never reuse vectors from a different encoder."""
    from trajectory_memory.embedding_cache import CachedEncoder, DEFAULT_CACHE_DIR
    mem_texts = [m["text"] for u in users for m in u["memories"]]
    q_texts = [q["question"] for u in users for q in u["questions"]]
    t0 = time.time()
    if a.encoder == "lsa":
        from trajectory_memory.encoders import LSAEncoder
        enc = LSAEncoder(64, 2026)
        enc.fit(mem_texts)  # retain the original fitted corpus, including repetitions
    else:
        from trajectory_memory.encoders import MiniLMEncoder
        enc = MiniLMEncoder(a.model_dir, a.revision, a.onnx_threads)
    with CachedEncoder(enc, getattr(a, "cache_dir", DEFAULT_CACHE_DIR),
                       getattr(a, "cache_batch_size", 64)) as cached:
        M, Q = cached.encode(mem_texts), cached.encode(q_texts)
        meta = dict(enc.metadata, encode_seconds=round(time.time() - t0, 1),
                    embedding_cache=cached.stats())
    return M, Q, meta


def build(a):
    users = make_dataset(a.copies, tuple(a.users))
    M, Q, meta = embed(users, a)
    cases, mi, qi = [], 0, 0
    for u in users:
        n_m, n_q = len(u["memories"]), len(u["questions"])
        E = M[mi:mi + n_m].astype(np.float64); QV = Q[qi:qi + n_q].astype(np.float64)
        mi += n_m; qi += n_q
        cost = np.array([word_cost(m["text"]) for m in u["memories"]])
        relevance = getattr(a, "relevance", "cosine")     # result files from before this option are cosine
        docs = [keyword_tokens(m["text"]) for m in u["memories"]] if relevance == "bm25" else None
        city = u["questions"][0]["required"][0][1]
        for q, qv in zip(u["questions"], QV):
            if relevance == "bm25":
                s = bm25_scores(q["question"], docs)
                s = s / s.max() if s.max() > 0 else s          # 0..1 per question; selection is unchanged by this scaling
            else:
                s = E @ qv
            order = np.argsort(-s, kind="stable")[:POOL]
            cases.append(dict(user=u["user"], split=u["split"], q=q, emb=E[order], rel=s[order], cost=cost[order],
                              tags=[{tuple(t) for t in u["memories"][k]["tags"]} for k in order], old=u["old_university"],
                              texts=[u["memories"][k]["text"] for k in order], city=city))
    assert mi == len(M) and qi == len(Q)
    return cases, meta


# ------------------------------------------------------------------ geometry
def prepare_geometry(cases):
    cal = [c for c in cases if c["split"] == "calibration"]
    raw = [distances(c["emb"]) for c in cal[::3]]
    pos = np.concatenate([d[np.triu_indices(len(d), 1)] for d in raw]); unit = float(np.median(pos[pos > 1e-10]))
    conf = calibrate_profiles([d / unit for d in raw], 129, 65, .002)
    dense = np.array(conf["dense_scales"]); logt = np.log(dense)
    for c in cases:
        geo = Geometry.make(distances(c["emb"], unit), dense)
        n = len(c["rel"])
        slope = np.maximum(np.gradient(geo.full, logt), 0); slope /= max(slope.sum(), 1e-12)
        c.update(geo=geo, n=n, slope=slope, jstar=int(np.argmin(np.abs(geo.full - (1 + n) / 2))), sim=c["emb"] @ c["emb"].T,
                 logt=logt)
    return dict(unit=unit, low=float(dense[0]), high=float(dense[-1]), W=log_weights(dense))


# ---------------------------------------------------------- threshold pools
def pool_size(c, rule, tau):
    """Pools are prefixes of the relevance ranking (capped at POOL), so index i is the same memory in every pool."""
    if rule == "topk" or tau is None:
        return c["n"]
    cut = tau if rule == "absolute" else c["rel"][0] - tau
    return max(1, int(np.sum(c["rel"] >= cut)))          # always keep at least the top memory


def view(c, k):
    """Candidate pool = first k memories of the full pool, with its own magnitude geometry."""
    if k == c["n"]:
        return c
    cache = c.setdefault("_views", {})
    if k not in cache:
        bank = c["geo"].bank[:, :k, :k]
        full = np.array([magnitude(b) for b in bank])
        slope = np.maximum(np.gradient(full, c["logt"]), 0); slope /= max(slope.sum(), 1e-12)
        v = {key: val for key, val in c.items() if key != "_views"}
        v.update(emb=c["emb"][:k], rel=c["rel"][:k], cost=c["cost"][:k], tags=c["tags"][:k], sim=c["sim"][:k, :k], n=k,
                 geo=Geometry(bank, full, c["geo"].scales), slope=slope, jstar=int(np.argmin(np.abs(full - (1 + k) / 2))))
        cache[k] = v
    return cache[k]


def pool_stats(cases, rule, tau):
    size, recall, precision, fallback = [], [], [], []
    for c in cases:
        k = pool_size(c, rule, tau)
        req = [tuple(r) + (True,) for r in c["q"]["required"]]
        size.append(k)
        recall.append(np.mean([any(t in tg for tg in c["tags"][:k]) for t in req]))
        precision.append(np.mean([bool(set(req) & tg) for tg in c["tags"][:k]]))
        if rule == "absolute":
            fallback.append(c["rel"][0] < tau)
    out = dict(mean_pool_size=float(np.mean(size)), median_pool_size=float(np.median(size)),
               required_facts_in_pool=float(np.mean(recall)), share_of_pool_useful=float(np.mean(precision)))
    if fallback:
        out["share_no_memory_passed_kept_top1"] = float(np.mean(fallback))
    return out


def geom(variant, c, st, S, fits, W):
    g = st.gains(); full, n = c["geo"].full, c["n"]
    if variant == "V0 as run": return n * (W @ (g / full[:, None]))
    if not S: return np.zeros(n)
    if variant == "V1 minus single credit": return n * (W @ (g / full[:, None]))
    if variant == "V2 excess ratio": return (n - 1) * (W @ (g / np.maximum(full - 1, 1e-9)[:, None]))
    if variant == "V3 slope-weighted": return c["slope"] @ g
    if variant == "V4 one scale per pool": return g[c["jstar"]]
    raw = W @ (g / full[:, None]); lo, hi = raw[fits].min(), raw[fits].max()
    return (raw - lo) / (hi - lo) if hi - lo > 1e-12 else np.zeros(n)


def choose(method, c, budget, lam, W):
    rel, cost, n = c["rel"], c["cost"], c["n"]
    if method == "relevance": return select(None, rel, cost, budget, method="relevance")["indices"]
    if method == "MMR": return select(None, rel, cost, budget, method="mmr", mmr_lambda=lam, similarity=c["sim"])["indices"]
    r = (rel - rel.min()) / max(np.ptp(rel), 1e-12)
    st, S, spent, total = SchurState(c["geo"].bank), [], 0, 0.0
    while True:
        fits = [x for x in range(n) if x not in S and spent + cost[x] <= budget]
        if not fits: break
        gains = lam * r + (1 - lam) * geom(method, c, st, S, fits, W)
        j = max(fits, key=lambda x: (gains[x] / cost[x], rel[x], -x))
        S.append(j); spent += cost[j]; total += gains[j]; st.add(j)
    fits = [x for x in range(n) if cost[x] <= budget]
    first = n * (W @ (1 / ((1 + 1e-8) * c["geo"].full))) if method == "V0 as run" else 0.0
    if fits:
        j = max(fits, key=lambda x: (lam * r[x] + (1 - lam) * first, rel[x], -x))
        if lam * r[j] + (1 - lam) * first > total + 1e-10: S = [j]
    return S


# ------------------------------------------------------------------- metrics
def metrics(c, S):
    req = [tuple(x) + (True,) for x in c["q"]["required"]]
    got = set().union(*[c["tags"][s] for s in S]) if S else set()
    covered = [t in got for t in req]
    seen, dup = set(), 0
    for s in S:
        if c["tags"][s] and c["tags"][s] <= seen: dup += 1
        seen |= c["tags"][s]
    stale = int(not covered[0] and ("university", c["old"], False) in got) if c["q"]["transfer_q"] else None
    return dict(coverage=float(np.mean(covered)), complete=int(all(covered)), dup=dup / len(S) if S else 0.0,
                tokens=int(c["cost"][S].sum()) if S else 0, stale_only=stale)


def tokens_to_complete(c, method, lam, W):
    req = [tuple(x) + (True,) for x in c["q"]["required"]]
    if not all(any(t in tg for tg in c["tags"]) for t in req): return None
    S = choose(method, c, int(c["cost"].sum()), lam, W)
    got, spent = set(), 0
    for s in S:
        got |= c["tags"][s]; spent += int(c["cost"][s])
        if all(t in got for t in req): return spent
    return None


def user_mean(rows):
    g = defaultdict(list)
    for u, v in rows: g[u].append(v)
    return float(np.mean([np.mean(v) for v in g.values()]))


def keyword_diagnostics(cases):
    """How often keyword matching can see the memories a question needs (validation users; labels for diagnosis only)."""
    val = [c for c in cases if c["split"] == "validation"]
    facts_findable, no_overlap = [], []
    for c in val:
        req = [tuple(r) + (True,) for r in c["q"]["required"]]
        matched = c["rel"] > 0
        facts_findable.append(np.mean([any(t in c["tags"][i] for i in np.where(matched)[0]) for t in req]))
        no_overlap.append(not matched.any())
    return dict(required_facts_findable_by_keywords=float(np.mean(facts_findable)),
                questions_with_no_keyword_match=float(np.mean(no_overlap)),
                note="computed within the 32-memory pool")


def encoder_diagnostics(cases, W):
    """Does the encoder place paraphrases of the same fact close together? (labels used for diagnosis only)"""
    val = [c for c in cases if c["split"] == "validation"]
    same, diff, other, seen = [], [], [], set()
    for c in val:
        if c["user"] in seen: continue
        seen.add(c["user"])
        for x in range(c["n"]):
            for y in range(x + 1, c["n"]):
                tx, ty = c["tags"][x], c["tags"][y]
                (same if tx and ty and tx & ty else diff if tx and ty else other).append(c["sim"][x, y])
    stat = lambda v: dict(n=len(v), median=float(np.median(v)), p90=float(np.quantile(v, .9)))
    out = dict(cosine_same_fact=stat(same), cosine_different_facts=stat(diff), cosine_with_distractors=stat(other))
    out["reachable_required_facts"] = float(np.mean([np.mean([any(tuple(r) + (True,) in t for t in c["tags"]) for r in c["q"]["required"]]) for c in val]))
    for b in BUDGETS:
        out[f"relevance_duplicate_picks@{b}"] = float(np.mean([metrics(c, choose("relevance", c, b, 1, W))["dup"] for c in val]))
    return out


# ---------------------------------------------------------------------- main
def run(a):
    t0 = time.time()
    cases, enc_meta = build(a)
    info = prepare_geometry(cases); W = info["W"]
    val = [c for c in cases if c["split"] == "validation"]; test = [c for c in cases if c["split"] == "test"]
    lams = [a.force_lambda] if a.force_lambda is not None else a.lams
    taus = [None] if a.pool_rule == "topk" else a.taus
    strict = lambda t: 0 if t is None else (t if a.pool_rule == "absolute" else -t)   # stricter pool wins ties
    configs = [("relevance", 1.0)] + [("MMR", l) for l in lams] + [(v, l) for v in VARIANTS for l in lams]
    valscore = {}
    for t in taus:
        vv = [view(c, pool_size(c, a.pool_rule, t)) for c in val]
        for m, l in configs:
            valscore[(m, l, t)] = user_mean([(c["user"], metrics(c, choose(m, c, b, l, W))["coverage"]) for c in vv for b in BUDGETS])
    best = {m: max([(l, t) for (mm, l, t) in valscore if mm == m], key=lambda lt: (valscore[(m, *lt)], strict(lt[1]), lt[0]))
            for m in ["relevance", "MMR"] + VARIANTS}
    frozen = {m: best[m][0] for m in best}
    frozen_tau = {m: best[m][1] for m in best}
    key = lambda m, l, t: f"{m}|{l}" if t is None else f"{m}|{l}|tau={t}"
    res = dict(args=vars(a), encoder=enc_meta, calibration={k: v for k, v in info.items() if k != "W"},
               encoder_diagnostics=encoder_diagnostics(cases, W),
               validation={key(m, l, t): v for (m, l, t), v in valscore.items()}, frozen=frozen, test={})
    if a.relevance == "bm25":
        res["keyword_diagnostics"] = keyword_diagnostics(cases)
    if a.pool_rule != "topk":
        res["frozen_tau"] = frozen_tau
        res["pool_stats_validation"] = {str(t): pool_stats(val, a.pool_rule, t) for t in taus}
    base = None
    for m, l in frozen.items():
        rows, t = [], frozen_tau[m]
        for c in test:
            v = view(c, pool_size(c, a.pool_rule, t))
            for b in BUDGETS:
                x = metrics(v, choose(m, v, b, l, W)); x.update(user=c["user"], budget=b, case=c["q"]["id"]); rows.append(x)
        summary = {}
        for b in BUDGETS:
            rb = [r for r in rows if r["budget"] == b]
            summary[str(b)] = {k: user_mean([(r["user"], r[k]) for r in rb]) for k in ["coverage", "complete", "dup", "tokens"]}
            summary[str(b)]["stale_only"] = float(np.mean([r["stale_only"] for r in rb if r["stale_only"] is not None]))
        cov = {(r["case"], r["budget"]): (r["user"], r["coverage"]) for r in rows}
        base = base or cov
        diffs = defaultdict(list)
        for k, (u, v) in cov.items(): diffs[u].append(v - base[k][1])
        d = np.array([np.mean(v) for v in diffs.values()])
        boot = np.random.default_rng(71).choice(d, (2000, len(d))).mean(1)
        summary["coverage_diff_vs_relevance"] = [float(d.mean()), float(np.quantile(boot, .025)), float(np.quantile(boot, .975))]
        if a.pool_rule != "topk":
            summary["pool"] = pool_stats(test, a.pool_rule, t) | {"tau": t}
        res["test"][m] = summary
    res["elapsed_seconds"] = round(time.time() - t0, 1)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(res, open(a.out, "w"), indent=1)
    mode = f" (all methods forced to lambda={a.force_lambda})" if a.force_lambda is not None else ""
    if a.relevance == "bm25":
        mode += " [relevance: bm25 keywords]"
    if a.pool_rule != "topk":
        mode += f" [pool rule {a.pool_rule}, tau per method: {frozen_tau}]"
    print(f"{a.encoder} copies={a.copies}{mode} done in {res['elapsed_seconds']}s -> {a.out}", flush=True)


def parser():
    p = argparse.ArgumentParser()
    p.add_argument("--copies", type=int, required=True, choices=range(1, 7))
    p.add_argument("--encoder", choices=["lsa", "minilm"], default="lsa")
    p.add_argument("--out", required=True)
    p.add_argument("--model-dir", help="offline folder with tokenizer.json and onnx/model.onnx (MiniLM only)")
    p.add_argument("--revision", default=DEFAULT_REVISION, help="Hugging Face revision for MiniLM download")
    p.add_argument("--onnx-threads", type=int, default=2)
    p.add_argument("--cache-dir", default=str(HERE / "cache"))
    p.add_argument("--cache-batch-size", type=int, default=64, help="unique texts committed per cache batch")
    p.add_argument("--users", type=int, nargs=3, default=[15, 50, 100], metavar=("CAL", "VAL", "TEST"))
    p.add_argument("--lams", type=float, nargs="+", default=LAMS,
                   help="lambda grid searched on validation (default: %(default)s); e.g. add 0")
    p.add_argument("--force-lambda", type=float, default=None,
                   help="skip the search: run MMR and V0-V5 at this lambda (relevance stays 1.0). Use a separate --out file")
    p.add_argument("--pool-rule", choices=POOL_RULES, default="topk",
                   help="topk: 32 most relevant (default). absolute: relevance >= tau. margin: relevance >= best - tau. "
                        "Threshold pools are capped at 32 and always keep the top memory; tau is chosen on validation per method")
    p.add_argument("--taus", type=float, nargs="+", default=None, help="threshold grid (defaults depend on --pool-rule)")
    p.add_argument("--relevance", choices=["cosine", "bm25"], default="cosine",
                   help="cosine: question vs memory embedding (default). bm25: keyword match, scaled 0-1 per question. "
                        "The embeddings are still used for magnitude and MMR distances; every method uses the same relevance")
    return p


def check_args(a):
    if a.cache_batch_size < 1:
        raise SystemExit("--cache-batch-size must be positive")
    if a.pool_rule != "topk" and a.taus is None:
        a.taus = DEFAULT_TAUS[a.pool_rule]
        if a.relevance == "bm25" and a.pool_rule == "absolute":
            a.taus = [1e-6, .2, .4, .6]          # 1e-6 = any shared word with the question
    if a.pool_rule == "topk" and a.taus is not None:
        raise SystemExit("--taus only applies with --pool-rule absolute or margin")
    for l in ([a.force_lambda] if a.force_lambda is not None else a.lams):
        if not 0 <= l <= 1:
            raise SystemExit(f"lambda must be in [0, 1], got {l}")


if __name__ == "__main__":
    args = parser().parse_args()
    check_args(args)
    with threadpool_limits(limits=1):
        run(args)
