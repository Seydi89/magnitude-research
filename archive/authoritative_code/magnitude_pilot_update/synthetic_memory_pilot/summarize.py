"""Print all results in results/ side by side (every method).

Usage: python summarize.py [results_dir]

Runs are grouped by copies per fact. Within a group, each result file is one run:
  "minilm"                  main run (lambda chosen on validation)
  "minilm lam=0"            every method forced to lambda 0 (--force-lambda 0)
  "minilm grid+0"           lambda chosen on validation from a custom grid (--lams ...)
"""
import json, sys
from pathlib import Path

ORDER = ["relevance", "MMR", "V0 as run", "V1 minus single credit", "V2 excess ratio",
         "V3 slope-weighted", "V4 one scale per pool", "V5 per-step rescale"]
DEFAULT_LAMS = [.95, .9, .8, .7, .5, .3]


def run_label(args):
    lab = args["encoder"]
    if args.get("relevance", "cosine") == "bm25":
        lab += " bm25"
    if args.get("pool_rule", "topk") != "topk":
        lab += f" {args['pool_rule']}"
    if args.get("force_lambda") is not None:
        return f"{lab} lam={args['force_lambda']:g}"
    lams = args.get("lams", DEFAULT_LAMS)
    if sorted(lams) != sorted(DEFAULT_LAMS):
        extra = sorted(set(lams) - set(DEFAULT_LAMS))
        return f"{lab} grid" + ("+" + ",".join(f"{x:g}" for x in extra) if extra else "-custom")
    return lab


root = Path(sys.argv[1] if len(sys.argv) > 1 else "results")
runs, files = {}, {}
for f in sorted(root.glob("*.json")):
    if f.name.endswith(".detail.json"):
        continue
    r = json.load(open(f))
    if not all(k in r for k in ["args", "test", "frozen", "encoder_diagnostics"]):
        continue  # LongMemEval diagnostic reports may share this directory.
    detail = f.with_suffix(".detail.json")
    r["_detail"] = json.load(open(detail)) if detail.exists() else None
    key = (run_label(r["args"]), r["args"]["copies"])
    if key in runs:
        print(f"warning: {f.name} and {files[key].name} are the same kind of run; showing {f.name}")
    runs[key], files[key] = r, f
if not runs:
    sys.exit(f"no result files in {root}")

labels = sorted({l for l, _ in runs}, key=lambda l: (l.split()[0], len(l.split()), l))
stale = []   # result files whose detail file predates the fair MMR comparison
for copies in sorted({c for _, c in runs}):
    present = [l for l in labels if (l, copies) in runs]
    print(f"\n{'=' * 30} {copies} copies per fact {'=' * 30}")
    print("Encoder check (cosine between memories; higher same-fact vs different-fact gap = duplicates visible):")
    shown = set()
    for l in present:
        e = l.split()[0] + (" bm25" if " bm25" in l else "")
        if e in shown: continue
        shown.add(e)
        d = runs[(l, copies)]["encoder_diagnostics"]
        print(f"  {e:11s} same fact {d['cosine_same_fact']['median']:.2f} | different facts {d['cosine_different_facts']['median']:.2f} | "
              f"distractors {d['cosine_with_distractors']['median']:.2f} | facts reachable in pool {d['reachable_required_facts']:.0%} | "
              f"relevance duplicate picks @32 {d['relevance_duplicate_picks@32']:.0%} @64 {d['relevance_duplicate_picks@64']:.0%}")
        k = runs[(l, copies)].get("keyword_diagnostics")
        if k:
            print(f"  {'':11s} keywords: needed facts findable by keywords {k['required_facts_findable_by_keywords']:.0%} | "
                  f"questions with no keyword match {k['questions_with_no_keyword_match']:.0%}")
    for l in present:
        r = runs[(l, copies)]
        t2c = (r["_detail"] or {}).get("tokens_to_complete", {})
        how = "lambda forced for every method" if "lam=" in l else "frozen lambda from validation"
        if r.get("frozen_tau"):
            how += "; relevance threshold (tau) chosen on validation per method"
        print(f"\n  [{l}]  test, {how}   ({files[(l, copies)].name})")
        thr = bool(r.get("frozen_tau"))
        print(f"  {'method':24s} {'lam':>4} " + (f"{'tau':>5} {'pool':>5} {'facts in pool':>13} " if thr else "") + f"{'cov@32':>7} {'cov@64':>7} {'full@32':>8} {'full@64':>8} {'dup@32':>7} "
              f"{'coverage diff vs relevance [95% CI]':>36} {'extra tokens to cover all [95% CI]':>36}")
        for m in ORDER:
            t = r["test"][m]; a, b = t["32"], t["64"]; d = t["coverage_diff_vs_relevance"]
            tok = t2c.get(m)
            old_mmr = bool(tok) and m == "MMR" and not tok.get("no_early_stop")
            if old_mmr:
                stale.append(files[(l, copies)])
            tok_s = "" if not tok else (f"{tok['diff']:+.1f} [{tok['ci95'][0]:+.1f}, {tok['ci95'][1]:+.1f}]" + (" *" if old_mmr else ""))
            p = t.get("pool", {})
            pool_s = (f"{p['tau']:>5g} {p['mean_pool_size']:5.1f} {p['required_facts_in_pool']:13.0%} " if thr else "")
            print(f"  {m:24s} {r['frozen'][m]:>4g} " + pool_s + f"{a['coverage']:7.3f} {b['coverage']:7.3f} {a['complete']:8.3f} {b['complete']:8.3f} "
                  f"{a['dup']:7.2f} {f'{d[0]:+.3f} [{d[1]:+.3f}, {d[2]:+.3f}]':>36} {tok_s:>36}")
    if len(present) > 1:
        print("\n  Coverage difference vs relevance, runs side by side:")
        for m in ORDER[1:]:
            print(f"  {m:24s} " + "   ".join(
                f"{l}: {runs[(l, copies)]['test'][m]['coverage_diff_vs_relevance'][0]:+.3f} "
                f"(lam {runs[(l, copies)]['frozen'][m]:g})" for l in present))
if stale:
    print("\n* Old detail file for: " + ", ".join(f.stem for f in stale) +
          ". MMR stopped early there, so its token comparison used easier questions.")
    print("  Re-run:  " + "  ".join(f"python detail.py {f}" for f in stale))
