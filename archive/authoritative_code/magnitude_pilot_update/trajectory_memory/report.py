"""Self-contained evidence browser and exportable numerical profile figure."""
import json
from pathlib import Path
import numpy as np
from .data import dump

NAMES = {
    "current_relevance": "Current question only",
    "history_relevance": "History + relevance",
    "concat_relevance": "Concatenated recent history",
    "permuted_history_relevance": "Reversed-history control",
    "history_mmr": "History + MMR",
    "history_fixed_magnitude": "History + fixed magnitude",
    "history_profile_magnitude": "History + magnitude profile",
    "history_dense_magnitude": "History + dense profile audit",
    "history_profile_diagnostic": "Profile diagnostic (lambda=0.5)",
    "history_dense_diagnostic": "Dense diagnostic (lambda=0.5)",
}


def make_report(out, data, queries, labels, corpus, summary, details, audits,
                comparisons, scales, frozen, manifest):
    test = [q for q in queries if q["split"] == "test"]
    errors = np.array([a["selected_profile_interpolation_error"] for a in audits])
    integral_errors = np.array([a["integral_error"] for a in audits])
    same = np.mean([a["same_selected_set"] for a in audits])
    convergence = json.loads((out / "dense_convergence.json").read_text())
    meta = manifest["data_manifest"]
    lines = ["# History-dependent retrieval and magnitude selection", "",
             f"Encoder: **{meta['encoder']['name']}**. Corpus: {len(corpus)} static passages.",
             f"Test: {len(test)} real recorded turns from {manifest['test_conversations']} conversations.",
             f"Budget unit: {meta['encoder']['token_unit']}.", "",
             "This is an exploratory internal split of TopiOCQA dev. The default corpus pools all released dev reference passages globally, including passages for other conversations. It is annotation-derived and much easier than the full Wikipedia corpus. Gold is never inserted into a query's candidate pool.", "",
             "History contains recorded prior answers. Current answers, topics, rationales and gold IDs are kept outside retrieval. This is teacher-forced conversational retrieval, not autonomous dialogue or personal long-term memory.", "",
             "## Held-out results", "",
             "Follow-up turns only. Equal weight per conversation; average turns within each conversation. First turns and source-topic transition breakdowns are in the CSV files.", "",
             "| Method | Budget | Evidence recall | Candidate recall | Mean cost |",
             "|---|---:|---:|---:|---:|"]
    for r in summary:
        lines.append(f"| {NAMES[r['method']]} | {r['budget']} | {r['evidence_recall']:.3%} | {r['candidate_recall']:.3%} | {r['tokens']:.1f} |")
    lines += ["", "## Paired differences", "", "Exploratory conversation-cluster bootstrap, 2,000 resamples; both budgets combined.", ""]
    for c in comparisons:
        low, high = c["ci95"]
        lines.append(f"- {NAMES[c['a']]} minus {NAMES[c['b']]}: {100*c['mean_difference']:+.2f} percentage points; 95% interval [{100*low:+.2f}, {100*high:+.2f}].")
    lines += ["", "## Magnitude profile and numerical checks", "",
              f"The finite log-scale integral uses **{len(scales['scales'])} adaptive nodes**, checked against **{len(scales['dense_scales'])} dense nodes**.",
              f"Scales range from {scales['scales'][0]:.6g} to {scales['scales'][-1]:.6g}, after calibration distance normalization.",
              f"Calibration interpolation error: {scales['calibration_max_interpolation_error']:.6g}; requested tolerance: {scales['tolerance']:.6g}; met: {scales['tolerance_met']}.",
              f"Held-out selected-profile interpolation error: median {np.median(errors):.6g}, max {np.max(errors):.6g}.",
              f"Held-out integral error: median {np.median(integral_errors):.6g}, max {np.max(integral_errors):.6g}.",
              f"Adaptive and dense selection agree on {same:.2%} of selected sets.",
              "Numerical selection audits use the explicit lambda=0.5 diagnostic, so geometry is active even if validation chooses relevance-only. This diagnostic is not the validation-selected method.",
              f"Dense-grid convergence: {len(convergence)} comparisons against {2*len(scales['dense_scales'])-1} nodes; max integral gap {max((r['integral_gap'] for r in convergence), default=0):.6g}; all selected sets identical: {all(r['same_selected_set'] for r in convergence)}.",
              "These are empirical finite-grid checks, not a guarantee for every continuous scale, unseen geometry or possible subset. Dense profiles are numerical references, not exact integrals.", "",
              "## Frozen choices", "",
              f"History weight: {frozen['history_beta']}. Profile relevance weight: {frozen['profile_lambda']}. Fixed scale: {frozen['fixed_scale']:.5g}; fixed relevance weight: {frozen['fixed_lambda']}.",
              "Zero history weight and relevance-only selection are allowed to win validation. No test result chooses a hyperparameter.", "",
              "## Interpretation boundaries", "",
              "- History vs current-only measures the value of incorporating recorded conversational context.",
              "- Profile vs history-relevance/MMR/fixed-magnitude measures the incremental contribution of multiscale selection.",
              "- Adaptive nodes approximate the same fixed log-scale integral; they are not a learned narrowing/broadening controller.",
              "- Topic switch/return tags come from source document annotations and are used only for post-hoc breakdowns. They do not validate semantic scope detection.",
              "- Missing an annotated passage is not proof that a different selected passage cannot answer the question. Inspect the evidence browser.",
              "- No LLM answer quality or evidence entailment was measured in this retrieval run.", "",
              "Open **explorer.html** for actual question histories, selected memories, missed gold passages and magnitude curves.",
              "See **profile_audit.csv**, **metrics.csv**, **candidate_pools.json**, **selections.json**, and **frozen_validation.json** for exact decisions.", "",
              f"Run time excluding preparation: {manifest['elapsed_seconds']:.1f} seconds."]
    (out / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    bundle = dict(queries=test, labels={q["id"]: labels[q["id"]] for q in test},
                  corpus={m["id"]: m for m in corpus}, rows=details, profiles=audits,
                  scales=scales, summary=summary, names=NAMES, encoder=meta["encoder"], frozen=frozen)
    serialized = json.dumps(bundle, ensure_ascii=False).replace("</", "<\\/")
    (out / "explorer.html").write_text(HTML.replace("__DATA__", serialized), encoding="utf-8")
    plot_profiles(out, scales, audits)


def plot_profiles(out, scales, audits):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    # Deterministic first examples, not cherry-picked successes.
    examples = audits[::max(1, len(audits) // 4)][:4]
    fig, axes = plt.subplots(2, 2, figsize=(10, 6.5), constrained_layout=True)
    x = np.array(scales["dense_scales"])
    ix = np.array(scales["indices"])
    for ax, case in zip(axes.flat, examples):
        y = np.array(case["selected_profile"]) / np.array(case["full_profile"])
        ax.semilogx(x, y, color="#0c6f80", label="Dense selected / candidate magnitude")
        ax.semilogx(x[ix], y[ix], "o--", markersize=3, color="#db8b22", label="Adaptive nodes")
        ax.set(title=f"{case['case']} · budget {case['budget']}", xlabel="Scale (log axis)", ylabel="Magnitude ratio", ylim=(0, 1.02))
        ax.grid(alpha=.15)
    axes[0, 0].legend(fontsize=7, loc="best")
    fig.suptitle("Magnitude profiles: numerical integration audit", fontsize=15)
    fig.savefig(out / "magnitude_profiles.png", dpi=160)
    plt.close(fig)


HTML = r'''<!doctype html>
<html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Conversation memory selection — evidence explorer</title>
<style>
:root{color-scheme:light;--ink:#173b43;--teal:#087c86;--muted:#617179;--gold:#c08429}
*{box-sizing:border-box}body{margin:0;background:#f5f7f6;color:var(--ink);font:15px/1.55 system-ui,sans-serif}
main{max-width:1250px;margin:auto;padding:32px}h1{font-size:30px;letter-spacing:-1px;margin:0 0 8px}h2{font-size:19px}
.lead{color:var(--muted);max-width:920px}section,.card{background:white;border:1px solid #dde5e3;border-radius:10px;padding:20px;margin:18px 0}
.controls{display:grid;grid-template-columns:2fr 1fr 100px;gap:12px}label{display:block;font-size:12px;font-weight:700;color:var(--muted)}
select{width:100%;padding:10px;border:1px solid #bbc8c6;border-radius:6px;background:white;color:var(--ink)}
.grid{display:grid;grid-template-columns:1fr 1fr;gap:22px}.stats{display:flex;gap:30px;flex-wrap:wrap}.stat{font-size:24px;font-weight:650}.stat small{display:block;font-size:12px;color:var(--muted);font-weight:400}
.passage{border-left:4px solid #c7d2d0;padding:12px 16px;margin:14px 0;background:#f9fbfa}.gold{border-color:var(--teal);background:#eef9f6}.missing{border-color:var(--gold);background:#fff9ed}
code{font-size:11px;overflow-wrap:anywhere}.badge{font-size:11px;text-transform:uppercase;color:var(--teal);font-weight:700}pre{white-space:pre-wrap;font:13px/1.6 system-ui}table{border-collapse:collapse;width:100%;font-size:13px}td,th{text-align:left;padding:7px;border-bottom:1px solid #e4ebe8}
svg{width:100%;height:auto}a{color:var(--teal)}details summary{cursor:pointer}p{margin:8px 0}#q{font-size:20px;font-weight:600}.warning{font-size:13px;color:#805e26}
@media(max-width:800px){main{padding:15px}.grid,.controls{grid-template-columns:1fr}}
</style><main>
<p class="badge">Real TopiOCQA conversations · inspectable evidence</p><h1>What did the selector remember?</h1>
<p class="lead">Compare the actual passages selected for a conversational question. Green marks an annotated supporting passage; amber marks missing annotated evidence. An unmarked passage may still contain a valid answer.</p>
<p id="encoder" class="warning"></p><details><summary>Aggregate follow-up results</summary><div id="aggregate"></div></details>
<section><div class="controls"><div><label for="case">Conversation turn</label><select id="case"></select></div><div><label for="method">Selection method</label><select id="method"></select></div><div><label for="budget">Budget</label><select id="budget"></select></div></div>
<p id="q"></p><p id="policy" class="warning"></p><div id="stats" class="stats"></div></section>
<div class="grid"><div><section><h2>Conversation prefix</h2><div id="history"></div></section><section><h2>Annotated evidence</h2><div id="gold"></div></section></div>
<div><section><h2>Selected memories</h2><div id="selected"></div></section><section><h2>Profile integration audit</h2><p class="lead">Fixed lambda=0.5 diagnostic subset, independent of the method selector above. Teal: dense reference. Orange: adaptive sampling. Y = selected / candidate magnitude.</p><svg id="chart" viewBox="0 0 560 250" role="img" aria-label="Selected to candidate magnitude ratio across scales"></svg><p id="audit"></p></section></div></div>
<section><details><summary>Greedy decisions and candidate ranking</summary><pre id="trace"></pre></details></section>
<p class="lead">Restricted annotation-derived passage corpus, not full Wikipedia. Recorded prior answers are available. No current gold answers or topic labels enter retrieval. Evidence retrieval results do not establish LLM answer quality.</p>
</main><script>
const D=__DATA__;
const el=id=>document.getElementById(id), esc=x=>String(x).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
el('encoder').textContent=D.encoder.name+' · '+D.encoder.token_unit+(D.encoder.pretrained?'':' · This run uses lexical/latent geometry, not a pretrained semantic encoder.');
el('case').innerHTML=D.queries.map(q=>`<option value="${esc(q.id)}">${esc(q.id)} · ${esc(D.labels[q.id].transition)} · ${esc(q.question)}</option>`).join('');
el('method').innerHTML=Object.entries(D.names).map(([k,v])=>`<option value="${k}">${v}</option>`).join('');el('method').value='history_profile_diagnostic';
el('budget').innerHTML=[...new Set(D.rows.map(r=>r.budget))].map(b=>`<option>${b}</option>`).join('');
el('aggregate').innerHTML='<table><tr><th>Method</th><th>Budget</th><th>Recall</th><th>Candidate recall</th></tr>'+D.summary.map(r=>`<tr><td>${D.names[r.method]}</td><td>${r.budget}</td><td>${(100*r.evidence_recall).toFixed(1)}%</td><td>${(100*r.candidate_recall).toFixed(1)}%</td></tr>`).join('')+'</table>';
function passage(id,cls,badge){const p=D.corpus[id];return p?`<div class="passage ${cls}"><span class="badge">${badge}</span><p><b>${esc(p.title)}</b></p><code>${esc(id)} · ${p.cost} cost units</code><p>${esc(p.text)}</p></div>`:`<p>Passage ${esc(id)} is absent from the indexed corpus.</p>`;}
function render(){
 const id=el('case').value,b=+el('budget').value,m=el('method').value,q=D.queries.find(q=>q.id===id),l=D.labels[id],r=D.rows.find(r=>r.case===id&&r.budget===b&&r.method===m);
 const lambda=m.includes('diagnostic')?.5:m==='history_fixed_magnitude'?D.frozen.fixed_lambda:m.includes('magnitude')?D.frozen.profile_lambda:null;
 el('policy').textContent=lambda===null?'':m.includes('diagnostic')?`Fixed diagnostic: 50% relevance / 50% magnitude. The validation-selected profile method uses relevance weight ${D.frozen.profile_lambda}.`:`Validation-selected relevance weight ${lambda}; magnitude weight ${1-lambda}.`+(lambda===1?' Geometry is disabled in this selected policy.':'');
 el('q').textContent=q.question;el('stats').innerHTML=`<div class="stat">${(100*r.evidence_recall).toFixed(0)}%<small>Annotated evidence recalled</small></div><div class="stat">${r.selected_count}<small>Selected passages</small></div><div class="stat">${r.tokens}/${b}<small>Memory cost</small></div><div class="stat">${(100*r.candidate_recall).toFixed(0)}%<small>Candidate recall</small></div>`;
 el('history').innerHTML=q.history.length?q.history.map(h=>`<p class="badge">Turn ${h.turn}</p><p><b>User:</b> ${esc(h.question)}</p><p><b>Recorded answer:</b> ${esc(h.answer)}</p>`).join(''):'<p>First turn: no prior history.</p>';
 el('selected').innerHTML=r.selected_ids.map(id=>passage(id,l.gold_ids.includes(id)?'gold':'',l.gold_ids.includes(id)?'Annotated support':'Selected')).join('')||'<p>No passage fits the budget.</p>';
 el('gold').innerHTML=l.gold_ids.map(id=>passage(id,r.selected_ids.includes(id)?'gold':'missing',r.selected_ids.includes(id)?'Recovered':r.candidate_ids.includes(id)?'Retrieved, then omitted':'Not in candidate pool')).join('')+`<details><summary>Reference answers (evaluation only)</summary><pre>${esc(JSON.stringify(l.answers,null,2))}</pre></details>`;
 el('trace').textContent=JSON.stringify({trace:r.trace,candidates:r.candidate_ids.map((id,i)=>({id,relevance:r.relevance[i]}))},null,2);
 const a=D.profiles.find(p=>p.case===id&&p.budget===b),x=D.scales.dense_scales,ratio=a.selected_profile.map((y,i)=>y/a.full_profile[i]),lo=Math.log(x[0]),hi=Math.log(x[x.length-1]);
 const px=i=>48+480*(Math.log(x[i])-lo)/(hi-lo),py=y=>205-175*y,pts=ratio.map((y,i)=>`${px(i)},${py(y)}`).join(' '),ap=D.scales.indices.map(i=>`${px(i)},${py(ratio[i])}`).join(' ');
 el('chart').innerHTML=`<path d="M48 25V205H530" fill="none" stroke="#9badaa"/>`+[0,.5,1].map(y=>`<text x="14" y="${py(y)+4}" font-size="12">${y}</text><line x1="48" y1="${py(y)}" x2="530" y2="${py(y)}" stroke="#edf1ef"/>`).join('')+`<polyline points="${pts}" fill="none" stroke="#087c86" stroke-width="3"/><polyline points="${ap}" fill="none" stroke="#d59432" stroke-dasharray="4 4"/>`+D.scales.indices.map(i=>`<circle cx="${px(i)}" cy="${py(ratio[i])}" r="3" fill="#d59432"/>`).join('')+`<text x="48" y="225" font-size="12">${x[0].toPrecision(3)}</text><text x="475" y="225" font-size="12">${x[x.length-1].toPrecision(3)}</text><text x="230" y="245" font-size="12">Scale (log axis)</text>`;
 el('audit').textContent=`${D.scales.scales.length} adaptive / ${x.length} dense nodes. Integral error ${a.integral_error.toExponential(2)}. Same selected set as dense selection: ${a.same_selected_set?'yes':'no'}.`;
}
['case','method','budget'].forEach(id=>el(id).addEventListener('change',render));render();
</script></html>'''
