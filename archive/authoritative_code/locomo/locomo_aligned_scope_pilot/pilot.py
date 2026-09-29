"""Aligned, controlled LoCoMo question-scope diagnostic. Not natural dialogue or official QA scores."""
import argparse,csv,hashlib,json,re,sys,importlib.metadata
from pathlib import Path
import numpy as np
from scipy.spatial.distance import pdist
from core import *
from episodes import build,label_signal

def dump(path,obj):Path(path).write_text(json.dumps(obj,indent=2))
def writecsv(path,rows):
 if not rows:return
 with Path(path).open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def bootstrap(rows,a,b):
 pairs={}
 for r in rows:pairs.setdefault((r['case'],r['budget']),{})[r['method']]=r
 groups={}
 for p in pairs.values():groups.setdefault(p[a]['group'],[]).append(p[a]['recall']-p[b]['recall'])
 v=np.array([np.mean(x) for x in groups.values()]);rng=np.random.default_rng(31)
 return dict(mean=float(v.mean()),ci95=np.quantile(rng.choice(v,(2000,len(v)),replace=True).mean(1),[.025,.975]).tolist(),groups=len(v))

def main(a):
 out=Path(a.out)
 if out.exists() and any(out.iterdir()):raise SystemExit('Use a new, empty output directory.')
 out.mkdir(parents=True,exist_ok=True)
 raw=json.loads(Path(a.data).read_text())
 if len(raw)<7:raise ValueError('Need at least seven independent conversations.')
 from sentence_transformers import SentenceTransformer
 model=SentenceTransformer(a.model,device=a.device,revision=a.revision)
 tok=model.tokenizer
 # Split at original whitespace boundaries. Keep all original text and provenance.
 def chunks(text):
  words=re.findall(r'\S+\s*',text);result=[];current=''
  for word in words:
   if current and len(tok.encode(current+word,add_special_tokens=True))>180:
    result.append(current);current=''
   current+=word
  if current:result.append(current)
  if any(len(tok.encode(t,add_special_tokens=True))>model.max_seq_length for t in result):raise ValueError('Unsplit oversized token sequence')
  return result
 convs=[];texts=set();skip=[]
 ordered=sorted(raw,key=lambda c:str(c['sample_id']))
 for ci,c in enumerate(ordered):
  split='calibration' if ci<3 else 'validation' if ci<6 else 'test'
  turns=[];sessions=[]
  for key in sorted((k for k in c['conversation'] if re.fullmatch(r'session_\d+',k)),key=lambda k:int(k.split('_')[1])):
   session=c['conversation'][key];date=c['conversation'].get(key+'_date_time','')
   for turn in session:
    text=turn.get('text','')
    if not text.strip():continue
    parts=chunks(text)
    # Speaker is explicit in model input; date remains metadata to avoid time driving scope geometry.
    ps=[f"{turn['speaker']}: {s}" for s in parts]
    if any(len(tok.encode(t,add_special_tokens=True))>model.max_seq_length for t in ps):raise ValueError('Speaker prefix exceeds encoder limit')
    texts.update(ps)
    turns.append(dict(id=turn['dia_id'],text=text,speaker=turn['speaker'],date=date,session=key,parts=ps))
   sessions.append(len(turns))
  qas=[]
  for qi,q in enumerate(c['qa']):
   evidence=q.get('evidence',[])
   if int(q.get('category',0)) not in [1,4] or not evidence:continue
   if not all(isinstance(e,str) for e in evidence):continue
   ids={t['id'] for t in turns}
   if not set(evidence)<=ids:
    skip.append(dict(group=c['sample_id'],qa=qi,reason='Unresolved evidence IDs'));continue
   texts.add(q['question']);qas.append(dict(index=qi,question=q['question'],gold=sorted(set(evidence)),category=q['category']))
  convs.append(dict(group=str(c['sample_id']),split=split,turns=turns,sessions=sessions,qas=qas))
 texts=sorted(texts)
 key=hashlib.sha256(json.dumps([texts,a.model,a.revision]).encode()).hexdigest();cache=Path(a.cache);cache.mkdir(parents=True,exist_ok=True)
 cp=cache/(key+'.npz')
 if cp.exists():embs=np.load(cp)['emb']
 else:
  if any(len(tok.encode(t,add_special_tokens=True))>model.max_seq_length for t in texts):raise ValueError('Encoder input too long')
  embs=model.encode(texts,normalize_embeddings=True,batch_size=64,show_progress_bar=True);np.savez_compressed(cp,emb=embs)
 emb=dict(zip(texts,embs));cost={t:len(tok.encode(t,add_special_tokens=False))+1 for t in texts}
 cases,checkpoints,blind,diagnostics=build(convs,emb,cost,a)
 human={}
 if a.labels:
  for r in csv.DictReader(open(a.labels)):
   if r['label_A_to_B'] and r['label_B_to_C']:
    if r['checkpoint'] in human:raise ValueError('One adjudicated row per case required')
    human[r['checkpoint']]=[r['label_A_to_B'].strip().lower(),r['label_B_to_C'].strip().lower()]
  needed={c['id'] for c in cases if c['split'] in ['validation','test']}
  if not needed<=set(human):raise ValueError('Complete validation/test transition labels required for the human controller.')
 writecsv(out/'scope_annotation_blind.csv',blind);writecsv(out/'checkpoint_counts.csv',diagnostics);dump(out/'skipped.json',skip)
 cal=[c for c in cases if c['split']=='calibration'];val=[c for c in cases if c['split']=='validation'];test=[c for c in cases if c['split']=='test']
 if not cal or not val or not test:raise ValueError('Empty split after eligibility filtering')
 ds=np.concatenate([pdist(c['x']) for c in cal]);unit=float(np.median(ds[ds>1e-10]))
 cf=[f for f in checkpoints if f['split']=='calibration']
 scale=max(float(np.std([f['trend'] for f in cf])),1e-6);uscale=max(float(np.std([f['unweighted'] for f in cf])),1e-6)
 mean=float(np.mean([f['level'] for f in cf]));std=max(float(np.std([f['level'] for f in cf])),1e-6)
 # Deadband is descriptive only; primary controller uses continuous drift.
 threshold=max(1e-10,.25*scale)
 for f in checkpoints:
  f['predicted_transitions']=['broadening' if d>threshold else 'narrowing' if d < -threshold else 'stable' for d in f['deltas']]
  f['predicted_scope_change']='broadening' if f['trend']>threshold else 'narrowing' if f['trend'] < -threshold else 'stable'
 dump(out/'scope_features.json',checkpoints)
 for c in cases:c['k']=kernels(c['x'],unit)
 def omega(c,name):
  if name=='uniform':return np.ones(6)/6
  if name.startswith('fixed_'):return np.eye(6)[int(name[-1])]
  kind,sign,strength=name.split(':');f=c['feat']
  if kind=='constructed':u=label_signal(c['constructed_labels'],f['beta'])
  elif kind=='human':u=label_signal(human[c['id']],f['beta'])
  else:u=f['trend']/scale if kind=='trend' else f['unweighted']/uscale if kind=='unweighted' else (f['level']-mean)/std
  return scale_weights(u,int(sign),float(strength))
 def evaluate(c,name,b,lam,custom=None):
  w=custom if custom is not None else omega(c,name) if name not in ['relevance','mmr'] else np.ones(6)/6
  ix,spent,trace=select(c['k'],np.array([m['cost'] for m in c['candidates']]),c['rel'],b,w,lam,name if name in ['relevance','mmr'] else None)
  selected={c['candidates'][j]['id'] for j in ix}
  hit=[e for e,v in c['required'].items() if set(v)<=selected]
  r=dict(case=c['id'],checkpoint=c['checkpoint'],group=c['group'],kind=c['kind'],constructed_direction=c['constructed_direction'],method=name,budget=b,tokens=spent,recall=len(hit)/len(c['gold']),
   candidate_recall=len(c['reachable'])/len(c['gold']),conditional_recall=len(hit)/len(c['reachable']) if c['reachable'] else None,
   complete=int(len(hit)==len(c['gold'])),selected_ids=[c['candidates'][j]['id'] for j in ix],weights=w.tolist(),trace=trace)
  return r
 # All scale policies share lambda, chosen using uniform weights on validation only.
 lambdas=[0.,.25,.5,.75,1.];lscores={l:float(np.mean([evaluate(c,'uniform',b,l)['recall'] for c in val if c['kind']=='primary' for b in a.budgets])) for l in lambdas}
 lam=max(lambdas,key=lambda l:lscores[l])
 fixed=['uniform']+[f'fixed_{i}' for i in range(6)]
 kinds=['trend','unweighted','level','constructed']+(['human'] if human else [])
 grid=[f'{kind}:{sign}:{st}' for kind in kinds for sign in [-1,1] for st in [.5,1.,2.]]
 scores={n:float(np.mean([evaluate(c,n,b,lam)['recall'] for c in val if c['kind']=='primary' for b in a.budgets])) for n in fixed+grid}
 best_fixed=max(fixed,key=lambda n:scores[n]);chosen={k:max([n for n in grid if n.startswith(k+':')],key=lambda n:scores[n]) for k in kinds}
 frozen=dict(lambda_value=lam,lambda_validation=lscores,scale_validation=scores,best_fixed=best_fixed,chosen=chosen,distance_unit=unit,trend_scale=scale,unweighted_scale=uscale,level_mean=mean,level_std=std,deadband=threshold)
 dump(out/'frozen_validation.json',frozen)
 methods=list(dict.fromkeys(['relevance','mmr']+fixed+list(chosen.values())+['trend:-1:1.0','trend:1:1.0']))
 rows=[];rng=np.random.default_rng(a.seed);shuffle={};representative={c['checkpoint']:c for c in test}
 for kind in ['primary','control']:
  cpids=sorted(c['checkpoint'] for c in test if c['kind']==kind);shuffle.update(dict(zip(cpids,rng.permutation(cpids))))
 for i,c in enumerate(test):
  for b in a.budgets:
   for n in methods:rows.append(evaluate(c,n,b,lam))
   r=evaluate(c,chosen['trend'],b,lam,omega(representative[shuffle[c['checkpoint']]],chosen['trend']));r['method']='shuffled';rows.append(r)
  if (i+1)%20==0:print(f'Test questions {i+1}/{len(test)}',flush=True)
 dump(out/'selections.json',rows)
 flat=[{k:v for k,v in r.items() if k not in ['selected_ids','weights','trace']} for r in rows];writecsv(out/'metrics.csv',flat)
 pairs={}
 for r in rows:pairs.setdefault((r['case'],r['budget']),{})[r['method']]=r
 diag=[]
 for (cid,b),pr in pairs.items():
  diag.append(dict(case=cid,budget=b,large_minus_small=np.mean([pr[f'fixed_{i}']['recall'] for i in [4,5]])-np.mean([pr[f'fixed_{i}']['recall'] for i in [0,1]]),oracle_headroom=max(pr[n]['recall'] for n in fixed)-pr[best_fixed]['recall']))
 writecsv(out/'scale_diagnostics.csv',diag)
 comparisons={n:bootstrap([r for r in rows if r['kind']=='primary'],chosen['trend'],n) for n in [best_fixed,chosen['level'],chosen['unweighted'],'relevance','mmr','shuffled']};dump(out/'comparisons.json',comparisons)
 dump(out/'cases.json',[{k:v for k,v in c.items() if k not in ['x','k','rel']}|{'rel':c['rel'].tolist()} for c in cases])
 versions={n:importlib.metadata.version(n) for n in ['numpy','scipy','torch','sentence-transformers','transformers']}
 dump(out/'manifest.json',dict(args=vars(a),versions=versions,data_sha256=hashlib.sha256(Path(a.data).read_bytes()).hexdigest(),source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),core_sha256=hashlib.sha256(Path(__file__).with_name('core.py').read_bytes()).hexdigest(),episodes_sha256=hashlib.sha256(Path(__file__).with_name('episodes.py').read_bytes()).hexdigest(),taus=TAUS.tolist(),ridge=RIDGE,split_counts={k:len([c for c in cases if c['split']==k]) for k in ['calibration','validation','test']}))
 lines=['# Aligned LoCoMo question-scope diagnostic','',f'Episodes: {len(cal)} calibration, {len(val)} validation, {len(test)} test.',
 f'Independent test conversations: {len(set(c["group"] for c in test))}. Common relevance weight: {lam}.',
 f'Validation-chosen fixed policy: {best_fixed}; adaptive: {chosen["trend"]}.',
 'Controlled revisions of real LoCoMo question bundles, not naturally recorded conversations or official QA scores.',
 'Each final scope is exactly the three active questions being answered. Gold = their evidence-ID union.',
 'All states have three questions, so point count cannot encode breadth. Each adjacent transition is measured and annotated separately.',
 'Constructed labels come from a keyword topic taxonomy, not human judgments. They are an independent construction control, not a semantic ground truth.',
 '', '| Method | Budget | Primary evidence recall | Candidate ceiling | Tokens |','|---|---:|---:|---:|---:|']
 for n in methods+['shuffled']:
  for b in a.budgets:
   rr=[r for r in rows if r['method']==n and r['budget']==b and r['kind']=='primary']
   lines.append(f'| {n} | {b} | {np.mean([r["recall"] for r in rr]):.3f} | {np.mean([r["candidate_recall"] for r in rr]):.3f} | {np.mean([r["tokens"] for r in rr]):.1f} |')
 lines+=['','## Paired adaptive minus control differences']
 for n,v in comparisons.items():lines.append(f'- {n}: {v["mean"]:+.4f}, exploratory 95% interval {v["ci95"]}.')
 lines+=['','## Interpretation',
 'Primary table includes changing-scope episodes; metrics.csv includes matched stable-ending controls.',
 'The same ending questions, candidate pool and gold occur after a changing path and a stable path. Inspect whether path-dependent selection actually helps.',
 'The human annotation sheet and detector both compare A to B and B to C separately. Recency beta combines those same transitions.',
 'The current request is explicitly revised at each state. Previous question bundles are not obligations that remain active.',
 'Four independent test conversations are insufficient for a definitive go/stop claim. No answer-quality metric is measured.',
 'A result using keyword-based constructed labels is a control, not validation of semantic scope. Human coherence and transition labels remain necessary.',
 'If lambda=1, relevance-only wins and changing scales cannot affect selection. Do not force a geometric effect after inspecting results.',
 'Use the same validation/test split and budgets for any replication; do not select a controller based on test scores.']
 scope_rows=[]
 for f in checkpoints:
  if f['split']=='test':
   for j,pred in enumerate(f['predicted_transitions']):scope_rows.append(dict(case=f['id'],transition=j,prediction=pred,constructed_label=f['constructed_labels'][j],human_label=human.get(f['id'],[None,None])[j]))
 writecsv(out/'scope_agreement.csv',scope_rows)
 lines += ['',f'Agreement with constructed transition labels: {np.mean([r["prediction"]==r["constructed_label"] for r in scope_rows]):.3f}. Not human-validated.']
 (out/'REPORT.md').write_text('\n'.join(lines));print(f'Done: {out}/REPORT.md')

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--data',default='data/locomo10.json');p.add_argument('--out',default='runs/semantic')
 p.add_argument('--model',default='sentence-transformers/all-MiniLM-L6-v2');p.add_argument('--revision',default='1110a243fdf4706b3f48f1d95db1a4f5529b4d41');p.add_argument('--device',default='cpu');p.add_argument('--cache',default='cache')
 p.add_argument('--bundles',type=int,default=4);p.add_argument('--labels');p.add_argument('--candidates',type=int,default=32);p.add_argument('--budgets',type=int,nargs='+',default=[256,512]);p.add_argument('--decay',type=float,default=.8);p.add_argument('--seed',type=int,default=2026)
 a=p.parse_args()
 if min(a.budgets)<=0 or a.bundles<1 or a.candidates<2 or not 0<a.decay<=1:p.error('Invalid positive counts/budgets or decay')
 main(a)
