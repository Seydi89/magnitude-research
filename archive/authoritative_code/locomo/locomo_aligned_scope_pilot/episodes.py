"""Controlled revisions using actual LoCoMo questions. Not natural dialogue."""
import hashlib,re
import numpy as np
THEMES={
 'education':r'\b(school|college|university|study|studies|degree|education|learn|learning|student|class|classes)\b',
 'work':r'\b(job|jobs|work|working|career|business|company|app|profession)\b',
 'travel':r'\b(travel|trip|visit|visited|vacation|holiday|country|countries|japan|sweden|canada|switzerland)\b',
 'music':r'\b(music|musician|band|concert|guitar|piano|song|singing|singer)\b',
 'art':r'\b(art|artist|paint|painting|paintings|painted|drawing|draw|pottery|museum|gallery)\b',
 'sport':r'\b(sport|sports|run|running|race|marathon|swim|swimming|hike|hiking|basketball|cycling|fitness|gym)\b',
 'games':r'\b(game|games|gaming|video|tournament|esports|streaming)\b',
 'animals':r'\b(dog|dogs|cat|cats|pet|pets|animal|animals|shelter)\b',
 'family':r'\b(family|mother|father|parent|parents|son|daughter|children|kids|child|sister|brother|married|wedding)\b',
 'food':r'\b(food|cook|cooking|meal|baking|restaurant|recipe|recipes|dinner)\b'}
VALUES={'narrowing':-1.,'stable':0.,'broadening':1.}
def spread(x):
 x=np.asarray(x,dtype=np.float64);return float(np.mean(np.sum((x-x.mean(0))**2,axis=1)))
def features(states,decay):
 widths=np.array([spread(s) for s in states]);changes=np.diff(widths);beta=np.array([decay,1.]);beta/=beta.sum()
 return dict(trend=float(beta@changes),unweighted=float(changes.mean()),level=float(widths[-1]),widths=widths.tolist(),deltas=changes.tolist(),beta=beta.tolist())
def label_signal(labels,beta):return float(np.dot(beta,[VALUES[x] for x in labels]))
def build(convs,emb,cost,a):
 cases=[];checkpoints=[];blind=[];diagnostics=[]
 for c in convs:
  bytheme={};seen=set()
  for q in c['qas']:
   if q['question'] in seen:continue
   seen.add(q['question']);themes=[t for t,p in THEMES.items() if re.search(p,q['question'],re.I)]
   if len(themes)==1:bytheme.setdefault(themes[0],[]).append(q)
  eligible=sorted(k for k,v in bytheme.items() if len(v)>=3)
  memories=[]
  for t in c['turns']:
   for j,p in enumerate(t['parts']):memories.append(dict(id=f"{t['id']}#{j}",source=t['id'],text=p,cost=cost[p],date=t['date']))
  mx=np.array([emb[m['text']] for m in memories])
  for theme in eligible[:a.bundles]:
   others=sorted(k for k in bytheme if k!=theme)
   if len(others)<2:continue
   ordered=lambda qs:sorted(qs,key=lambda q:hashlib.sha256(f"{a.seed}:{c['group']}:{q['index']}".encode()).hexdigest())
   focused=ordered(bytheme[theme])[:3];other=[ordered(bytheme[t])[0] for t in others[:2]]
   broad=[focused[0],*other];middle=[focused[0],focused[1],other[0]]
   scenarios=[('narrowing',[broad,middle,focused],['narrowing','narrowing']),('broadening',[focused,middle,broad],['broadening','broadening']),
              ('stable_focused',[focused]*3,['stable']*2),('stable_broad',[broad]*3,['stable']*2)]
   for direction,states,labels in scenarios:
    cid=c['group']+'/'+hashlib.sha256((theme+':'+direction).encode()).hexdigest()[:12];current=states[-1];texts=[[q['question'] for q in s] for s in states]
    # C is aligned exactly to the current active questions, not unrelated archive chatter.
    queries=np.array([emb[t] for t in texts[-1]]);sim=mx@queries.T;rel=sim.max(axis=1)
    ranks=[sorted(range(len(memories)),key=lambda j:(-sim[j,k],memories[j]['id'])) for k in range(3)]
    ix=[];seen_ids=set()
    for depth in range(len(memories)):
     for ranking in ranks:
      j=ranking[depth]
      if j not in seen_ids:ix.append(j);seen_ids.add(j)
      if len(ix)>=a.candidates:break
     if len(ix)>=a.candidates:break
    pool=[memories[j] for j in ix]
    gold=sorted({g for q in current for g in q['gold']});required={g:[m['id'] for m in memories if m['source']==g] for g in gold};ids={m['id'] for m in pool}
    feat=features([np.array([emb[t] for t in s]) for s in texts],a.decay)
    question='Please answer these current questions using my stored conversation:\n'+'\n'.join(f'{i+1}. {t}' for i,t in enumerate(texts[-1]))
    cases.append(dict(id=cid,checkpoint=cid,group=c['group'],split=c['split'],kind='primary' if direction in ['narrowing','broadening'] else 'control',
     bundle=theme,constructed_direction=direction,constructed_labels=labels,states=texts,source_question_indices=[[q['index'] for q in s] for s in states],
     question=question,category='constructed_bundle',gold=gold,required=required,reachable=[g for g,v in required.items() if set(v)<=ids],candidates=pool,rel=rel[ix],x=mx[ix],feat=feat))
    checkpoints.append(dict(id=cid,group=c['group'],split=c['split'],constructed_labels=labels,**feat))
    blind.append(dict(checkpoint=cid,group=c['group'],split=c['split'],state_A='\n'.join(texts[0]),state_B='\n'.join(texts[1]),state_C='\n'.join(texts[2]),
     label_A_to_B='',label_B_to_C='',coherent_current_request='',notes=''))
    diagnostics.append(dict(case=cid,split=c['split'],constructed_direction=direction,source_questions=str([[q['index'] for q in s] for s in states]),topic=theme))
 return cases,checkpoints,blind,diagnostics
