"""Score independent human labels on precisely the two detector transitions."""
import argparse,csv,json
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--run',required=True);p.add_argument('--labels',required=True);p.add_argument('--out',required=True);a=p.parse_args()
features={f['id']:f for f in json.loads((Path(a.run)/'scope_features.json').read_text())};out=Path(a.out)
if out.exists():raise SystemExit('Use a new output file.')
valid={'narrowing','stable','broadening'};scored=[];seen=set()
for r in csv.DictReader(open(a.labels)):
 key=r['checkpoint']
 if key not in features:raise ValueError('Unknown checkpoint')
 if key in seen:raise ValueError('One adjudicated row per checkpoint')
 seen.add(key);f=features[key]
 if f['split']!='test':continue
 for j,column in enumerate(['label_A_to_B','label_B_to_C']):
  human=r[column].strip().lower()
  if not human:continue
  if human not in valid:raise ValueError('Allowed labels: narrowing, stable, broadening; leave uncertain blank')
  scored.append(dict(case=key,transition=column,human=human,predicted=f['predicted_transitions'][j],match=human==f['predicted_transitions'][j]))
result=dict(annotated_transitions=len(scored),total_test_transitions=2*sum(f['split']=='test' for f in features.values()),accuracy=sum(r['match'] for r in scored)/len(scored) if scored else None,rows=scored)
out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(result,indent=2));print(result['accuracy'])
