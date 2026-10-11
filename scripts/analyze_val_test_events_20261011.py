# Copyright (C) 2026 STURM output-path study contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
# Source-release notice added 2026-10-11; numerical code unchanged.
from pathlib import Path
import json, statistics, hashlib, sys
import numpy as np
R=Path(__file__).resolve().parents[1]
V=R/'results/val_event_counts_20261010_v1'
T=R/'results/output_path_followup_collected_20261010_v1'
O=Path(sys.argv[1]) if len(sys.argv)>1 else R/'results/val_test_event_analysis_20261011_v1';O.mkdir(exist_ok=False)
assert json.loads((V/'reevaluation_complete.json').read_text())['old_spatial_counts_equal']
sources=[];metrics={};report={}
for split,folder in [('val',V),('test',T)]:
 for system in ['main_only','decoder','fpn']:
  for seed in [2026,2027,2028]:
   f=folder/f'{system}_{seed}_reevaluation.json';sources.append({'path':str(f),'sha256':hashlib.sha256(f.read_bytes()).hexdigest()})
   metrics[split,system,seed]=json.loads(f.read_text())['metrics']
def f1(q):
 c=q['confusion_matrix'];return 2*c[1][1]/(2*c[1][1]+c[0][1]+c[1][0])
for view in ['clean','degraded']:
 report[view]={}
 for system in ['decoder','fpn']:
  rows=[]
  for split in ['val','test']:
   events=sorted(metrics[split,'main_only',2026][view]['per_event']);assert len(events)==12
   for event in events:
    delta=[100*(f1(metrics[split,system,s][view]['per_event'][event])-f1(metrics[split,'main_only',s][view]['per_event'][event])) for s in [2026,2027,2028]]
    rows.append({'split':split,'event':event,'mean_difference_pp':statistics.mean(delta),'seed_differences_pp':delta})
  assert len(set(x['event'] for x in rows))==24
  domains={}
  for domain in ['val','test','combined']:
   rs=[x for x in rows if domain=='combined' or x['split']==domain];a=np.array([x['mean_difference_pp'] for x in rs]);rng=np.random.default_rng(20261011)
   bs=a[rng.integers(0,len(a),(20000,len(a)))].mean(axis=1)
   domains[domain]={'events':len(rs),'positive':int((a>0).sum()),'negative':int((a<0).sum()),'zero':int((a==0).sum()),'event_mean_difference_pp':float(a.mean()),'descriptive_bootstrap95_pp':np.quantile(bs,[.025,.975]).tolist()}
  report[view][system]={'domains':domains,'event_differences':rows}
output={'results':report,'scope':'Descriptive event analysis; val participated in checkpoint selection; average seeds within event; combined is not independent test','test_access':False,'new_training':False,'sources':sources}
(O/'analysis.json').write_text(json.dumps(output,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps({v:{s:r['domains'] for s,r in q.items()} for v,q in report.items()},ensure_ascii=False,indent=2))
