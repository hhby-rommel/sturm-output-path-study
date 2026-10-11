# Copyright (C) 2026 STURM output-path study contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
# Source-release notice added 2026-10-11; numerical code unchanged.
from pathlib import Path
import json,hashlib,csv,statistics,numpy as np
R=Path(__file__).resolve().parents[1];O=R/'results/manuscript_existing_counts_audit_20261010_v1';O.mkdir(exist_ok=True)
T=R/'results/output_path_followup_collected_20261010_v1';V=R/'results/spatial_val_analysis_20261010_v1/spatial_val_20261010_v1/summary.json'
v=json.loads(V.read_text());seeds=[2026,2027,2028];systems=['main_only','decoder','fpn']
counts={};rows=[];sources=[V]
def convert(q):
 cm=q['confusion_matrix'];return dict(TP=cm[1][1],FP=cm[0][1],FN=cm[1][0])
def f1(q):return 2*q['TP']/(2*q['TP']+q['FP']+q['FN']) if 2*q['TP']+q['FP']+q['FN'] else 0.
def add(a,b):return {k:a[k]+b[k] for k in ['TP','FP','FN']}
def stat(a):return dict(mean=statistics.mean(a),sample_SD=statistics.stdev(a),values=a)
for system in systems:
 for seed in seeds:
  file=T/f'{system}_{seed}_reevaluation.json';sources.append(file);q=json.loads(file.read_text())['metrics']
  for view in ['clean','degraded']:
   c=convert(q[view]['overall']);vc={k:sum(v[f'{system}_{seed}/{view}/{sensor}'][k] for sensor in ['S1','S2']) for k in ['TP','FP','FN']}
   counts[system,seed,view]=dict(test=c,val=vc,combined=add(c,vc))
   es=q[view]['per_event'];assert len(es)==12
   for k in ['TP','FP','FN']:assert sum(convert(e)[k] for e in es.values())==c[k]
   for event,e in es.items():
    ec=convert(e);gt=ec['TP']+ec['FN'];bias=100*(ec['FP']-ec['FN'])/gt if gt else None
    rows.append(dict(system=system,seed=seed,view=view,event=event,**ec,GT_foreground_pixels=gt,predicted_foreground_pixels=ec['TP']+ec['FP'],F1=f1(ec),signed_foreground_extent_bias_percent=bias,absolute_foreground_extent_error_percent=abs(bias) if bias is not None else None))
events=sorted(set(x['event'] for x in rows));summary={};paired={}
for view in ['clean','degraded']:
 summary[view]={}
 for system in systems:
  summary[view][system]={domain:stat([f1(counts[system,seed,view][domain]) for seed in seeds]) for domain in ['val','test','combined']}
  for metric in ['signed_foreground_extent_bias_percent','absolute_foreground_extent_error_percent']:
   vals=[statistics.mean(r[metric] for r in rows if r['system']==system and r['seed']==seed and r['view']==view and r[metric] is not None) for seed in seeds]
   summary[view][system][metric]=stat(vals)
 paired[view]={}
 for system in ['decoder','fpn']:
  rr=[]
  for event in events:
   delta=[next(r['F1'] for r in rows if (r['system'],r['seed'],r['view'],r['event'])==(system,seed,view,event))-next(r['F1'] for r in rows if (r['system'],r['seed'],r['view'],r['event'])==('main_only',seed,view,event)) for seed in seeds]
   rr.append(dict(event=event,mean_difference_pp=100*statistics.mean(delta),seed_differences_pp=[100*x for x in delta]))
  means=np.array([r['mean_difference_pp'] for r in rr]);rng=np.random.default_rng(20261010)
  bs=means[rng.integers(0,len(means),size=(20000,len(means)))].mean(1)
  paired[view][system]=dict(events=rr,positive=int((means>0).sum()),negative=int((means<0).sum()),zero=int((means==0).sum()),event_mean_difference_pp=float(means.mean()),descriptive_event_bootstrap95_pp=np.quantile(bs,[.025,.975]).tolist(),bootstrap_scope='12 test events; average seeds within each event; no independent seed-event replication')
with (O/'test_event_counts_and_extent.csv').open('w',encoding='utf-8-sig',newline='') as f:
 w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
report=dict(summary=summary,test_event_paired=paired,val_event_counts_available=False,combined_event_sign_analysis_completed=False,combined_pooled_role='Diagnostic only: val was used in checkpoint selection; no claim of independent 24-event test',extent_role='Relative water-foreground extent in evaluation tile domains; no deduplication of overlaps or physical km2; not flood-only extent',new_inference=False,test_access=False,sources=[dict(path=str(p),SHA256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in sources])
(O/'analysis.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps({'combined':{v:{s:summary[v][s]['combined']['mean'] for s in systems} for v in summary},'extent_MARE':{s:summary['degraded'][s]['absolute_foreground_extent_error_percent']['mean'] for s in systems},'signs':{s:[paired['degraded'][s]['positive'],paired['degraded'][s]['negative']] for s in ['decoder','fpn']}},ensure_ascii=False))
