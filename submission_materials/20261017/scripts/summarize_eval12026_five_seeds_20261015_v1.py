"""Arithmetic summaries of 15 already-evaluated frozen checkpoints; no model execution."""
import csv
import json
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
R = ROOT / 'results/eval12026_extension_20261015_v1'
historical = json.loads((R/'historical12026/reevaluation_complete.json').read_text())
done = json.loads((R/'supplementary12026/completion.json').read_text())
assert done['historical_input_stream_match'] and done['input_stream_sha256'] == historical['input_stream_sha256']
rows, events, sensors = [], [], []

def counts(m):
    (tn, fp), (fn, tp) = m['confusion_matrix']
    assert tp + fn > 0
    return dict(tn=tn,fp=fp,fn=fn,tp=tp,reference_foreground=tp+fn,predicted_foreground=tp+fp,
                signed_extent_bias_pct=100*(fp-fn)/(tp+fn),absolute_extent_error_pct=100*abs(fp-fn)/(tp+fn))

def measures(m):
    return {**{k:m[k] for k in ['flood_f1','flood_iou','flood_precision','flood_recall','pixel_accuracy','valid_pixel_count']},**counts(m)}

for system in ['main_only','decoder','fpn']:
    for seed in [2026,2027,2028,2029,2030]:
        p=R/('historical12026' if seed<2029 else 'supplementary12026')/f'{system}_{seed}_reevaluation.json'
        result=json.loads(p.read_text())
        assert result['evaluation_seed']==12026 and result['shared_input_stream_sha256']==done['input_stream_sha256']
        for raw,view in [('clean','clean'),('degraded','perturbed')]:
            m=result['metrics'][raw];ee=m['per_event'];assert len(ee)==12
            rows.append(dict(system=system,seed=seed,condition=view,**measures(m['overall']),
                             event_macro_f1=statistics.mean(v['flood_f1'] for v in ee.values()),
                             event_equal_absolute_extent_error_pct=statistics.mean(counts(v)['absolute_extent_error_pct'] for v in ee.values()),
                             source=str(p)))
            for event,v in sorted(ee.items()):events.append(dict(system=system,seed=seed,condition=view,event=event,**measures(v)))
            for sensor,v in sorted(m['per_sensor'].items()):sensors.append(dict(system=system,seed=seed,condition=view,sensor=sensor,**measures(v)))

def export(name,records):
    with (R/name).open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.DictWriter(f,fieldnames=list(records[0]));w.writeheader();w.writerows(records)

export('five_seed_per_run_metrics.csv',rows)
export('five_seed_event_metrics_and_counts.csv',events)
export('five_seed_sensor_metrics.csv',sensors)
summary=[]
for system in ['main_only','decoder','fpn']:
    for view in ['clean','perturbed']:
        selected=[q for q in rows if q['system']==system and q['condition']==view]
        assert len(selected)==5
        row=dict(system=system,condition=view,n_seeds=5)
        for metric in ['flood_f1','event_macro_f1','event_equal_absolute_extent_error_pct']:
            vv=[q[metric] for q in selected];row[metric+'_mean']=statistics.mean(vv);row[metric+'_sample_sd']=statistics.stdev(vv)
        summary.append(row)
export('five_seed_mean_sample_sd.csv',summary)
paired=[]
for system in ['decoder','fpn']:
    for view in ['clean','perturbed']:
        for seed in [2026,2027,2028,2029,2030]:
            a=next(q for q in rows if q['system']==system and q['condition']==view and q['seed']==seed)
            n=next(q for q in rows if q['system']=='main_only' and q['condition']==view and q['seed']==seed)
            paired.append(dict(comparison=system+'-native',condition=view,seed=seed,
                               pooled_f1_delta_pp=100*(a['flood_f1']-n['flood_f1']),
                               event_macro_f1_delta_pp=100*(a['event_macro_f1']-n['event_macro_f1']),
                               extent_error_delta_pp=a['event_equal_absolute_extent_error_pct']-n['event_equal_absolute_extent_error_pct']))
export('five_seed_paired_differences.csv',paired)
print(json.dumps({'condition':'perturbed','summary':[q for q in summary if q['condition']=='perturbed'],
                  'rows':len(rows),'event_rows':len(events),'sensor_rows':len(sensors)},ensure_ascii=True))
