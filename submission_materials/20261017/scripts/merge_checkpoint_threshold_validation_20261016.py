"""Saved-count merge only; requires completed six-checkpoint server inference."""
from pathlib import Path
import argparse, csv, json, statistics

ROOT = Path(__file__).resolve().parents[1]
SYSTEMS = ('main_only', 'decoder', 'fpn')
SEEDS = range(2026, 2031)
VIEWS = ('clean', 'degraded')

def read(p):
    return json.loads(p.read_text(encoding='utf-8'))

def write_csv(p, rows):
    assert rows
    with p.open('w', newline='', encoding='utf-8-sig') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)

def metrics(c):
    tp, fp, fn = (c[k] for k in ('TP', 'FP', 'FN'))
    den = 2 * tp + fp + fn
    return dict(F1=2 * tp / den if den else 0,
                extent_error_percent=100 * abs(fp-fn)/(tp+fn) if tp+fn else None)

def add_event_sensors(counts):
    out = {}
    for key, c in counts.items():
        event = key.split('|', 1)[1]
        dst = out.setdefault(event, dict(TP=0, FP=0, FN=0))
        for k in dst: dst[k] += c[k]
    return out

def summarize(counts):
    event = [metrics(c) for c in counts.values()]
    pooled = {k: sum(c[k] for c in counts.values()) for k in ('TP','FP','FN')}
    defined = [q['extent_error_percent'] for q in event if q['extent_error_percent'] is not None]
    return dict(pooled_F1=metrics(pooled)['F1'], event_macro_F1=statistics.mean(q['F1'] for q in event),
                event_equal_extent_error_percent=statistics.mean(defined) if defined else None,
                events=len(event), relative_defined_events=len(defined))

def merge(new, output):
    # All gates precede any output creation. Missing results cannot become a partial formal table.
    for stage, tiles in [('val', 4853), ('test', 4863)]:
        c = read(new/stage/'reevaluation_complete.json')
        assert c['status']=='COMPLETE' and c['models']==6 and c['tiles']==tiles
    assert read(new/'test/reevaluation_complete.json')['baseline_fixed_test_counts_equal']
    assert read(new/'val/reevaluation_complete.json')['baseline_val_event_counts_equal']
    oldval = ROOT/'results/extent_calibration_val_20261011_v1'
    oldtest = ROOT/'results/extent_calibration_test_20261011_v1'
    val_counts, thresholds, test_counts = {}, [], {}
    for valdir, testdir in [(oldval, oldtest), (new/'val', new/'test')]:
        vc = read(valdir/'event_threshold_counts.json')['counts']
        tc = read(testdir/'calibrated_counts.json')['calibrated']
        ts = read(valdir/'selected_thresholds_frozen.json')['thresholds']
        assert not (set(val_counts)&set(vc)) and not (set(test_counts)&set(tc))
        val_counts.update(vc); test_counts.update(tc); thresholds.extend(ts)
    expected = {f'{s}_{n}' for s in SYSTEMS for n in SEEDS}
    assert set(val_counts)==expected and set(test_counts)==expected and len(thresholds)==60
    assert len({(q['model'],q['view'],q['sensor']) for q in thresholds})==60
    selected = {(q['model'],q['view'],q['sensor']):q for q in thresholds}
    eventrows, val_summaries, transferrows = [], [], []
    cache = {}
    for key in sorted(expected):
        system, seed = key.rsplit('_',1)
        for view in VIEWS:
            baseline = {}
            for event_sensor, grid in val_counts[key][view].items():
                assert len(grid)==19 and abs(grid[9]['tau']-.5)<1e-12
                baseline[event_sensor] = grid[9]
            base_events = add_event_sensors(baseline)
            assert len(base_events)==12
            cache[(system,int(seed),view)] = {e:metrics(c) for e,c in base_events.items()}
            val_summaries.append(dict(system=system,seed=int(seed),condition=view,**summarize(base_events)))
            for e,c in base_events.items():
                eventrows.append(dict(system=system,seed=int(seed),condition=view,event_id=e,**c,**metrics(c)))
            # Test baseline is read from historical fixed12026 counts, never inferred again here.
            bdir = oldtest if int(seed)<=2028 else new/'test'
            raw = read(bdir/f'{key}_reevaluation.json')['metrics'][view]['per_event']
            baseline_test = {e:dict(TP=q['confusion_matrix'][1][1],FP=q['confusion_matrix'][0][1],FN=q['confusion_matrix'][1][0]) for e,q in raw.items()}
            for strategy in ('baseline','extent_selected','F1_selected'):
                chosen_val = {}
                for event_sensor, grid in val_counts[key][view].items():
                    sensor = event_sensor.split('|',1)[0]
                    tau = selected[(key,view,sensor)][strategy]['tau']
                    chosen_val[event_sensor] = next(q for q in grid if abs(q['tau']-tau)<1e-12)
                cv = add_event_sensors(chosen_val)
                ct = baseline_test if strategy=='baseline' else add_event_sensors(test_counts[key][view][strategy])
                assert set(cv)==set(base_events) and set(ct)==set(baseline_test) and len(ct)==12
                for split, counts in [('val',cv),('test',ct)]:
                    transferrows.append(dict(system=system,seed=int(seed),condition=view,strategy=strategy,split=split,**summarize(counts)))
    paired, paired_events = [], []
    for view in VIEWS:
        for a,b in [('decoder','main_only'),('fpn','main_only'),('fpn','decoder')]:
            events = set(cache[(a,2026,view)])
            assert all(set(cache[(s,n,view)])==events for s in SYSTEMS for n in SEEDS)
            diffs = []
            for e in sorted(events):
                d = 100*statistics.mean(cache[(a,n,view)][e]['F1']-cache[(b,n,view)][e]['F1'] for n in SEEDS)
                diffs.append(d)
                paired_events.append(dict(condition=view,comparison=f'{a}-{b}',event_id=e,seed_mean_delta_F1_pp=d))
            paired.append(dict(condition=view,comparison=f'{a}-{b}',seeds=5,events=12,
                event_equal_delta_F1_pp=statistics.mean(diffs),event_median_delta_F1_pp=statistics.median(diffs),
                positive_events=sum(d>0 for d in diffs),negative_events=sum(d<0 for d in diffs),
                zero_events=sum(d==0 for d in diffs),at_least_0_5pp_events=sum(d>=.5 for d in diffs)))
    agg = []
    for system in SYSTEMS:
        for view in VIEWS:
            for strategy in ('baseline','extent_selected','F1_selected'):
                for split in ('val','test'):
                    rows=[q for q in transferrows if (q['system'],q['condition'],q['strategy'],q['split'])==(system,view,strategy,split)]
                    assert len(rows)==5
                    for metric in ('pooled_F1','event_macro_F1','event_equal_extent_error_percent'):
                        values=[q[metric] for q in rows if q[metric] is not None]
                        agg.append(dict(system=system,condition=view,strategy=strategy,split=split,metric=metric,
                            seeds=5,defined_seeds=len(values),mean=statistics.mean(values) if values else None,
                            sample_SD=statistics.stdev(values) if len(values)>1 else None))
    output.mkdir(parents=True,exist_ok=False)
    for name, rows in [('validation_event_counts',eventrows),('validation_per_seed',val_summaries),
                       ('validation_paired_event_summary',paired),('validation_paired_event_differences',paired_events),
                       ('threshold_transfer_per_seed',transferrows),('threshold_transfer_five_seed_summary',agg)]:
        write_csv(output/f'{name}.csv',rows)
    (output/'completion.json').write_text(json.dumps(dict(status='SAVED_COUNT_MERGE_COMPLETE',models=15,
        training=False,inference=False,new_test_access=False,evaluation_seed=12026,
        aggregation='sum sensor counts within event; paired seed mean within event; event equal',
        new_source=str(new)),indent=2),encoding='utf-8')

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--new-results',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();merge(a.new_results,a.output)
