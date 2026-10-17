"""Reproduce Tables E.7.6a–d from retained counts; never load models or pixels."""
from pathlib import Path
import csv,json,statistics as st,math
P=Path(__file__).resolve().parent
def read(p):
    with p.open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def matrix(p):
    with p.open(encoding='utf-8-sig',newline='') as f:return list(csv.reader(f))
def mean(rows,k):return st.mean(float(q[k]) for q in rows)
overall=read(P/'source_data/EMSR551_spatial/overall_per_seed.csv')
regional=read(P/'evidence/EMSR551_spatial/event_region_counts.csv')
systems={'main_only':'Native','decoder':'Decoder','fpn':'FPN'}
regions={'boundary':'Boundary','off_boundary':'Off-boundary','excluded_neighbourhood':'Excluded neighbourhood'}
tables={k:matrix(P/('source_data/Table_E_7_6'+k+'_EMSR551_'+{'a':'overall','b':'regions','c':'paired','d':'exposure'}[k]+'.csv')) for k in 'abcd'}
made={k:[v[0]] for k,v in tables.items()}
for condition in ['clean','perturbed']:
    for system,label in systems.items():
        q=[r for r in overall if r['condition']==condition and r['system']==system];assert len(q)==5
        for r in q:
            tp,fp,fn,ref=[int(r[k]) for k in ['TP','FP','FN','reference_foreground']]
            assert tp+fn==ref
            assert math.isclose(100*abs(fp-fn)/ref,float(r['absolute_extent_error_percent']),abs_tol=1e-10)
        made['a'].append([condition+' / '+label]+[f'{mean(q,k):.1f}' for k in ['FP','FN','signed_imbalance']]+[f'{mean(q,"absolute_extent_error_percent"):.2f}'])
        for region,regionlabel in regions.items():
            rr=[r for r in regional if r['system']==system and r['condition'].replace('degraded','perturbed')==condition and int(r['radius'])==2 and r['region']==region];assert len(rr)==5
            made['b'].append([condition+' / '+label,regionlabel]+[f'{mean(rr,k):.1f}' for k in ['FP','FN']]+[f'{100*mean(rr,k):.2f}' for k in ['FP_share','FN_share']])
    for system in ['decoder','fpn']:
        base={int(r['seed']):r for r in overall if r['condition']==condition and r['system']=='main_only'}
        qq=[r for r in overall if r['condition']==condition and r['system']==system]
        delta=lambda k:st.mean(float(r[k])-float(base[int(r['seed'])][k]) for r in qq)
        improved=sum(float(r['absolute_extent_error_percent'])<float(base[int(r['seed'])]['absolute_extent_error_percent']) for r in qq)
        made['c'].append([condition+' / '+systems[system]+'-Native']+[f'{delta(k):.1f}' for k in ['FP','FN','signed_imbalance']]+[f'{delta("absolute_extent_error_percent"):.2f}',f'{improved}/5'])
    refrows=[r for r in regional if r['system']=='main_only' and int(r['seed'])==2026 and int(r['radius'])==2 and r['condition'].replace('degraded','perturbed')==condition]
    assert len(refrows)==3
    valid=sum(int(r['valid_pixels']) for r in refrows);ref=sum(int(r['reference_foreground']) for r in refrows)
    for region,label in regions.items():
        r=next(r for r in refrows if r['region']==region)
        made['d'].append([condition,label,f'{100*int(r["valid_pixels"])/valid:.2f}',f'{100*int(r["reference_foreground"])/ref:.2f}'])
for k in tables:assert tables[k]==made[k],('Table E.7.6'+k,tables[k],made[k])
print(json.dumps({'status':'PASS','tables':['E.7.6'+k for k in tables],'seed_condition_records':len(overall),'model_calls':0,'inputs':'saved per-seed confusion and region counts'}))
