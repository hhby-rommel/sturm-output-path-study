"""Four requested extent analyses from saved uniform-12026 confusion matrices only."""
from pathlib import Path
import csv
import json
import statistics
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
R=ROOT/'results/eval12026_extension_20261015_v1'
O=ROOT/'results/five_seed_extent_audit_20261015_v1'
O.mkdir(exist_ok=False)
SYSTEMS=['main_only','decoder','fpn'];SEEDS=[2026,2027,2028,2029,2030]
EXTREMES={'EMSR518','EMSR492','EMSR422'}
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def export(name,records):
    with (O/name).open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(records[0]));w.writeheader();w.writerows(records)

done=read(R/'supplementary12026/completion.json');assert done['historical_input_stream_match']
stream=done['input_stream_sha256']
raw={}
for system in SYSTEMS:
    for seed in SEEDS:
        p=R/('historical12026' if seed<2029 else 'supplementary12026')/f'{system}_{seed}_reevaluation.json'
        x=read(p);assert x['evaluation_seed']==12026 and x['shared_input_stream_sha256']==stream
        raw[system,seed]=x['metrics']
events=sorted(raw['main_only',2026]['degraded']['per_event'])
assert len(events)==12 and EXTREMES.issubset(events)
paired_seeds=[];paired_summary=[];event_summary=[];e8a=[];e8b=[];agreements=[];decomp=[];ranks=[];contributions=[];exclusion=[]
for raw_view,view in [('degraded','perturbed'),('clean','clean')]:
    bias=np.zeros((3,5,12));f1=np.zeros_like(bias)
    for si,system in enumerate(SYSTEMS):
        for ki,seed in enumerate(SEEDS):
            mm=raw[system,seed][raw_view]['per_event'];assert sorted(mm)==events
            for ei,event in enumerate(events):
                m=mm[event];(tn,fp),(fn,tp)=m['confusion_matrix'];assert tp+fn>0
                bias[si,ki,ei]=100*(fp-fn)/(tp+fn)
                f1[si,ki,ei]=2*tp/(2*tp+fp+fn)
                assert abs(f1[si,ki,ei]-m['flood_f1'])<1e-12
    error=np.abs(bias)
    mean_bias=bias.mean(axis=1);mean_error=error.mean(axis=1);mean_f1=f1.mean(axis=1)
    for a,b in [('decoder','main_only'),('fpn','main_only'),('fpn','decoder')]:
        ai,bi=SYSTEMS.index(a),SYSTEMS.index(b)
        # Equal events inside each seed; paired seed differences before sample SD.
        delta=(error[ai]-error[bi]).mean(axis=1)
        for ki,seed in enumerate(SEEDS):paired_seeds.append(dict(condition=view,comparison=f'{a}-{b}',seed=seed,
             a_event_equal_error_pct=float(error[ai,ki].mean()),b_event_equal_error_pct=float(error[bi,ki].mean()),
             paired_extent_difference_pp=float(delta[ki]),improved=bool(delta[ki]<-1e-12)))
        paired_summary.append(dict(condition=view,comparison=f'{a}-{b}',n_seeds=5,mean_difference_pp=float(delta.mean()),
             paired_sample_sd_pp=float(delta.std(ddof=1)),improved_seeds=int((delta < -1e-12).sum()),
             worsened_seeds=int((delta > 1e-12).sum()),unchanged_seeds=int((np.abs(delta)<=1e-12).sum())))
    for ei,event in enumerate(events):
        e8a.append(dict(condition=view,event=event,decoder_minus_native_f1_pp=float(100*(mean_f1[1,ei]-mean_f1[0,ei])),
                       fpn_minus_native_f1_pp=float(100*(mean_f1[2,ei]-mean_f1[0,ei])),
                       decoder_minus_native_absolute_extent_error_pp=float(mean_error[1,ei]-mean_error[0,ei]),
                       fpn_minus_native_absolute_extent_error_pp=float(mean_error[2,ei]-mean_error[0,ei])))
        e8b.append(dict(condition=view,event=event,native_signed_bias_pct=float(mean_bias[0,ei]),
                       decoder_signed_bias_pct=float(mean_bias[1,ei]),fpn_signed_bias_pct=float(mean_bias[2,ei])))
        for si,system in enumerate(SYSTEMS):
            event_summary.append(dict(condition=view,event=event,system=system,n_seeds=5,
                mean_f1=float(mean_f1[si,ei]),sample_sd_f1=float(f1[si,:,ei].std(ddof=1)),
                mean_signed_bias_pct=float(mean_bias[si,ei]),sample_sd_signed_bias_pp=float(bias[si,:,ei].std(ddof=1)),
                mean_absolute_extent_error_pct=float(mean_error[si,ei]),sample_sd_absolute_extent_error_pp=float(error[si,:,ei].std(ddof=1)),
                absolute_mean_signed_bias_pct=float(abs(mean_bias[si,ei]))))
    for si,system in [(1,'decoder'),(2,'fpn')]:
        gain=mean_f1[si]-mean_f1[0];ed=mean_error[si]-mean_error[0]
        pos=gain>1e-12;lower=ed < -1e-12;higher=ed > 1e-12
        agreements.append(dict(condition=view,system=system,f1_improved_events=int(pos.sum()),
            among_f1_improved_extent_lower=int((pos&lower).sum()),among_f1_improved_extent_higher=int((pos&higher).sum()),
            among_f1_improved_extent_unchanged=int((pos&~lower&~higher).sum()),
            lower_event_ids=','.join(e for e,p,l in zip(events,pos,lower) if p and l),
            higher_event_ids=','.join(e for e,p,h in zip(events,pos,higher) if p and h)))
    # Same balanced descriptive SS decomposition used by the three-seed analysis.
    matrix=mean_bias.T;grand=matrix.mean()
    ss_event=3*np.square(matrix.mean(axis=1)-grand).sum()
    ss_path=12*np.square(matrix.mean(axis=0)-grand).sum()
    ss_inter=np.square(matrix-matrix.mean(axis=1,keepdims=True)-matrix.mean(axis=0,keepdims=True)+grand).sum()
    total=np.square(matrix-grand).sum();assert np.isclose(ss_event+ss_path+ss_inter,total)
    for term,ss,df in [('Event',ss_event,11),('Path',ss_path,2),('Event_path_interaction',ss_inter,22)]:
        decomp.append(dict(condition=view,analysis='five_seed_mean_36_cells',term=term,SS_squared_pp=float(ss),
                           total_SS_squared_pp=float(total),share_percent=float(100*ss/total),df=df))
    within=np.square(bias-mean_bias[:,None,:]).sum()
    all_total=np.square(bias-grand).sum();assert np.isclose(5*(ss_event+ss_path+ss_inter)+within,all_total)
    for term,ss,df in [('Event',5*ss_event,11),('Path',5*ss_path,2),('Event_path_interaction',5*ss_inter,22),('Within_event_path_seed_variation',within,144)]:
        decomp.append(dict(condition=view,analysis='all_180_seed_records_descriptive',term=term,SS_squared_pp=float(ss),
                           total_SS_squared_pp=float(all_total),share_percent=float(100*ss/all_total),df=df))
    excluded=np.array([e in EXTREMES for e in events]);keep=~excluded
    for si,system in enumerate(SYSTEMS):
        order=np.argsort(-mean_error[si]);total_error=mean_error[si].sum()
        for rank,ei in enumerate(order,start=1):
            ranks.append(dict(condition=view,system=system,rank=rank,event=events[ei],mean_absolute_extent_error_pct=float(mean_error[si,ei]),
                               contribution_to_sum_of_event_errors_percent=float(100*mean_error[si,ei]/total_error),predefined_extreme=bool(excluded[ei])))
        contributions.append(dict(condition=view,system=system,predefined_three=','.join(sorted(EXTREMES)),
             actual_top_three=','.join(events[i] for i in order[:3]),predefined_are_top_three=set(events[i] for i in order[:3])==EXTREMES,
             contribution_percent=float(100*mean_error[si,excluded].sum()/total_error),
             all_12_event_equal_error_pct=float(mean_error[si].mean()),three_event_mean_error_pct=float(mean_error[si,excluded].mean())))
        per_seed=error[si][:,keep].mean(axis=1)
        exclusion.append(dict(condition=view,system=system,retained_events=9,excluded=','.join(sorted(EXTREMES)),
              event_equal_error_mean_pct=float(per_seed.mean()),event_equal_error_sample_sd_pp=float(per_seed.std(ddof=1)),
              seed2026_pct=float(per_seed[0]),seed2027_pct=float(per_seed[1]),seed2028_pct=float(per_seed[2]),seed2029_pct=float(per_seed[3]),seed2030_pct=float(per_seed[4])))

for name,rr in [('paired_extent_by_seed.csv',paired_seeds),('paired_extent_summary.csv',paired_summary),
                ('Table_E_8a_five_seed.csv',e8a),('Table_E_8b_five_seed.csv',e8b),('event_extent_mean_sd.csv',event_summary),
                ('F1_extent_event_agreement.csv',agreements),('bias_descriptive_decomposition.csv',decomp),
                ('event_extent_error_ranking.csv',ranks),('three_extreme_event_contribution.csv',contributions),
                ('excluding_three_events_extent.csv',exclusion)]:export(name,rr)

def f(x):return f'{x:+.3f}'
table=['# 五seed事件表（独立审计稿，尚未写回Supplement）','','Table E.8a. Five-seed mean event-level F1 and absolute extent-error differences under controlled perturbation.','',
       '| Event | ΔF1 Decoder−Native (pp) | ΔF1 FPN−Native (pp) | Δ absolute error Decoder−Native (pp) | Δ absolute error FPN−Native (pp) |',
       '|---|---:|---:|---:|---:|']
for r in [r for r in e8a if r['condition']=='perturbed']:
    table.append('| '+r['event']+' | '+' | '.join(f(r[k]) for k in ['decoder_minus_native_f1_pp','fpn_minus_native_f1_pp','decoder_minus_native_absolute_extent_error_pp','fpn_minus_native_absolute_extent_error_pp'])+' |')
table+=['','Table E.8b. Five-seed mean event-level signed extent bias under controlled perturbation.','',
        '| Event | Native signed bias (%) | Decoder signed bias (%) | FPN signed bias (%) |','|---|---:|---:|---:|']
for r in [r for r in e8b if r['condition']=='perturbed']:
    table.append('| '+r['event']+' | '+' | '.join(f(r[k]) for k in ['native_signed_bias_pct','decoder_signed_bias_pct','fpn_signed_bias_pct'])+' |')
table+=['','E.8a先对每seed每事件计算绝对误差，再对五seed平均；E.8b平均signed bias。mean(|bias|)与|mean(bias)|不可互换。CSV同时提供clean结果。']
(O/'Tables_E8a_E8b_five_seed.md').write_text('\n'.join(table)+'\n',encoding='utf-8')
payload=dict(scope='Current Decoder2029; three paths x five seeds; uniform12026 evaluation; saved counts only',
             paired_extent=paired_summary,event_agreement=agreements,decomposition=decomp,extreme_contribution=contributions,excluding_three=exclusion,
             definitions=dict(signed_bias='100*(FP-FN)/(TP+FN)',absolute_error='abs(signed_bias)',paired_difference='Within each seed average12 event absolute errors, subtract paired paths, then mean and sample SD across5 differences',
                              extreme_share='Sum of three event mean absolute relative errors divided by sum over all12 event mean errors; not pixel error, area or a signed bias contribution',
                              zero_denominator='No zero reference foreground denominator in these saved event domains',tie_tolerance=1e-12),
             no_inference=True,no_new_test=True,paper_modified=False)
(O/'audit.json').write_text(json.dumps(payload,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps({k:[r for r in payload[k] if r['condition']=='perturbed'] for k in ['paired_extent','event_agreement','decomposition','extreme_contribution','excluding_three']},ensure_ascii=True))
