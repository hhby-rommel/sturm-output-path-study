"""Read-only evidence audit and existing-count sensitivity analysis. No models or hashes."""
from pathlib import Path
import csv
import io
import json
import math
import statistics
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
R = ROOT / 'results/eval12026_extension_20261015_v1'
O = ROOT / 'results/five_seed_audit_20261015_v1'
if O.exists():
    assert not any(O.iterdir()), 'Existing completed audit; use a new version'
else:O.mkdir()
SEEDS = [2026,2027,2028,2029,2030]
SYSTEMS = ['main_only','decoder','fpn']
checks, protocols, source_matrix = [], [], []

def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def check(name, value, detail):
    assert value, name
    checks.append(dict(check=name,status='PASS',detail=detail))
def close(a,b):return math.isclose(float(a),float(b),rel_tol=1e-11,abs_tol=1e-11)
def export(name,rows):
    fields=list(dict.fromkeys(k for row in rows for k in row))
    with (O/name).open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)

old_assets=read(R/'historical12026/frozen_assets.json')
new_assets=read(R/'control/frozen_assets.json')
assets={(q['system'],q['seed']):q for q in old_assets+new_assets}
check('15_unique_assets',len(assets)==15,'Three paths x five seeds; current Decoder2029 chosen by user')
stream=read(R/'historical12026/reevaluation_complete.json')['input_stream_sha256']
completion=read(R/'supplementary12026/completion.json')
check('historical_new_input_stream',completion['input_stream_sha256']==stream and completion['historical_input_stream_match'],'Reuse existing stream receipts; no new hashes')
fresh=read(R/'control/collected_run_records.json')
archive=ROOT/'results/core_server_assets_20261003_v1/files'
decoder_reaudit=read(ROOT/'results/decoder_protocol_reaudit_20261002_v1/server_metadata.json')
engine_equiv=read(ROOT/'results/decoder_protocol_reaudit_20261002_v1/engine_equivalence.json')
check('prior_decoder_engine_compatibility',all(engine_equiv[k] for k in ['model_state_exact','optimizer_state_exact','cpu_rng_exact','reported_loss_exact']),
      'Reuse prior recorded compatibility check; not rerun here')
shared_fpn_protocol=set();shared_fpn_lock=set()
native_provenance=[]
for system in SYSTEMS:
    for seed in SEEDS:
        q=assets[system,seed]
        if seed>=2029:
            files=fresh[f'{system}_{seed}']
            base=str(R/'original_run_records'/f'{system}_{seed}')
        elif system=='fpn':
            folder=ROOT/'results/fpn_formal_collection_20261010_v1/records'/f'seed{seed}'
            files={n:(folder/n).read_text() for n in ['history.json','completion.json']}
            base=str(folder)
        else:
            rel=Path(q['checkpoint'].replace('/root/autodl-tmp/sturm_codex_20260919/','')).parent
            folder=archive/rel
            if not (folder/'result.json').exists() and system=='main_only':
                folder=ROOT/'results/decoder_main_only_control_20261002_v1/server_completed'/str(seed)
            files={n:(folder/n).read_text() for n in ['result.json','provenance.json','history.csv']}
            base=str(folder)
        if system=='fpn':
            result=json.loads(files['completion.json']);history=json.loads(files['history.json'])
            selection=result['best'];epochs=result['epochs'];ident=result['identity']
            shared_fpn_protocol.add(ident['protocol_sha256']);shared_fpn_lock.add(ident['lock_sha256'])
            check(f'{system}_{seed}_checkpoint_identity',result['best_sha256']==q['checkpoint_sha256'],'Existing frozen best digest equals training completion; not recomputed')
            candidates=[dict(epoch=h['epoch'],clean_flood_f1=h['validation']['clean']['overall']['flood_f1'],
                             degraded_flood_f1=h['validation']['degraded']['overall']['flood_f1']) for h in history]
            initialization=ident['pretrained_load']
            optimizer='AdamW3e-4/1e-4; no scheduler; source recipe and execution lock'
            bn='frozen running statistics; source recipe'
            if seed==2029:
                receipt=json.loads(files['resume_receipt.json'])
                check('fpn2029_restored_state',receipt['next_epoch']==10 and len(receipt['restored'])==10 and not receipt['numerical_protocol_changed'],
                      'Resume receipt: model/optimizer/RNG/sampler/history/selection/patience; no future bitwise trajectory claim')
        else:
            result=json.loads(files['result.json']);pr=json.loads(files['provenance.json'])
            history=list(csv.DictReader(io.StringIO(files['history.csv'])))
            selection=result['best_selection'];epochs=result['epochs_completed'];initialization=result['initialization']
            candidates=[dict(epoch=int(h['epoch']),clean_flood_f1=float(h['clean_f1']),degraded_flood_f1=float(h['degraded_f1'])) for h in history]
            check(f'{system}_{seed}_budget',result['epochs_requested']==50,'Requested max50')
            if result.get('optimization_spec'):
                spec=result['optimization_spec']
                check(f'{system}_{seed}_optimizer',spec['optimizer']=='adamw' and spec['learning_rate']==.0003 and spec['weight_decay']==.0001 and spec['scheduler']=='none' and result['freeze_batchnorm_running_stats'],
                      'Result metadata explicitly records optimizer and frozen BN')
                optimizer='AdamW3e-4/1e-4; no scheduler';bn='True in result metadata'
            else:
                prior=next(x for x in decoder_reaudit['runs'] if x['arm']=='decoder' and x['seed']==seed)
                ck=prior['checkpoints']['best.pt']
                check(f'decoder_{seed}_historical_optimizer',ck['max_epochs']==50 and ck['patience']==10 and ck['scheduler_state_is_none'] and all(g['lr']==.0003 and g['weight_decay']==.0001 for g in ck['parameter_groups']),
                      'Missing newer JSON fields resolved by existing checkpoint audit and archived historical source')
                optimizer='AdamW3e-4/1e-4; no scheduler; historical checkpoint audit';bn='freeze_batchnorm=True in archived historical runner'
            if system=='main_only':native_provenance.append(pr['hashes'])
            source_matrix.append(dict(system=system,seed=seed,manifest_recorded_digest=pr['hashes']['data/manifests/manifest.csv'],
                                      split_recorded_digest=pr['hashes']['data/splits/event_split_v1.csv'],
                                      runner_recorded_digest=pr['hashes'].get('scripts/server_main.py','historical provenance omits source digests'),
                                      engine_recorded_digest=pr['hashes'].get('src/training/engine.py','resolved using previous source audit'),
                                      config_recorded_digest=pr['hashes']['configs/project.yaml']))
        best=max(candidates,key=lambda c:(c['degraded_flood_f1'],c['clean_flood_f1'],-c['epoch']))
        check(f'{system}_{seed}_selection',best==selection,'Reconstructed from every saved validation epoch with original tie rule')
        check(f'{system}_{seed}_history_and_stop',len(history)==epochs and epochs<=50 and epochs-int(selection['epoch'])==10,'History complete; patience10 stopping, budget includes resumed epochs')
        check(f'{system}_{seed}_initialization',initialization['matched_keys']==359 and initialization['skipped_shape_keys']==5 and initialization['skipped_name_keys']==0,'Registered compatible initialization record')
        if seed<2029 and system!='fpn':check(f'{system}_{seed}_frozen_epoch',int(q['registration']['identity']['selected_epoch'].split('.')[0])==selection['epoch'],'Historical frozen checkpoint registration matches saved selection')
        if seed>=2029:check(f'{system}_{seed}_frozen_epoch',q['selection']==selection and q['selected_epoch']==selection['epoch'],'Supplementary frozen asset matches completed training selection')
        protocols.append(dict(system=system,seed=seed,epochs=epochs,best_epoch=int(selection['epoch']),stale=10,
                              optimizer=optimizer,bn=bn,pretrained_matched=359,supervision='main clean + perturbed CE+Dice',
                              source=base,checkpoint=q['checkpoint'],checkpoint_recorded_digest=q['checkpoint_sha256']))
check('fpn_five_seed_protocol_lock',len(shared_fpn_protocol)==len(shared_fpn_lock)==1,'All five run identities share protocol and execution lock; only seed/resume entry differs')
check('native_five_seed_config_sources',all(x==native_provenance[0] for x in native_provenance),'Existing provenance maps identical across all five Native runs')

root_split=list(csv.DictReader((ROOT/'data/splits/event_split_v1.csv').open(encoding='utf-8-sig')))
w_split=list(csv.DictReader((R/'server_sources/experiments/followup_server_20261008_v1/workspace/data/splits/event_split_v1.csv').open(encoding='utf-8-sig')))
check('root_workspace_split_equivalence',root_split==w_split,'Compare parsed rows, not relocation-dependent path hashes')
splits={k:{q['event_id'] for q in root_split if q['split']==k} for k in ['train','val','test']}
check('event_disjoint_36_12_12',[len(splits[k]) for k in ['train','val','test']]==[36,12,12] and sum(map(len,splits.values()))==len(set.union(*splits.values())),'One original split, no new split')

metrics={};rows=[];domains={};events=sorted(splits['test'])
for system in SYSTEMS:
    for seed in SEEDS:
        p=R/('historical12026' if seed<2029 else 'supplementary12026')/f'{system}_{seed}_reevaluation.json'
        q=read(p);check(f'{system}_{seed}_evaluation_stream',q['evaluation_seed']==12026 and q['shared_input_stream_sha256']==stream,'Uniform paper sample-identity stream')
        for raw,view in [('clean','clean'),('degraded','perturbed')]:
            m=q['metrics'][raw];metrics[system,seed,view]=m
            check(f'{system}_{seed}_{view}_events',set(m['per_event'])==splits['test'],'All12 held-out events, no duplicate event IDs')
            overall=np.array(m['overall']['confusion_matrix'],dtype=np.int64)
            for group in ['per_event','per_sensor']:
                check(f'{system}_{seed}_{view}_{group}_conservation',np.array_equal(overall,sum((np.array(x['confusion_matrix'],dtype=np.int64) for x in m[group].values()),start=np.zeros((2,2),dtype=np.int64))),
                      'Disjoint group counts sum to overall counts')
            for group in [dict(overall=m['overall']),m['per_event'],m['per_sensor']]:
                for name,v in group.items():
                    cm=np.array(v['confusion_matrix']);tp=int(cm[1,1]);fp=int(cm[0,1]);fn=int(cm[1,0]);ref=tp+fn
                    assert ref>0 and (cm>=0).all() and int(cm.sum())==v['valid_pixel_count']
                    assert close(v['flood_f1'],2*tp/(2*tp+fp+fn))
                    domain=(view,name);signature=(v['valid_pixel_count'],ref)
                    if domain in domains:assert domains[domain]==signature
                    else:domains[domain]=signature
            f1=np.array([m['per_event'][e]['flood_f1'] for e in events]);bias=[]
            for e in events:
                cm=m['per_event'][e]['confusion_matrix'];bias.append(100*(cm[0][1]-cm[1][0])/(cm[1][1]+cm[1][0]))
            rows.append(dict(system=system,seed=seed,condition=view,pooled_f1=m['overall']['flood_f1'],event_macro_f1=float(f1.mean()),extent_error_pct=float(np.abs(bias).mean())))
check('all_domains_and_count_derived_f1',True,'All15 checkpoints share reference foreground and valid domains; F1 recomputed from integer counts; zero reference denominators absent')
prior_csv=list(csv.DictReader((R/'five_seed_per_run_metrics.csv').open(encoding='utf-8-sig')))
for r in rows:
    p=next(q for q in prior_csv if q['system']==r['system'] and int(q['seed'])==r['seed'] and q['condition']==r['condition'])
    assert close(r['pooled_f1'],p['flood_f1']) and close(r['event_macro_f1'],p['event_macro_f1']) and close(r['extent_error_pct'],p['event_equal_absolute_extent_error_pct'])
check('exported_per_run_statistics',True,'Independent count-based recomputation agrees with all30 exported rows')
summary=[]
for system in SYSTEMS:
    for view in ['clean','perturbed']:
        rr=[r for r in rows if r['system']==system and r['condition']==view]
        for metric in ['pooled_f1','event_macro_f1','extent_error_pct']:
            vals=[r[metric] for r in rr];summary.append(dict(system=system,condition=view,metric=metric,n=5,mean=statistics.mean(vals),sample_sd=statistics.stdev(vals)))

pairs=[];event_pairs=[];loeo=[];subsets=[];rankings=[];partitions=[];cases=[]
comparisons=[('decoder','main_only'),('fpn','main_only'),('fpn','decoder')]
rng=np.random.default_rng(20261014)
# Fixed descriptive event bootstrap, matching the registered Phase1 resample count and RNG seed.
bootstrap_indices={n:rng.integers(0,n,size=(20000,n)) for n in [12,11,9]}
for view in ['clean','perturbed']:
    cube_f1=np.array([[[metrics[system,seed,view]['per_event'][e]['flood_f1'] for e in events] for seed in SEEDS] for system in SYSTEMS])
    cube_bias=np.empty_like(cube_f1)
    for si,system in enumerate(SYSTEMS):
        for ki,seed in enumerate(SEEDS):
            for ei,e in enumerate(events):
                cm=metrics[system,seed,view]['per_event'][e]['confusion_matrix'];cube_bias[si,ki,ei]=100*(cm[0][1]-cm[1][0])/(cm[1][1]+cm[1][0])
    for a,b in comparisons:
        ai,bi=SYSTEMS.index(a),SYSTEMS.index(b);diff=100*(cube_f1[ai]-cube_f1[bi]);ev=diff.mean(axis=0)
        rr_a=[next(r for r in rows if r['system']==a and r['seed']==seed and r['condition']==view) for seed in SEEDS]
        rr_b=[next(r for r in rows if r['system']==b and r['seed']==seed and r['condition']==view) for seed in SEEDS]
        pooled=np.array([100*(x['pooled_f1']-y['pooled_f1']) for x,y in zip(rr_a,rr_b)])
        extent=(np.abs(cube_bias[ai])-np.abs(cube_bias[bi])).mean(axis=1)
        ci=np.quantile(ev[bootstrap_indices[12]].mean(axis=1),[.025,.975])
        pairs.append(dict(comparison=f'{a}-{b}',condition=view,pooled_delta_mean_pp=float(pooled.mean()),pooled_delta_sample_sd_pp=float(pooled.std(ddof=1)),
                          pooled_positive_seeds=int((pooled>0).sum()),event_delta_mean_pp=float(ev.mean()),event_delta_seed_sample_sd_pp=float(diff.mean(axis=1).std(ddof=1)),
                          event_positive_seeds=int((diff.mean(axis=1)>0).sum()),positive_events=int((ev>0).sum()),negative_events=int((ev<0).sum()),
                          descriptive_event_bootstrap_low_pp=float(ci[0]),descriptive_event_bootstrap_high_pp=float(ci[1]),
                          extent_error_delta_mean_pp=float(extent.mean()),extent_improved_seeds=int((extent<0).sum())))
        for ei,e in enumerate(events):
            ext=float((np.abs(cube_bias[ai,:,ei])-np.abs(cube_bias[bi,:,ei])).mean())
            event_pairs.append(dict(comparison=f'{a}-{b}',condition=view,event=e,f1_delta_pp=float(ev[ei]),extent_error_delta_pp=ext,
                                    positive_f1_worse_extent=bool(ev[ei]>0 and ext>0)))
            keep=np.arange(12)!=ei;v=ev[keep];ci=np.quantile(v[bootstrap_indices[11]].mean(axis=1),[.025,.975])
            loeo.append(dict(comparison=f'{a}-{b}',condition=view,excluded=e,event_delta_mean_pp=float(v.mean()),
                             descriptive_low_pp=float(ci[0]),descriptive_high_pp=float(ci[1])))
        for label,excluded in [('remove_EMSR629',{'EMSR629'}),('remove_three_extremes',{'EMSR518','EMSR492','EMSR422'})]:
            keep=np.array([e not in excluded for e in events]);v=ev[keep]
            subsets.append(dict(comparison=f'{a}-{b}',condition=view,subset=label,n_events=int(keep.sum()),f1_delta_mean_pp=float(v.mean()),
                                extent_error_delta_mean_pp=float((np.abs(cube_bias[ai])-np.abs(cube_bias[bi]))[:,keep].mean())))
    x=cube_bias.mean(axis=1).T;grand=x.mean();event_ss=3*np.square(x.mean(axis=1)-grand).sum();path_ss=12*np.square(x.mean(axis=0)-grand).sum()
    interaction=np.square(x-x.mean(axis=1,keepdims=True)-x.mean(axis=0,keepdims=True)+grand).sum();total=np.square(x-grand).sum()
    partitions.append(dict(condition=view,definition='Descriptive SS partition of 12event x3path matrix after averaging five signed seed biases within each cell',
                           event_pct=float(100*event_ss/total),path_pct=float(100*path_ss/total),interaction_pct=float(100*interaction/total)))
    for metric,reverse in [('pooled_f1',True),('event_macro_f1',True),('extent_error_pct',False)]:
        means={s:statistics.mean(r[metric] for r in rows if r['system']==s and r['condition']==view) for s in SYSTEMS}
        rankings.append(dict(condition=view,criterion=metric,order=' > '.join(sorted(SYSTEMS,key=means.get,reverse=reverse)),values=json.dumps(means)))
    eidx=events.index('EMSR551')
    for si,system in enumerate(SYSTEMS):
        for ki,seed in enumerate(SEEDS):cases.append(dict(condition=view,system=system,seed=seed,event='EMSR551',f1=float(cube_f1[si,ki,eidx]),signed_bias_pct=float(cube_bias[si,ki,eidx]),absolute_error_pct=float(abs(cube_bias[si,ki,eidx]))))

for name,rr in [('protocol_matrix.csv',protocols),('existing_provenance_matrix.csv',source_matrix),('recomputed_summary.csv',summary),
                ('paired_effect_audit.csv',pairs),('event_pair_audit.csv',event_pairs),('leave_one_event_out.csv',loeo),
                ('event_exclusion_sensitivity.csv',subsets),('metric_rankings.csv',rankings),('descriptive_bias_partition.csv',partitions),('EMSR551_five_seed_case.csv',cases),('audit_checks.csv',checks)]:export(name,rr)
report=dict(scope='Three paths x five seeds2026-2030; current Decoder2029 only',status='PASS_FOR_UNIFIED_FIVE_SEED_DESCRIPTIVE_STATISTICS',
            checks_passed=len(checks),checks=checks,paired_effects=pairs,rankings=rankings,descriptive_bias_partition=partitions,
            statistical_unit='Five paired seeds for seed SD; event bootstrap first averages paired seed differences inside each of12 events',
            bootstrap=dict(resamples=20000,rng_seed=20261014,unit='event',meaning='descriptive split sensitivity, not new independent split'),
            no_training=True,no_model_inference=True,no_test_access=True,no_new_hashes=True,paper_modified=False,
            archive_source='Local frozen collections and existing checkpoint/source audit receipts; live historical fetch not needed for completed checks',
            remaining_scope='One event split; five-seed validation reversal, threshold transfer and hardware timing were not newly measured')
(O/'audit.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(dict(status=report['status'],checks=len(checks),perturbed_pairs=[r for r in pairs if r['condition']=='perturbed'],partition=partitions),ensure_ascii=True))
