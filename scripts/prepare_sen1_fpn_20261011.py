# Copyright (C) 2026 STURM output-path study contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
# Source-release notice added 2026-10-11; numerical code unchanged.
from pathlib import Path
import json,copy,hashlib,shutil,tarfile
R=Path(__file__).resolve().parents[1];P=R/'prepared_controls/sen1_fpn_20261011_v1';P.mkdir(exist_ok=False)
parent=R/'prepared_controls/sen1_formal_20261005_v1';plan=json.loads((parent/'protocol.json').read_text(encoding='utf-8'));sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
base=copy.deepcopy(plan)
plan['tasks']=[dict(next(q for q in base['tasks'] if q['id']==f'decoder_seed{seed}'),id=f'fpn_seed{seed}',system='fpn',ordinal=i+1) for i,seed in enumerate([2026,2027,2028])]
plan.update(status='PREPARED_AUTHORIZED_PENDING_ADMISSION',study='Sen1Floods11 FPN external S1 extension, original decoder recipe policy',parent_protocol_sha256=sha(parent/'protocol.json'),model_workspace='prepared_controls/sen1_fpn_20261011_v1/workspace',maximum_training_updates=37800,maximum_sample_exposures=604800)
plan['optimization']={'fpn':copy.deepcopy(base['optimization']['decoder'])}
plan['numeric']['BN']='Backbone/neck BN frozen; only4 new FPN smoothing BN train; same policy as original Sen1 decoder'
plan['numeric']['max_epochs']={'fpn':50}
plan['new_decoder_BN_names']=[f'fpn_smooth.{name}.1' for name in ['neck_p3','p2','p1','input']]
plan['BN_policy']={'fpn':'Freeze all original BN; update only four new FPN head BN clean then degraded; affine gradients retained'}
plan['initialization']='Original Cityscapes transfer into shared core; same seed construct; FPN width34 is pre-existing STURM reference, no accuracy search'
plan['authorization_state']={'formal_launch_authorized':True,'test_execution_authorized':True,'source':'User20261011 explicitly restore Sen1-FPN three seeds under original serial protocol'}
plan['execution_entry_status']='Source ready; strict GPU two-step replay admission required before each seed'
plan['cost_estimate']={'status':'Estimate after admission/first epoch; max50/patience10 fixed, no result-based extension'}
plan['compatibility_note']='FPN has four new BN modules instead of decoder six; adapt module names to architecture, preserve new-head-only BN policy. Strict deterministic CUDA bilinear backward gate passed on original torch2.1.0+cu121; no backend change.'
workspace=P/'workspace';shutil.copytree(R/base['model_workspace']/'src',workspace/'src',ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
shutil.copy2(R/'src/models/yolo26_fpn_reference.py',workspace/'src/models/yolo26_fpn_reference.py')
for file in ['test_manifest.json','storage_budget.json']:shutil.copy2(parent/file,P/file)
plan['locked_files']=base['locked_files'];(P/'protocol.json').write_text(json.dumps(plan,ensure_ascii=False,indent=2),encoding='utf-8')
s=(R/'scripts/sen1_formal_task_20261005.py').read_text(encoding='utf-8')
s=s.replace("P=R/'prepared_controls/sen1_formal_20261005_v1'","P=R/'prepared_controls/sen1_fpn_20261011_v1'")
s=s.replace("choices=['audit','train','test']","choices=['audit','admission','train','test']")
s=s.replace("{'decoder':20086342,'main_only':19766120,'unet':414365486}[system]","{'decoder':20086342,'fpn':21000000,'main_only':19766120,'unet':414365486}[system]")
s=s.replace('from src.models.yolo26_unet_decoder import Yolo26UNetDecoder','from src.models.yolo26_fpn_reference import Yolo26FPNReference\n    from src.models.yolo26_unet_decoder import Yolo26UNetDecoder')
s=s.replace("cls=Yolo26UNetDecoder if system=='decoder' else SharedYolo26MainOnly","cls=Yolo26FPNReference if system=='fpn' else Yolo26UNetDecoder if system=='decoder' else SharedYolo26MainOnly")
s=s.replace("routes={'S1_bridge':", "routes={'fpn':lambda k:k.startswith(('fpn_laterals.','fpn_smooth.','fpn_head.')),'S1_bridge':")
needle="    if args.mode=='audit':"
admission="""    if args.mode=='admission':
        out.mkdir(parents=True,exist_ok=False)
        initial_state=copy.deepcopy(m.state_dict());bn_before={k:v.clone() for k,v in m.state_dict().items() if 'running_' in k or 'num_batches_tracked' in k}
        initial_rng=rng();tic=time.perf_counter();first=[train_step(m,opt,sched,criterion,system,i) for i in [1,2]];torch.cuda.synchronize();elapsed=time.perf_counter()-tic
        state1=copy.deepcopy(m.state_dict());opt1=copy.deepcopy(opt.state_dict())
        allowed=plan['new_decoder_BN_names']
        changed=[k for k,v in bn_before.items() if not torch.equal(v,state1[k])]
        assert changed and all(any(k.startswith(name+'.') for name in allowed) for k in changed)
        assert all(int(dict(m.named_modules())[n].num_batches_tracked)==4 for n in allowed)
        assert all(item['route_gradient_l2'].get('fpn',0)>0 and item['route_gradient_l2'].get('core',0)>0 for item in first)
        # Reconstruct from the original initialization; audit updates never warm-start formal training.
        m2,opt2,sched2,_=make(system,plan['optimization'][system]['lr']);m2.load_state_dict(initial_state,strict=True);restore(initial_rng)
        second=[train_step(m2,opt2,sched2,criterion,system,i) for i in [1,2]]
        assert exact_tree(state1,m2.state_dict()) and exact_tree(opt1,opt2.state_dict()),'Strict two-step replay differed'
        assert all(a['input_fingerprints']==b['input_fingerprints'] for a,b in zip(first,second))
        dump(out/'admission.json',dict(status='PASS',task=selected,updates_per_replay=2,formal_warm_start=False,test_access=False,strict_replay_equal=True,BN_changed_only_new_FPN=True,BN_names=allowed,pretrained_load=init,initial_state_hashes=initialization,input_fingerprints=[x['input_fingerprints'] for x in first],step_seconds_mean=elapsed/2,steps=first,protocol_sha256=sha(P/'protocol.json'),source_sha256=sha(Path(__file__))))
        print('SEN1_FPN_ADMISSION_PASS',selected['seed'],elapsed/2,flush=True);return
"""
assert s.count(needle)==1;s=s.replace(needle,admission+needle)
(R/'scripts/sen1_fpn_task_20261011.py').write_text(s,encoding='utf-8')
lock=json.loads((parent/'execution_lock.json').read_text());files=copy.deepcopy(lock['files'])
files.extend({'path':str(p.relative_to(R)).replace('\\','/'),'sha256':sha(p)} for p in workspace.rglob('*') if p.is_file())
for p in [R/'scripts/sen1_fpn_task_20261011.py',P/'test_manifest.json']+[R/q['path'] for q in plan['sequence_files']]:files.append({'path':str(p.relative_to(R)).replace('\\','/'),'sha256':sha(p)})
(P/'execution_lock.json').write_text(json.dumps({'protocol_sha256':sha(P/'protocol.json'),'files':files},indent=2),encoding='utf-8')
(P/'authorization.json').write_text(json.dumps({'formal_launch_authorized':True,'once_test_authorized':True,'Bolivia_included':False,'protocol_sha256':sha(P/'protocol.json'),'execution_lock_sha256':sha(P/'execution_lock.json'),'user_instruction':'恢复Sen1-FPN三个种子，按原协议串行'},ensure_ascii=False,indent=2),encoding='utf-8')
with tarfile.open(P/'deployment.tar.gz','w:gz') as z:
 for p in P.rglob('*'):
  if p.is_file() and p.name!='deployment.tar.gz':z.add(p,arcname=str(p.relative_to(R)).replace('\\','/'))
 z.add(R/'scripts/sen1_fpn_task_20261011.py',arcname='scripts/sen1_fpn_task_20261011.py')
print(json.dumps({'folder':str(P),'protocol_SHA256':sha(P/'protocol.json'),'seeds':[2026,2027,2028],'max_epochs':50,'patience':10},ensure_ascii=False))
