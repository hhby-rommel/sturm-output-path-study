# Copyright (C) 2026 STURM output-path study contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
# Source-release notice added 2026-10-11; numerical code unchanged.
"""Local bounded learning diagnostic on frozen Sen1Floods11 train/val only."""
from pathlib import Path
import argparse, copy, gc, hashlib, json, random, sys, time, platform, os
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
import numpy as np
import torch
R=Path(__file__).resolve().parents[1]
P=R/'prepared_controls/sen1_gpu_prevalidation_20261005_v1'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def dump(p,x):
    q=p.with_suffix(p.suffix+'.tmp');q.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8');q.replace(p)
def rng():return {'python':random.getstate(),'numpy':np.random.get_state(),'torch':torch.get_rng_state(),'cuda':torch.cuda.get_rng_state_all()}
def restore(v):random.setstate(v['python']);np.random.set_state(v['numpy']);torch.set_rng_state(v['torch']);torch.cuda.set_rng_state_all(v['cuda'])

def exact_tree(a,b):
    if isinstance(a,torch.Tensor):return isinstance(b,torch.Tensor) and torch.equal(a.cpu(),b.cpu())
    if isinstance(a,np.ndarray):return isinstance(b,np.ndarray) and np.array_equal(a,b)
    if isinstance(a,dict):return a.keys()==b.keys() and all(exact_tree(a[k],b[k]) for k in a)
    if isinstance(a,(list,tuple)):return len(a)==len(b) and all(exact_tree(x,y) for x,y in zip(a,b))
    return a==b

def close_tree(a,b):
    if isinstance(a,dict):return a.keys()==b.keys() and all(close_tree(a[k],b[k]) for k in a)
    if isinstance(a,float):return abs(a-b)<=1e-6+1e-5*abs(a)
    return a==b

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);parser.add_argument('--preflight',action='store_true');parser.add_argument('--arm',required=True);args=parser.parse_args()
    plan=json.loads((P/'protocol.json').read_text(encoding='utf-8'))
    assert sha(P/'protocol.json')==(P/'protocol.sha256').read_text().strip()
    for f in plan['locked_files']:assert sha(R/f['path'])==f['sha256'],f['path']
    out=args.output.resolve();assert not out.exists();out.mkdir()
    dump(out/'claim.json',{'kind':'local train/val learning diagnostic only','protocol_sha256':sha(P/'protocol.json'),'test_access':False})
    torch.set_num_threads(2)
    assert torch.cuda.is_available(), 'CUDA unavailable; no fallback'
    assert torch.cuda.get_device_name(0)==plan['device']['required_name'], 'GPU identity mismatch'
    torch.backends.cudnn.allow_tf32=False;torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    torch.use_deterministic_algorithms(True)
    device=torch.device('cuda:0')
    selected=next(a for a in plan['arms'] if a['id']==args.arm)
    budget=plan['diagnostic_budget']['preflight_updates_per_arm'] if args.preflight else selected['updates']
    eval_steps=[0,budget] if args.preflight else selected['evaluate_at']
    dump(out/'gpu_environment.json',{'name':torch.cuda.get_device_name(0),'torch':torch.__version__,'CUDA':torch.version.cuda,'backend':plan['device'],'strict_recovery':True})
    sys.path.insert(0,str(R/plan['model_workspace']))
    from src.models.yolo26_unet_decoder import Yolo26UNetDecoder
    from src.models.yolo26_main_only_control import SharedYolo26MainOnly
    from src.models.yolo26_semantic import load_matching_yolo26_weights
    from src.models.unet_semantic import SturmPaperUNet
    from src.training.masked_segmentation import MaskedSemanticLoss, _resize_logits as resize_logits
    from src.training.degradation import degrade_sensor_batch
    from src.training.engine import _freeze_batchnorm_running_stats
    from src.training.optimization import OptimizationSpec, build_optimizer, build_scheduler
    from src.evaluation.formal import seed_everything,fixed_rng,is_better_checkpoint
    cache_dir=R/plan['cache'];index=json.loads((R/plan['tile_index']).read_text())
    xs=np.load(cache_dir/'images.npy',mmap_mode='r');ys=np.load(cache_dir/'targets.npy',mmap_mode='r');vs=np.load(cache_dir/'valid.npy',mmap_mode='r')
    tr=[i for i,q in enumerate(index) if q['partition']=='train'];va=[i for i,q in enumerate(index) if q['partition']=='val']
    assert len(tr)==4032 and len(va)==1424
    train_panel=sorted(tr,key=lambda i:hashlib.sha256(json.dumps(index[i],sort_keys=True).encode()).hexdigest())[:256]
    gen=torch.Generator().manual_seed(plan['diagnostic_seed']+1)
    sequence=np.load(R/plan['sequence']);assert sequence.shape==(1025,16)
    if args.preflight:train_panel=tr[:16];va=va[:16];tr_eval=tr[:16]
    else:tr_eval=tr
    np.save(out/'sampler_draws.npy',sequence)
    dump(out/'sampler_exposure.json',{'draws':budget*16,'unique_windows':len(set(sequence[:budget].flatten().tolist())),'sequence_sha256':sha(out/'sampler_draws.npy'),'groups':{g:sum(index[tr[int(i)]]['group']==g for i in sequence[:budget].flatten()) for g in sorted({index[i]['group'] for i in tr})}})
    def batch(ids):
        return {'image':torch.from_numpy(np.array(xs[ids],copy=True)),'mask':torch.from_numpy(np.array(ys[ids],dtype=np.int64)),'valid_mask':torch.from_numpy(np.array(vs[ids],copy=True)),'sensor':'S1','quality':torch.zeros(len(ids),4),'quality_available':torch.zeros(len(ids),4)}
    def move(b):return {k:v.to(device) if isinstance(v,torch.Tensor) else v for k,v in b.items()}
    def summarize(cm):
        tp,fp,fn,tn=cm
        return {'TP':tp,'FP':fp,'FN':fn,'TN':tn,'F1':2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0.,'precision':tp/(tp+fp) if tp+fp else 0.,'recall':tp/(tp+fn) if tp+fn else 0.,'prediction_fraction':(tp+fp)/(tp+fp+fn+tn) if tp+fp+fn+tn else 0.}
    def evaluate(m,criterion,ids):
        m.eval();result={}
        eval_BN_before={k:z.clone() for k,z in m.state_dict().items() if 'running_' in k or 'num_batches_tracked' in k}
        for view in ['clean','degraded']:
            cm=[0]*4;ces=[];dices=[];prob_fg=0.;prob_bg=0.;nfg=0;nbg=0;by_group={}
            with fixed_rng(plan['diagnostic_seed'],1),torch.inference_mode():
                for offset in range(0,len(ids),16):
                    b=batch(ids[offset:offset+16]);b=degrade_sensor_batch(b,'S1') if view=='degraded' else b;b=move(b)
                    logits=m.forward_native(b['image'],'S1');assert torch.isfinite(logits).all()
                    loss,parts=criterion(logits,b['mask'],b['valid_mask']);assert torch.isfinite(loss)
                    p=resize_logits(logits,b['mask'].shape[-2:]).argmax(1);v=b['valid_mask'];y=b['mask']
                    probs=resize_logits(logits,y.shape[-2:]).softmax(1)[:,1];fg=v&(y==1);bg=v&(y==0)
                    prob_fg+=float(probs[fg].double().sum());prob_bg+=float(probs[bg].double().sum());nfg+=int(fg.sum());nbg+=int(bg.sum())
                    cm=[old+int((v&v2).sum()) for old,v2 in zip(cm,[(p==1)&(y==1),(p==1)&(y==0),(p==0)&(y==1),(p==0)&(y==0)])]
                    for j,i in enumerate(ids[offset:offset+16]):
                        g=index[i]['group'];old=by_group.setdefault(g,[0]*4)
                        for a,z in enumerate([(p[j]==1)&(y[j]==1),(p[j]==1)&(y[j]==0),(p[j]==0)&(y[j]==1),(p[j]==0)&(y[j]==0)]):old[a]+=int((v[j]&z).sum())
                    ces.append(float(parts['main_ce_loss']));dices.append(float(parts['main_dice_loss']))
            result[view]={**summarize(cm),'foreground_probability_on_fg':prob_fg/nfg if nfg else 0.,'foreground_probability_on_bg':prob_bg/nbg if nbg else 0.,'valid_pixels':sum(cm),'by_filename_group':{g:summarize(counts) for g,counts in sorted(by_group.items())},'CE_batch_mean':float(np.mean(ces)),'Dice_batch_mean':float(np.mean(dices))}
        assert all(torch.equal(z,m.state_dict()[k]) for k,z in eval_BN_before.items()),'Evaluation mutated BN'
        return result
    def make(system,learning_rate):
        seed_everything(plan['diagnostic_seed'])
        if system=='unet':
            model=SturmPaperUNet(base_channels=64,num_classes=2,common_channels=2,native_input=True,dropout=.30,use_batchnorm=False);report={'source':'original random native S1 expert; no pretrained STURM weight'}
        else:
            cls=Yolo26UNetDecoder if system=='decoder' else SharedYolo26MainOnly
            model=cls(common_channels=16,num_classes=2)
            report=load_matching_yolo26_weights(model.core,R/plan['pretrained'])
        model=model.to(device)
        c={**plan['optimization'][system],'lr':learning_rate}
        spec=OptimizationSpec(version='Sen1_training_diagnostic_original_'+system,optimizer='adamw',learning_rate=c['lr'],weight_decay=c['weight_decay'],scheduler=c['scheduler'],warmup_fraction=c.get('warmup_fraction',0.),minimum_lr_ratio=c.get('minimum_lr_ratio',1.),exclude_norm_and_bias_from_weight_decay=c['exclude_norm_bias_decay'],track_parameter_update=True)
        opt=build_optimizer(model,spec);scheduler=build_scheduler(opt,spec,total_steps=c.get('total_steps',12600))
        return model,opt,scheduler,report
    def train_step(m,opt,sched,criterion,system,step):
        m.train()
        if system!='unet':
            _freeze_batchnorm_running_stats(m)
            modules=dict(m.named_modules())
            if arm['new_decoder_BN_train']:
                for name in plan['new_decoder_BN_names']:modules[name].train()
            assert all(z.training==(arm['new_decoder_BN_train'] and name in plan['new_decoder_BN_names']) for name,z in modules.items() if isinstance(z,torch.nn.modules.batchnorm._BatchNorm)),'Wrong BN route'
        b=batch([tr[int(i)] for i in sequence[step-1]])
        with fixed_rng(plan['diagnostic_seed']+step,71):d=degrade_sensor_batch(b,'S1')
        fingerprints={view:{k:hashlib.sha256(v.contiguous().numpy().tobytes()).hexdigest() for k,v in z.items() if isinstance(v,torch.Tensor)} for view,z in [('clean',b),('degraded',d)]}
        lr_before=opt.param_groups[0]['lr']
        b=move(b);d=move(d)
        opt.zero_grad(set_to_none=True)
        lc,pc=criterion(m.forward_native(b['image'],'S1'),b['mask'],b['valid_mask'])
        ld,pd=criterion(m.forward_native(d['image'],'S1'),d['mask'],d['valid_mask'])
        loss=lc+ld;assert torch.isfinite(loss);loss.backward()
        route_grad={name:float(torch.stack([p.grad.detach().double().square().sum() for k,p in m.named_parameters() if p.grad is not None and predicate(k)]).sum().sqrt()) for name,predicate in routes.items() if any(p.grad is not None and predicate(k) for k,p in m.named_parameters())}
        grad=torch.nn.utils.clip_grad_norm_(m.parameters(),5);assert torch.isfinite(grad)
        opt.step()
        if sched:sched.step()
        return {'input_fingerprints':fingerprints,'step':step,'loss':float(loss.detach()),'clean_CE':float(pc['main_ce_loss']),'clean_Dice':float(pc['main_dice_loss']),'degraded_CE':float(pd['main_ce_loss']),'degraded_Dice':float(pd['main_dice_loss']),'gradient_preclip':float(grad),'clipped':bool(grad>5),'lr':opt.param_groups[0]['lr'],'lr_before_update':lr_before,'warmup_steps_completed':min(step,378) if system=='unet' else 0,'warmup_total_steps':378 if system=='unet' else 0,'route_gradient_l2':route_grad}
    routes={'S1_bridge':lambda k:k.startswith('input_adapters.') and 'S1' in k,'core':lambda k:k.startswith('core.'),'unet_encoder':lambda k:k.startswith('encoders.'),'unet_bottleneck':lambda k:k.startswith('bottleneck.'),'unet_upconvs':lambda k:k.startswith('upconvs.'),'decoder':lambda k:k.startswith(('decoder','p1_lateral','p2_lateral','p3_lateral','input_lateral','fuse_','up_to_')),'head':lambda k:'head' in k,'S2_inactive':lambda k:'S2' in k}
    reports=[];common=None;initial_reference=None;start=time.perf_counter()
    try:
        for arm in [selected]:
            system=arm['system'];label=arm['id']
            folder=out/label;folder.mkdir();m,opt,sched,init=make(system,arm['lr'])
            if system=='decoder':common={k:v.detach().clone() for k,v in m.state_dict().items() if not k.startswith(('decoder','p1_lateral','p2_lateral','p3_lateral','input_lateral','fuse_','up_to_'))}
            elif system=='main_only' and common is not None:
                s=m.state_dict();same=[k for k,v in common.items() if k in s and s[k].shape==v.shape]
                assert same and all(torch.equal(common[k],s[k]) for k in same)
                init['common_identical_tensors']=len(same);del common
            initial_hashes={k:hashlib.sha256(z.detach().cpu().contiguous().numpy().tobytes()).hexdigest() for k,z in m.state_dict().items()}
            if initial_reference is None:initial_reference=initial_hashes
            else:assert initial_reference==initial_hashes,'Paired initialization mismatch'
            dump(folder/'initial_state_hashes.json',initial_hashes)
            common_hashes={k:v for k,v in initial_hashes.items() if not k.startswith(('decoder','p1_lateral','p2_lateral','p3_lateral','input_lateral','fuse_','up_to_'))}
            dump(folder/'common_initial_hashes.json',common_hashes)
            torch.cuda.reset_peak_memory_stats(device)
            active_initial={k:p.detach().clone() for k,p in m.named_parameters() if p.requires_grad}
            if not active_initial:active_initial={k:p.detach().clone() for k,p in list(m.named_parameters())[:1]}
            bn={k:v.clone() for k,v in m.state_dict().items() if 'running_' in k or 'num_batches_tracked' in k}
            criterion=MaskedSemanticLoss(aux_weight=.4,ignore_index=-1)
            history=[];evaluations=[];best=None;tic=time.perf_counter()
            for step in range(budget+1):
                if step:
                    torch.cuda.synchronize(device);tt=time.perf_counter();item=train_step(m,opt,sched,criterion,system,step);torch.cuda.synchronize(device);item['elapsed_seconds']=time.perf_counter()-tt;history.append(item)
                    dump(folder/'history.json',history)
                if step in eval_steps:
                    ev={'step':step,'train_scope':'fixed_256_window_panel' if not args.preflight else 'preflight16','train_panel':evaluate(m,criterion,train_panel),'full_train':evaluate(m,criterion,tr_eval) if step in [0,budget] else None,'val_panel':evaluate(m,criterion,va)};evaluations.append(ev)
                    if step>0:
                        candidate={'epoch':step,'degraded_flood_f1':ev['val_panel']['degraded']['F1'],'clean_flood_f1':ev['val_panel']['clean']['F1']}
                        if is_better_checkpoint(candidate,best):
                            best=candidate
                            torch.save({'model':m.state_dict(),'selection':best,'arm':arm,'protocol_sha256':sha(P/'protocol.json')},folder/'best_diagnostic.pt')
                    dump(folder/'evaluations.json',evaluations)
                    print(label,'step',step,'train_recall',round(ev['train_panel']['clean']['recall'],4),'train_degraded_F1',round(ev['train_panel']['degraded']['F1'],4),flush=True)
                dump(out/'status.json',{'arm':label,'system':system,'step':step,'elapsed_total_seconds':time.perf_counter()-start,'test_access':False})
            params=dict(m.named_parameters());delta={k:float((params[k].detach()-v).double().square().sum().sqrt()) for k,v in active_initial.items()}
            assert any(v>0 for v in delta.values()),'No measured active parameter update'
            changed_BN_keys=[k for k,z in bn.items() if not torch.equal(m.state_dict()[k],z)]
            assert all(any(k.startswith(n+'.') for n in plan['new_decoder_BN_names']) and arm['new_decoder_BN_train'] for k in changed_BN_keys),'Unexpected BN drift'
            if arm['new_decoder_BN_train']:
                modules=dict(m.named_modules())
                assert all(int(modules[n].num_batches_tracked)==budget*2 for n in plan['new_decoder_BN_names']),'Wrong new BN update count'
                assert changed_BN_keys,'New BN not updated'
            dump(folder/'BN_route_check.json',{'changed_keys':changed_BN_keys,'only_new_decoder_BN_allowed':arm['new_decoder_BN_train'],'expected_training_forwards':budget*2,'eval_BN_unchanged':True})
            checkpoint={'model':m.state_dict(),'optimizer':opt.state_dict(),'scheduler':sched.state_dict() if sched else None,'rng':rng(),'sampler_sequence':sequence,'sampler_cursor':budget,'protocol_sha256':sha(P/'protocol.json')}
            torch.save(checkpoint,folder/'last_diagnostic.pt')
            del checkpoint
            # Two executions of the same logical next update; excluded from the512-step learning budget.
            reference=train_step(m,opt,sched,criterion,system,budget+1)
            expected={k:v.detach().clone() for k,v in m.state_dict().items()}
            raw=torch.load(folder/'last_diagnostic.pt',map_location='cpu',weights_only=False)
            m.load_state_dict(raw['model'],strict=True);opt.load_state_dict(raw['optimizer'])
            if sched:sched.load_state_dict(raw['scheduler'])
            restored_model=exact_tree(raw['model'],m.state_dict());restored_optimizer=exact_tree(raw['optimizer'],opt.state_dict())
            restore(raw['rng']);restored_rng=exact_tree(raw['rng'],rng());del raw
            assert restored_model and restored_optimizer and restored_rng,'Exact saved-state reload failed'
            replay=train_step(m,opt,sched,criterion,system,budget+1)
            differences={k:float((v-m.state_dict()[k]).abs().max()) for k,v in expected.items() if not torch.equal(v,m.state_dict()[k])}
            model_close=all(torch.allclose(v,m.state_dict()[k],atol=1e-6,rtol=1e-5) for k,v in expected.items())
            recovery={'saved_model_exact':restored_model,'saved_optimizer_exact':restored_optimizer,'saved_rng_exact':restored_rng,'next_input_exact':reference['input_fingerprints']==replay['input_fingerprints'],'next_model_exact':not differences,'next_metrics_exact':reference==replay,'next_model_within_tolerance':model_close,'next_metrics_within_tolerance':close_tree(reference,replay),'max_parameter_abs_difference':max(differences.values(),default=0.),'different_tensors':differences,'reference':reference,'replay':replay,'tolerance':{'atol':1e-6,'rtol':1e-5}}
            dump(folder/'recovery.json',recovery)
            assert recovery['next_model_exact'] and recovery['next_metrics_exact'],'GPU exact next-update recovery failed; no tolerance fallback'
            record={'system':system,'updates':budget,'arm':arm,'probe_update_executions':2,'resume_next_update_equal':recovery['next_model_exact'] and recovery['next_metrics_exact'],'resume_numerical_pass':True,'BN_unchanged':not changed_BN_keys,'BN_route_pass':True,'active_parameter_delta_l2':delta,'initialization':init,'best_diagnostic_step':best,'elapsed_seconds':time.perf_counter()-tic,'mean_step_seconds':float(np.mean([x['elapsed_seconds'] for x in history])),'clipped_fraction':float(np.mean([x['clipped'] for x in history])),'learning_summary':{'initial_train_recall':evaluations[0]['train_panel']['clean']['recall'],'final_train_recall':evaluations[-1]['train_panel']['clean']['recall'],'initial_val_degraded_F1':evaluations[0]['val_panel']['degraded']['F1'],'final_val_degraded_F1':evaluations[-1]['val_panel']['degraded']['F1']},'checkpoint_sha256':sha(folder/'last_diagnostic.pt')}
            record['memory_bytes']={'peak_allocated':torch.cuda.max_memory_allocated(device),'peak_reserved':torch.cuda.max_memory_reserved(device)}
            record['learning_admission']=any(ev['step']>0 and ev['train_panel']['clean']['F1']>.10 and ev['train_panel']['clean']['recall']>.10 and ev['train_panel']['clean']['foreground_probability_on_fg']>ev['train_panel']['clean']['foreground_probability_on_bg'] for ev in evaluations)
            dump(folder/'completion.json',record);reports.append(record)
            del m,opt,sched,expected,criterion,active_initial;gc.collect()
        dump(out/'completion.json',{'status':'SEN1_GPU_ARM_PREVALIDATION_COMPLETE','reports':reports,'elapsed_total_seconds':time.perf_counter()-start,'python':sys.version,'torch':torch.__version__,'platform':platform.platform(),'preflight':args.preflight,'device':plan['device'],'second_seed_started':False,'test_access':False,'formal_performance_reproduced':False,'formal_launch_authorized_by_this_diagnostic':False})
    except Exception as e:
        dump(out/'failure.json',{'type':type(e).__name__,'message':str(e),'test_access':False});raise
if __name__=='__main__':main()
