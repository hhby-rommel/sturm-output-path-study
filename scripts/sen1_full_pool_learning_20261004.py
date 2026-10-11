# Copyright (C) 2026 STURM output-path study contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Local bounded learning diagnostic on frozen Sen1Floods11 train/val only."""
from pathlib import Path
import argparse, copy, gc, hashlib, json, random, sys, time, platform
import numpy as np
import torch
R=Path(__file__).resolve().parents[1]
P=R/'prepared_controls/sen1_full_pool_diagnostic_20261004_v1'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def dump(p,x):
    q=p.with_suffix(p.suffix+'.tmp');q.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8');q.replace(p)
def rng():return {'python':random.getstate(),'numpy':np.random.get_state(),'torch':torch.get_rng_state()}
def restore(v):random.setstate(v['python']);np.random.set_state(v['numpy']);torch.set_rng_state(v['torch'])
def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    plan=json.loads((P/'protocol.json').read_text(encoding='utf-8'))
    assert sha(P/'protocol.json')==(P/'protocol.sha256').read_text().strip()
    for f in plan['locked_files']:assert sha(R/f['path'])==f['sha256'],f['path']
    out=args.output.resolve();assert not out.exists();out.mkdir()
    dump(out/'claim.json',{'kind':'local train/val learning diagnostic only','protocol_sha256':sha(P/'protocol.json'),'test_access':False})
    torch.set_num_threads(2)
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
    sequence=torch.randint(len(tr),(513,16),generator=gen).numpy()
    np.save(out/'sampler_draws.npy',sequence)
    dump(out/'sampler_exposure.json',{'draws':8192,'unique_windows':len(set(sequence[:512].flatten().tolist())),'sequence_sha256':sha(out/'sampler_draws.npy'),'groups':{g:sum(index[tr[int(i)]]['group']==g for i in sequence[:512].flatten()) for g in sorted({index[i]['group'] for i in tr})}})
    def batch(ids):
        return {'image':torch.from_numpy(np.array(xs[ids],copy=True)),'mask':torch.from_numpy(np.array(ys[ids],dtype=np.int64)),'valid_mask':torch.from_numpy(np.array(vs[ids],copy=True)),'sensor':'S1','quality':torch.zeros(len(ids),4),'quality_available':torch.zeros(len(ids),4)}
    def summarize(cm):
        tp,fp,fn,tn=cm
        return {'TP':tp,'FP':fp,'FN':fn,'TN':tn,'F1':2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0.,'precision':tp/(tp+fp) if tp+fp else 0.,'recall':tp/(tp+fn) if tp+fn else 0.,'prediction_fraction':(tp+fp)/(tp+fp+fn+tn) if tp+fp+fn+tn else 0.}
    def evaluate(m,criterion,ids):
        m.eval();result={}
        for view in ['clean','degraded']:
            cm=[0]*4;ces=[];dices=[];prob_fg=0.;prob_bg=0.;nfg=0;nbg=0;by_group={}
            with fixed_rng(plan['diagnostic_seed'],1),torch.inference_mode():
                for offset in range(0,len(ids),16):
                    b=batch(ids[offset:offset+16]);b=degrade_sensor_batch(b,'S1') if view=='degraded' else b
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
        return result
    def make(system):
        seed_everything(plan['diagnostic_seed'])
        if system=='unet':
            model=SturmPaperUNet(base_channels=64,num_classes=2,common_channels=2,native_input=True,dropout=.30,use_batchnorm=False);report={'source':'original random native S1 expert; no pretrained STURM weight'}
        else:
            cls=Yolo26UNetDecoder if system=='decoder' else SharedYolo26MainOnly
            model=cls(common_channels=16,num_classes=2)
            report=load_matching_yolo26_weights(model.core,R/plan['pretrained'])
        c=plan['optimization'][system]
        spec=OptimizationSpec(version='Sen1_training_diagnostic_original_'+system,optimizer='adamw',learning_rate=c['lr'],weight_decay=c['weight_decay'],scheduler=c['scheduler'],warmup_fraction=c.get('warmup_fraction',0.),minimum_lr_ratio=c.get('minimum_lr_ratio',1.),exclude_norm_and_bias_from_weight_decay=c['exclude_norm_bias_decay'],track_parameter_update=True)
        opt=build_optimizer(model,spec);scheduler=build_scheduler(opt,spec,total_steps=c.get('total_steps',12600))
        return model,opt,scheduler,report
    def train_step(m,opt,sched,criterion,system,step):
        m.train()
        if system!='unet':_freeze_batchnorm_running_stats(m)
        b=batch([tr[int(i)] for i in sequence[step-1]])
        with fixed_rng(plan['diagnostic_seed']+step,71):d=degrade_sensor_batch(b,'S1')
        lr_before=opt.param_groups[0]['lr']
        opt.zero_grad(set_to_none=True)
        lc,pc=criterion(m.forward_native(b['image'],'S1'),b['mask'],b['valid_mask'])
        ld,pd=criterion(m.forward_native(d['image'],'S1'),d['mask'],d['valid_mask'])
        loss=lc+ld;assert torch.isfinite(loss);loss.backward()
        route_grad={name:float(torch.stack([p.grad.detach().double().square().sum() for k,p in m.named_parameters() if p.grad is not None and predicate(k)]).sum().sqrt()) for name,predicate in routes.items() if any(p.grad is not None and predicate(k) for k,p in m.named_parameters())}
        grad=torch.nn.utils.clip_grad_norm_(m.parameters(),5);assert torch.isfinite(grad)
        opt.step()
        if sched:sched.step()
        return {'step':step,'loss':float(loss.detach()),'clean_CE':float(pc['main_ce_loss']),'clean_Dice':float(pc['main_dice_loss']),'degraded_CE':float(pd['main_ce_loss']),'degraded_Dice':float(pd['main_dice_loss']),'gradient_preclip':float(grad),'clipped':bool(grad>5),'lr':opt.param_groups[0]['lr'],'lr_before_update':lr_before,'warmup_steps_completed':min(step,378) if system=='unet' else 0,'warmup_total_steps':378 if system=='unet' else 0,'route_gradient_l2':route_grad}
    routes={'S1_bridge':lambda k:k.startswith('input_adapters.') and 'S1' in k,'core':lambda k:k.startswith('core.'),'decoder':lambda k:k.startswith(('decoder','p1_lateral','p2_lateral','p3_lateral','input_lateral','fuse_','up_to_')),'head':lambda k:'head' in k,'S2_inactive':lambda k:'S2' in k}
    reports=[];common=None;start=time.perf_counter()
    try:
        for system in ['decoder','main_only','unet']:
            folder=out/system;folder.mkdir();m,opt,sched,init=make(system)
            if system=='decoder':common={k:v.detach().clone() for k,v in m.state_dict().items() if not k.startswith(('decoder','p1_lateral','p2_lateral','p3_lateral','input_lateral','fuse_','up_to_'))}
            elif system=='main_only':
                s=m.state_dict();same=[k for k,v in common.items() if k in s and s[k].shape==v.shape]
                assert same and all(torch.equal(common[k],s[k]) for k in same)
                init['common_identical_tensors']=len(same);del common
            active_initial={k:p.detach().clone() for k,p in m.named_parameters() if p.requires_grad}
            if not active_initial:active_initial={k:p.detach().clone() for k,p in list(m.named_parameters())[:1]}
            bn={k:v.clone() for k,v in m.state_dict().items() if 'running_' in k or 'num_batches_tracked' in k}
            criterion=MaskedSemanticLoss(aux_weight=.4,ignore_index=-1)
            history=[];evaluations=[];best=None;tic=time.perf_counter()
            for step in range(513):
                if step:
                    tt=time.perf_counter();item=train_step(m,opt,sched,criterion,system,step);item['elapsed_seconds']=time.perf_counter()-tt;history.append(item)
                    dump(folder/'history.json',history)
                if step in plan['diagnostic_budget']['evaluate_at']:
                    ev={'step':step,'train_scope':'complete_train' if step in [0,512] else 'fixed_256_window_panel','train_panel':evaluate(m,criterion,tr if step in [0,512] else train_panel),'val_panel':evaluate(m,criterion,va) if step in [0,378,512] else None};evaluations.append(ev)
                    if ev['val_panel'] is not None:
                        candidate={'epoch':step,'degraded_flood_f1':ev['val_panel']['degraded']['F1'],'clean_flood_f1':ev['val_panel']['clean']['F1']}
                        if is_better_checkpoint(candidate,best):best=candidate
                    dump(folder/'evaluations.json',evaluations)
                    print(system,'step',step,'train_recall',round(ev['train_panel']['clean']['recall'],4),'train_degraded_F1',round(ev['train_panel']['degraded']['F1'],4),flush=True)
                dump(out/'status.json',{'system':system,'step':step,'elapsed_total_seconds':time.perf_counter()-start,'test_access':False})
            params=dict(m.named_parameters());delta={k:float((params[k].detach()-v).double().square().sum().sqrt()) for k,v in active_initial.items()}
            assert any(v>0 for v in delta.values()),'No measured active parameter update'
            assert all(torch.equal(m.state_dict()[k],v) for k,v in bn.items()),'BN state drift'
            checkpoint={'model':m.state_dict(),'optimizer':opt.state_dict(),'scheduler':sched.state_dict() if sched else None,'rng':rng(),'sampler_sequence':sequence,'sampler_cursor':512,'protocol_sha256':sha(P/'protocol.json')}
            torch.save(checkpoint,folder/'last_diagnostic.pt')
            del checkpoint
            # Two executions of the same logical next update; excluded from the512-step learning budget.
            reference=train_step(m,opt,sched,criterion,system,513)
            expected={k:v.detach().clone() for k,v in m.state_dict().items()}
            raw=torch.load(folder/'last_diagnostic.pt',map_location='cpu',weights_only=False)
            m.load_state_dict(raw['model'],strict=True);opt.load_state_dict(raw['optimizer'])
            if sched:sched.load_state_dict(raw['scheduler'])
            restore(raw['rng']);del raw
            replay=train_step(m,opt,sched,criterion,system,513)
            assert all(torch.equal(v,m.state_dict()[k]) for k,v in expected.items()) and reference==replay,'Resume next-update mismatch'
            record={'system':system,'updates':512,'probe_update_executions':2,'resume_next_update_equal':True,'BN_unchanged':True,'active_parameter_delta_l2':delta,'initialization':init,'best_diagnostic_step':best,'elapsed_seconds':time.perf_counter()-tic,'mean_step_seconds':float(np.mean([x['elapsed_seconds'] for x in history])),'clipped_fraction':float(np.mean([x['clipped'] for x in history])),'learning_summary':{'initial_train_recall':evaluations[0]['train_panel']['clean']['recall'],'final_train_recall':evaluations[-1]['train_panel']['clean']['recall'],'initial_val_degraded_F1':evaluations[0]['val_panel']['degraded']['F1'],'final_val_degraded_F1':evaluations[-1]['val_panel']['degraded']['F1']},'checkpoint_sha256':sha(folder/'last_diagnostic.pt')}
            dump(folder/'completion.json',record);reports.append(record)
            del m,opt,sched,expected,criterion,active_initial;gc.collect()
        dump(out/'completion.json',{'status':'FULL_POOL_THREE_MODEL_DIAGNOSTIC_COMPLETE','reports':reports,'elapsed_total_seconds':time.perf_counter()-start,'python':sys.version,'torch':torch.__version__,'platform':platform.platform(),'test_access':False,'formal_performance_reproduced':False,'formal_launch_authorized_by_this_diagnostic':False})
    except Exception as e:
        dump(out/'failure.json',{'type':type(e).__name__,'message':str(e),'test_access':False});raise
if __name__=='__main__':main()
