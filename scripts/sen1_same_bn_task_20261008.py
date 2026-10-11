# Copyright (C) 2026 STURM output-path study contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
# Source-release notice added 2026-10-11; numerical code unchanged.
"""Independent all-BN-frozen Sen1 train/val control; test path removed."""
from pathlib import Path
import argparse, copy, gc, hashlib, json, random, sys, time, platform, os
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
import numpy as np
import torch
R=Path(__file__).resolve().parents[1]
P=R/'prepared_controls/sen1_same_bn_20261008_v1'
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
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);parser.add_argument('--task',required=True);parser.add_argument('--mode',choices=['audit','train'],required=True);parser.add_argument('--execute-authorized',action='store_true');args=parser.parse_args()
    if not args.execute_authorized:parser.error('Prepared only; execution requires later user authorization and explicit flag')
    plan=json.loads((P/'protocol.json').read_text());lock=json.loads((P/'execution_lock.json').read_text())
    assert sha(P/'protocol.json')==lock['protocol_sha256']
    for record in lock['files']:
        assert sha(R/record['path'])==record['sha256'], ('Source/input lock changed',record['path'])
    selected=next(t for t in plan['tasks'] if t['id']==args.task);system=selected['system'];arm=selected
    import shutil
    torch.set_num_threads(2);assert torch.cuda.is_available() and torch.cuda.get_device_name(0)==plan['device']['required_name']
    torch.backends.cudnn.allow_tf32=False;torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True;torch.use_deterministic_algorithms(True)
    device=torch.device('cuda:0');out=args.output.resolve();assert out.is_relative_to(R/'results')
    def capacity(extra):assert shutil.disk_usage(R).free >= int(extra*1.15)+2*1024**3, 'Insufficient storage; preserve evidence and stop'
    def atomic_save(value,path):
        measured={'decoder':20086342,'main_only':19766120,'unet':414365486}[system];capacity(measured)
        temp=path.with_suffix(path.suffix+'.tmp');assert not temp.exists(),'Stale temporary checkpoint; no retry';torch.save(value,temp);temp.replace(path)
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
    cache_dir=R/plan['train_val_cache'];index=json.loads((cache_dir/'tile_index.json').read_text())
    xs=np.load(cache_dir/'images.npy',mmap_mode='r');ys=np.load(cache_dir/'targets.npy',mmap_mode='r');vs=np.load(cache_dir/'valid.npy',mmap_mode='r')
    tr=[i for i,q in enumerate(index) if q['partition']=='train'];va=[i for i,q in enumerate(index) if q['partition']=='val']
    assert len(tr)==4032 and len(va)==1424
    train_panel=sorted(tr,key=lambda i:hashlib.sha256(json.dumps(index[i],sort_keys=True).encode()).hexdigest())[:256]
    sequence=np.load(R/next(q['path'] for q in plan['sequence_files'] if q['seed']==selected['seed']))
    assert sequence.shape==(12600,16)
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
            cm=[0]*4;ces=[];dices=[];prob_fg=0.;prob_bg=0.;nfg=0;nbg=0;by_group={};by_chip={};domain_hash=hashlib.sha256();input_hash=hashlib.sha256()
            with fixed_rng(12026,1),torch.inference_mode():
                for offset in range(0,len(ids),16):
                    b=batch(ids[offset:offset+16]);b=degrade_sensor_batch(b,'S1') if view=='degraded' else b;b=move(b)
                    domain_hash.update(b['valid_mask'].detach().cpu().contiguous().numpy().tobytes());input_hash.update(b['image'].detach().cpu().contiguous().numpy().tobytes())
                    logits=m.forward_native(b['image'],'S1');assert torch.isfinite(logits).all()
                    loss,parts=criterion(logits,b['mask'],b['valid_mask']);assert torch.isfinite(loss)
                    p=resize_logits(logits,b['mask'].shape[-2:]).argmax(1);v=b['valid_mask'];y=b['mask']
                    probs=resize_logits(logits,y.shape[-2:]).softmax(1)[:,1];fg=v&(y==1);bg=v&(y==0)
                    prob_fg+=float(probs[fg].double().sum());prob_bg+=float(probs[bg].double().sum());nfg+=int(fg.sum());nbg+=int(bg.sum())
                    cm=[old+int((v&v2).sum()) for old,v2 in zip(cm,[(p==1)&(y==1),(p==1)&(y==0),(p==0)&(y==1),(p==0)&(y==0)])]
                    for j,i in enumerate(ids[offset:offset+16]):
                        g=index[i]['group'];old=by_group.setdefault(g,[0]*4)
                        chip_counts=by_chip.setdefault(index[i]['chip'],[0]*4)
                        for a,z in enumerate([(p[j]==1)&(y[j]==1),(p[j]==1)&(y[j]==0),(p[j]==0)&(y[j]==1),(p[j]==0)&(y[j]==0)]):
                            count=int((v[j]&z).sum());old[a]+=count;chip_counts[a]+=count
                    ces.append(float(parts['main_ce_loss']));dices.append(float(parts['main_dice_loss']))
            result[view]={**summarize(cm),'foreground_probability_on_fg':prob_fg/nfg if nfg else 0.,'foreground_probability_on_bg':prob_bg/nbg if nbg else 0.,'valid_pixels':sum(cm),'by_filename_group':{g:summarize(counts) for g,counts in sorted(by_group.items())},'by_chip':{g:summarize(counts) for g,counts in sorted(by_chip.items())},'group_macro_F1':float(np.mean([summarize(counts)['F1'] for counts in by_group.values()])), 'input_sha256':input_hash.hexdigest(),'valid_domain_sha256':domain_hash.hexdigest(),'CE_batch_mean':float(np.mean(ces)),'Dice_batch_mean':float(np.mean(dices))}
        assert all(torch.equal(z,m.state_dict()[k]) for k,z in eval_BN_before.items()),'Evaluation mutated BN'
        return result
    def make(system,learning_rate):
        seed_everything(selected['seed'])
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
        with fixed_rng(selected['seed']+step,71):d=degrade_sensor_batch(b,'S1')
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
    m,opt,sched,init=make(system,plan['optimization'][system]['lr'])
    initialization={k:hashlib.sha256(v.detach().cpu().contiguous().numpy().tobytes()).hexdigest() for k,v in m.state_dict().items()}
    if system!='unet':assert init['matched_keys']==359 and init['skipped_shape_keys']==5 and init['skipped_name_keys']==0
    criterion=MaskedSemanticLoss(aux_weight=.4,ignore_index=-1)
    if args.mode=='audit':
        out.mkdir(parents=True,exist_ok=False);m.train()
        if system!='unet':
            _freeze_batchnorm_running_stats(m)
            for name,module in m.named_modules():
                if selected['new_decoder_BN_train'] and name in plan['new_decoder_BN_names']:module.train()
            assert all(z.training==(selected['new_decoder_BN_train'] and n in plan['new_decoder_BN_names']) for n,z in m.named_modules() if isinstance(z,torch.nn.modules.batchnorm._BatchNorm))
        dump(out/(selected['id']+'_audit.json'),{'task':selected,'initial_state_hashes':initialization,'pretrained_load':init,'source_sha256':sha(Path(__file__)),'no_updates':True,'no_test_pixels':True,'sequence_sha256':sha(R/next(q['path'] for q in plan['sequence_files'] if q['seed']==selected['seed']))})
        return
    if args.mode=='train':
        assert not out.exists(),'Refuse overwrite/resume';out.mkdir()
        dump(out/'train_claim.json',{'task':selected,'protocol_sha256':sha(P/'protocol.json'),'source_sha256':sha(Path(__file__)),'test_access':False})
        dump(out/'initialization.json',{'state_hashes':initialization,'pretrained_load':init})
        bn_initial={k:v.clone() for k,v in m.state_dict().items() if 'running_' in k or 'num_batches_tracked' in k}
        start=time.perf_counter();history=[];best=None;bad=0;step=0;exposure=set();input_chain=hashlib.sha256()
        try:
            for epoch in range(1,selected['max_epochs']+1):
                epoch_steps=[]
                for _ in range(252):
                    step+=1;torch.cuda.synchronize();tic=time.perf_counter();item=train_step(m,opt,sched,criterion,system,step);torch.cuda.synchronize();item['elapsed_seconds']=time.perf_counter()-tic
                    input_chain.update(json.dumps(item['input_fingerprints'],sort_keys=True).encode());exposure.update(sequence[step-1].tolist());epoch_steps.append(item)
                    with (out/'steps.jsonl').open('a') as f:f.write(json.dumps(item)+'\n')
                    if step%16==0:dump(out/'status.json',{'epoch':epoch,'step':step,'task':selected['id'],'elapsed_seconds':time.perf_counter()-start,'test_access':False})
                val=evaluate(m,criterion,va)
                candidate={'epoch':epoch,'degraded_flood_f1':val['degraded']['F1'],'clean_flood_f1':val['clean']['F1']}
                improved=is_better_checkpoint(candidate,best)
                if improved:
                    best=candidate;bad=0;atomic_save({'model':m.state_dict(),'selection':best,'task':selected,'protocol_sha256':sha(P/'protocol.json'),'source_sha256':sha(Path(__file__))},out/'best.pt')
                else:bad+=1
                changed=[k for k,v in bn_initial.items() if not torch.equal(v,m.state_dict()[k])]
                assert all(selected['new_decoder_BN_train'] and any(k.startswith(n+'.') for n in plan['new_decoder_BN_names']) for k in changed),'Unexpected BN drift'
                if selected['new_decoder_BN_train']:assert all(int(dict(m.named_modules())[n].num_batches_tracked)==2*step for n in plan['new_decoder_BN_names'])
                rec={'epoch':epoch,'step':step,'train_loss_mean':float(np.mean([q['loss'] for q in epoch_steps])),'train_CE_clean':float(np.mean([q['clean_CE'] for q in epoch_steps])),'train_Dice_clean':float(np.mean([q['clean_Dice'] for q in epoch_steps])),'clip_fraction':float(np.mean([q['clipped'] for q in epoch_steps])),'lr':opt.param_groups[0]['lr'],'val':val,'best':best,'patience_used':bad,'elapsed_seconds':time.perf_counter()-start};history.append(rec);dump(out/'history.json',history)
                atomic_save({'model':m.state_dict(),'optimizer':opt.state_dict(),'scheduler':sched.state_dict() if sched else None,'rng':rng(),'sampler_cursor':step,'sequence_sha256':sha(R/next(q['path'] for q in plan['sequence_files'] if q['seed']==selected['seed'])),'epoch':epoch,'best':best,'patience_used':bad,'history':history,'protocol_sha256':sha(P/'protocol.json'),'source_sha256':sha(Path(__file__))},out/'last.pt')
                print(selected['id'],'epoch',epoch,'step',step,'val_degraded_F1',val['degraded']['F1'],'best',best['degraded_flood_f1'],'patience',bad,flush=True)
                if bad>=10:break
            dump(out/'completion.json',{'status':'SEN1_FORMAL_TRAIN_COMPLETE','task':selected,'epochs':epoch,'updates':step,'best':best,'best_sha256':sha(out/'best.pt'),'last_sha256':sha(out/'last.pt'),'input_chain_sha256':input_chain.hexdigest(),'unique_train_windows':len(exposure),'sample_exposures':step*16,'BN_route_pass':True,'elapsed_seconds':time.perf_counter()-start,'test_access':False,'stop_reason':'patience10' if bad>=10 else 'max_epochs'})
        except Exception as e:dump(out/'failure.json',{'type':type(e).__name__,'message':str(e),'test_access':False,'no_automatic_retry':True});raise
        return
if __name__=='__main__':main()
