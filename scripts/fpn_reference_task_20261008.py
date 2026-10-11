# Copyright (C) 2026 STURM output-path study contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
# Source-release notice added 2026-10-11; numerical code unchanged.
"""Standalone FPN train/val entry. No test branch or implicit launch."""
from pathlib import Path
import argparse,hashlib,importlib.util,json,shutil,sys,time
R=Path(__file__).resolve().parents[1]
W=R/'prepared_controls/decoder_reproduction_validation_20261004_v1/workspace_public_v3'
sys.path.insert(0,str(W));sys.path.append(str(R))
import torch
from src.models.yolo26_semantic import load_matching_yolo26_weights
from src.dataset.sturm_flood import SturmFloodDataset
from src.training.engine import train_one_epoch,_freeze_batchnorm_running_stats
from src.training.masked_segmentation import MaskedSemanticLoss
from src.evaluation.formal import seed_everything,evaluate_clean_and_degraded,is_better_checkpoint
from scripts.server_main import make_loader,save_checkpoint

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,v):
    tmp=p.with_suffix(p.suffix+'.tmp');assert not tmp.exists();tmp.write_text(json.dumps(v,indent=2,allow_nan=False)+'\n');tmp.replace(p)

def build(seed,device):
    spec=importlib.util.spec_from_file_location('src.models.yolo26_fpn_reference',R/'src/models/yolo26_fpn_reference.py')
    mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
    seed_everything(seed);m=mod.Yolo26FPNReference(common_channels=16)
    load=load_matching_yolo26_weights(m.core,W/'models/pretrained/yolo26n-sem.pt')
    assert load['matched_keys']==359 and load['skipped_shape_keys']==5
    assert sum(p.numel() for p in m.parameters())==1713714
    return m.to(device),load

def loaders(seed):
    manifest=R/'data/manifests/manifest.csv';split=R/'data/splits/event_split_v1.csv'
    train=SturmFloodDataset(manifest,split,'train');val=SturmFloodDataset(manifest,split,'val')
    assert len(train)==14561 and len(val)==4853
    return make_loader(train,batch_size=16,train=True,seed=seed+1),make_loader(val,batch_size=16,train=False,seed=seed+2)

def main():
    p=argparse.ArgumentParser();p.add_argument('--seed',type=int,choices=[2026,2027,2028],required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--mode',choices=['audit','train'],required=True)
    p.add_argument('--execute-authorized',action='store_true');a=p.parse_args()
    if not a.execute_authorized:p.error('Explicit authorization and execution flag required')
    P=R/'prepared_controls/fpn_reference_20261008_v1';lock=json.loads((P/'execution_lock.json').read_text())
    for q in lock['files']:assert sha(R/q['path'])==q['sha256'],q['path']
    plan=json.loads((P/'protocol.json').read_text())
    if a.mode=='train' and not plan['training_authorized']:p.error('Protocol remains prepared; formal training is not authorized')
    assert torch.cuda.is_available() and torch.cuda.get_device_name()== 'NVIDIA GeForce RTX 4090 D','Formal backend GPU differs; no fallback'
    torch.set_num_threads(4)
    # Original STURM backend, distinct from deterministic Sen1 control.
    torch.backends.cudnn.allow_tf32=True;torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=False;torch.use_deterministic_algorithms(False)
    assert torch.__version__=='2.1.0+cu121','Formal torch version differs'
    out=a.output.resolve();assert out.is_relative_to((R/'results').resolve())
    assert shutil.disk_usage(R).free >= 2*1024**3+100*1024**2,'Insufficient capacity; no asset deletion'
    out.mkdir(parents=True,exist_ok=False)
    try:
        m,report=build(a.seed,torch.device('cuda'));opt=torch.optim.AdamW(m.parameters(),lr=.0003,weight_decay=.0001)
        train,val=loaders(a.seed);criterion=MaskedSemanticLoss(aux_weight=.4).cuda()
        identity=dict(seed=a.seed,protocol_sha256=sha(P/'protocol.json'),source_sha256=sha(Path(__file__)),
                      lock_sha256=sha(P/'execution_lock.json'),pretrained_load=report,training_test_access=False)
        write(out/'scope.json',identity)
        if a.mode=='audit':
            m.train();_freeze_batchnorm_running_stats(m)
            assert all(not q.training for q in m.modules() if isinstance(q,torch.nn.modules.batchnorm._BatchNorm))
            write(out/'completion.json',dict(status='FPN_NONTRAINING_SERVER_AUDIT_PASS',identity=identity,test_access=False));return
        bn={k:v.cpu().clone() for k,v in m.state_dict().items() if 'running_' in k or 'num_batches_tracked' in k}
        history=[];best=None;stale=0;start=time.monotonic()
        for epoch in range(1,51):
            stats=train_one_epoch(m,train,criterion,opt,device=torch.device('cuda'),grad_clip_norm=5,
                model_variant='m1_degraded',freeze_batchnorm=True,track_parameter_update=True)
            assert stats['steps']==911
            ev=evaluate_clean_and_degraded(m,val,criterion,torch.device('cuda'),seed=a.seed,tag=1)
            candidate=dict(epoch=epoch,clean_flood_f1=ev['clean']['overall']['flood_f1'],degraded_flood_f1=ev['degraded']['overall']['flood_f1'])
            improved=is_better_checkpoint(candidate,best)
            if improved:best=candidate;stale=0
            else:stale+=1
            assert all(torch.equal(v,m.state_dict()[k].cpu()) for k,v in bn.items())
            history.append(dict(epoch=epoch,training=stats,validation=ev,selection=best,stale=stale,elapsed_seconds=time.monotonic()-start))
            write(out/'history.json',history)
            for name in (['best.pt','last.pt'] if improved else ['last.pt']):
                assert shutil.disk_usage(R).free >= 2*1024**3+30*1024**2
                assert not (out/name).with_suffix('.pt.tmp').exists(),'Stale temporary; stop without retry'
                save_checkpoint(out/name,m,opt,None,epoch=epoch,seed=a.seed,selection=best,stale_epochs=stale,
                    history=history,max_epochs=50,patience=10,optimizer_spec_version='optimizer_v1',train_loader=train)
            write(out/'status.json',dict(epoch=epoch,selection=best,stale=stale,test_access=False))
            print('FPN_TRAIN',a.seed,epoch,best,stale,flush=True)
            if stale>=10:break
        write(out/'completion.json',dict(status='FPN_TRAIN_VAL_COMPLETE_NO_TEST',identity=identity,best=best,
            best_sha256=sha(out/'best.pt'),last_sha256=sha(out/'last.pt'),epochs=epoch,stop_reason='patience10' if stale>=10 else 'max50',test_access=False))
    except Exception as e:
        write(out/'failure.json',dict(type=type(e).__name__,message=str(e),automatic_retry=False,test_access=False));raise

if __name__=='__main__':main()
