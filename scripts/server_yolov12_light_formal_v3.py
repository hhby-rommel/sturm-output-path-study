# Copyright (C) 2026 STURM output-path study contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
# Source-release notice added 2026-10-11; numerical code unchanged.
"""Isolated server YOLOv12 semantic train/val runner; no test phase.

Health uses nonformal seed 10130 and two complete epochs. Formal seed 2026
requires a successful health report with identical runtime/data/config hashes.
Fresh directories only, except explicitly requested epoch-boundary resume.
"""
import os
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
import argparse, copy, hashlib, importlib.util, itertools, json, random, sys, time, subprocess
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
PROJECT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'external/yolov12-official-local-v1'
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(SOURCE))
import numpy as np
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler
from src.dataset.sturm_flood import SturmFloodDataset, grouped_sensor_collate
from src.evaluation.formal import seed_everything, evaluate_clean_and_degraded, is_better_checkpoint
from src.training.engine import train_one_epoch
from src.training.masked_segmentation import MaskedSemanticLoss

CONFIG = dict(model='YOLOv12n-turbo layers0..14 + fresh P3 light semantic head + bilinear resize',
    common_channels=16, input_channels=dict(S1=2,S2=9), native_size=128,
    batch_size=16, epochs=50, patience=10, optimizer='AdamW', lr=3e-5,
    weight_decay=1e-4, norm_bias_decay=True, scheduler='none', clip=5,
    bn_running_stats='frozen affine trainable', precision='FP32',
    train_variant='m1_degraded', normalization='none_v1',
    sampler='replacement, num_samples=len(train), weight=0.5/sensor_count',
    selection='pixel_pooled_v1: degraded F1, clean F1, earlier epoch', val_tag=1,
    pretraining='COCO detection', attention='official manual non-Flash fallback',
    loss='main clean+degraded CE+Dice; mean present sensors; one global step',
    workers=0, sampler_seed_offset=1, loader_rng='dedicated +1001/+1002, explicit restoration', health_epochs=2, health_updates=1822, test_phase='absent; separately authorized future runner required', comparison_scope='system comparison: light-head lr=3e-5 versus historical decoder lr=3e-4; not pure head-only causality')

def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,v):
    tmp=p.with_suffix(p.suffix+'.tmp')
    tmp.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    os.replace(tmp,p)
def save(p,v):
    tmp=p.with_suffix(p.suffix+'.tmp');torch.save(v,tmp);os.replace(tmp,p)
def cpu(v):
    if isinstance(v,torch.Tensor):return v.detach().cpu().clone()
    if isinstance(v,dict):return {k:cpu(x) for k,x in v.items()}
    if isinstance(v,list):return [cpu(x) for x in v]
    if isinstance(v,tuple):return tuple(cpu(x) for x in v)
    return copy.deepcopy(v)
def exact(a,b):
    if isinstance(a,torch.Tensor):return torch.equal(a.cpu(),b.cpu())
    if isinstance(a,dict):return a.keys()==b.keys() and all(exact(a[k],b[k]) for k in a)
    if isinstance(a,(list,tuple)):return len(a)==len(b) and all(exact(x,y) for x,y in zip(a,b))
    return a==b
def rng():return dict(python=random.getstate(),numpy=np.random.get_state(),cpu=torch.get_rng_state(),cuda=torch.cuda.get_rng_state_all())
def restore_rng(r):
    random.setstate(r['python']);np.random.set_state(r['numpy']);torch.set_rng_state(r['cpu']);torch.cuda.set_rng_state_all(r['cuda'])
def runtime():
    names=['src/__init__.py','src/dataset/__init__.py','src/training/__init__.py','src/evaluation/__init__.py','scripts/server_yolov12_light_formal_v1.py','scripts/server_yolov12_light_formal_v2.py','scripts/server_yolov12_light_formal_v3.py','src/models/yolov12_comparator.py','src/models/yolov12_head_control_pretest.py',
        'src/training/engine.py','src/training/masked_segmentation.py','src/training/degradation.py',
        'src/training/dcl_losses.py','src/evaluation/formal.py','src/dataset/sturm_flood.py',
        'src/dataset/quality.py','configs/yolov12_official_local_source_lock_v1.json',
        'configs/yolov12n_turbo_asset_lock_v1.json','data/manifests/manifest.csv','data/splits/event_split_v1.csv']
    return {n:sha(ROOT/n) for n in names}
def assets():
    lock=json.loads((ROOT/'configs/yolov12_official_local_source_lock_v1.json').read_text())
    for rel,digest in lock['files'].items():
        if sha(SOURCE/rel)!=digest:raise RuntimeError('official source changed: '+rel)
    weight=json.loads((ROOT/'configs/yolov12n_turbo_asset_lock_v1.json').read_text())
    p=ROOT/weight['path'].replace('\\','/')
    if sha(p)!=weight['sha256']:raise RuntimeError('weight hash mismatch')
    return p
def build(weight):
    import ultralytics
    from ultralytics.nn.modules import block
    if not Path(ultralytics.__file__).resolve().is_relative_to(SOURCE.resolve()) or block.USE_FLASH_ATTN:
        raise RuntimeError('official isolated manual-attention backend required')
    spec=importlib.util.spec_from_file_location('local_yolov12_adapter',ROOT/'src/models/yolov12_comparator.py')
    mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
    m=mod.YoloV12SemanticComparator(SOURCE,weight);report=dict(m.load_report)
    m.release_reference()
    spec=importlib.util.spec_from_file_location('light_head_adapter',ROOT/'src/models/yolov12_head_control_pretest.py')
    head=importlib.util.module_from_spec(spec);spec.loader.exec_module(head)
    before={k:v.clone() for k,v in m.state_dict().items() if k.startswith(('core.','projections.','feature_adapters.'))}
    # Preserve all shared initialization; the fresh head has a registered seed.
    m=head.YoloV12LightHead(m,torch.initial_seed()+77)
    if not all(torch.equal(v,m.state_dict()[k]) for k,v in before.items()):raise RuntimeError('light-head common initialization differs')
    report.update(common_initial_keys=len(before),common_initialization_exact=True,head_seed=torch.initial_seed()+77,
                  identity='retained official trunk plus adapted semantic head, not native detector')
    return m,report
class Progress:
    def __init__(self,loader,label,out,epoch,trace=False):self.loader=loader;self.label=label;self.out=out;self.epoch=epoch;self.trace=trace
    def __iter__(self):
        start=time.monotonic();h=hashlib.sha256()
        for i,b in enumerate(self.loader,1):
            h.update(json.dumps({s:x['tile_id'] for s,x in b.items()},sort_keys=True).encode())
            if i==1 or i%50==0 or i==len(self.loader):
                elapsed=time.monotonic()-start
                print(f'[{self.label}] epoch={self.epoch} batch={i}/{len(self.loader)} elapsed={elapsed:.1f}s',flush=True)
                write(self.out/'progress.json',dict(phase=self.label,epoch=self.epoch,batch=i,total=len(self.loader),elapsed_seconds=elapsed))
            if time.monotonic()-start>1800:raise TimeoutError('phase exceeds 1800 seconds')
            yield b
        if self.trace:write(self.out/f'epoch_{self.epoch:03d}_sampling.json',dict(batches=i,ordered_tile_ids_sha256=h.hexdigest()))

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--phase',choices=('formal',),required=True)
    p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--health-dir',type=Path)
    p.add_argument('--resume',action='store_true')
    p.add_argument('--seed',type=int,choices=(2028,),required=True)
    a=p.parse_args();out=a.output_dir.resolve()
    if os.name!='posix' or not torch.cuda.is_available():raise RuntimeError('server Linux CUDA only')
    if ROOT.parent != PROJECT/'.codex_deployments' or ROOT.name != 'yolov12_light_server_20261001_v1':raise RuntimeError('wrong isolated deployment')
    busy=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
    if busy:raise RuntimeError('GPU compute job already active: '+busy)
    expected=PROJECT/('experiments/yolov12_light_server_health_20261001_v1' if a.phase=='health' else f'results/formal/yolov12n_turbo_light_semantic_v1/{a.seed}')
    if out!=expected.resolve():raise RuntimeError('unregistered output directory')
    if a.resume and a.phase!='formal':raise ValueError('resume applies only to formal epoch boundary')
    if a.phase=='formal' and not a.health_dir:raise ValueError('--health-dir is mandatory')
    locked=runtime();weight=assets()
    if a.phase=='formal':
        gate=json.loads((a.health_dir/'summary.json').read_text())
        if locked['scripts/server_yolov12_light_formal_v2.py']!='8439f646515fd7d411bf92eb7419abd0056659eedc258edf84ef5ae4edf1a6c6':raise RuntimeError('frozen seed2027 runner changed')
        reference={k:v for k,v in locked.items() if k not in ('scripts/server_yolov12_light_formal_v2.py','scripts/server_yolov12_light_formal_v3.py')}
        if reference['scripts/server_yolov12_light_formal_v1.py']!='7315eb79f5fecee8a2ed6a42bac1860309632887e926504149b865b0e97b9846':raise RuntimeError('original frozen gate runner changed')
        if gate['status']!='SERVER_FULL_HEALTH_PASSED' or gate['runtime_hashes']!=reference or gate['config']!=CONFIG:
            raise RuntimeError('successful matching full health report required')
        if gate['updates']!=1822 or not gate['full_val_replay_exact'] or not all(gate['epoch_boundary_resume'].values()):raise RuntimeError('incomplete health/restore gate')
        if gate['environment']['gpu']!=torch.cuda.get_device_name(0) or gate['environment']['torch']!=torch.__version__:
            raise RuntimeError('health environment mismatch')
    if not a.resume:out.mkdir(parents=True,exist_ok=False)
    elif not (out/'last.pt').exists():raise RuntimeError('no resumable epoch boundary')
    mutex=PROJECT/'experiments/yolov12_light_gpu.lock'
    handle=os.open(mutex,os.O_CREAT|os.O_EXCL|os.O_WRONLY)
    os.write(handle,str(os.getpid()).encode());os.close(handle)
    try:
        torch.set_num_threads(2);torch.set_num_interop_threads(2);torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        seed=10130 if a.phase=='health' else a.seed
        seed_everything(seed)
        datasets={s:SturmFloodDataset(ROOT/'data/manifests/manifest.csv',ROOT/'data/splits/event_split_v1.csv',s) for s in ('train','val')}
        assert len(datasets['train'])==14561 and len(datasets['val'])==4853
        assert not ({r['event_id'] for r in datasets['train'].rows}&{r['event_id'] for r in datasets['val'].rows})
        counts={s:sum(r['sensor']==s for r in datasets['train'].rows) for s in ('S1','S2')}
        sampler=WeightedRandomSampler(torch.tensor([.5/counts[r['sensor']] for r in datasets['train'].rows],dtype=torch.double),len(datasets['train']),replacement=True,generator=torch.Generator().manual_seed(seed+1))
        loader=DataLoader(datasets['train'],batch_size=16,sampler=sampler,num_workers=0,collate_fn=grouped_sensor_collate,generator=torch.Generator().manual_seed(seed+1001))
        val=DataLoader(datasets['val'],batch_size=16,num_workers=0,collate_fn=grouped_sensor_collate,generator=torch.Generator().manual_seed(seed+1002))
        m,init=build(weight);device=torch.device('cuda');m=m.to(device)
        assert sum(x.numel() for x in m.parameters())==1504834
        opt=torch.optim.AdamW(m.parameters(),lr=3e-5,weight_decay=1e-4)
        criterion=MaskedSemanticLoss();best=None;history=[];start_epoch=1;bad=0
        seed_everything(seed)
        from importlib.metadata import version
        import platform
        env=dict(torch=torch.__version__,cuda=torch.version.cuda,gpu=torch.cuda.get_device_name(0),python=platform.python_version(),packages={n:version(n) for n in ('numpy','torchvision','rasterio','scipy')})
        if env!=gate['environment']:raise RuntimeError('full health environment changed')
        provenance=dict(runner_version='seed2028_extension_v3',health_reference_runtime_hashes=reference,status='SERVER_FULL_HEALTH' if a.phase=='health' else 'FORMAL_TRAINING',config=CONFIG,
            seed=seed,parameters=1504834,train_tiles=14561,val_tiles=4853,train_sensor_counts=counts,
            runtime_hashes=locked,environment=env,initialization=init,health_dir=str(a.health_dir) if a.health_dir else None,
            test_images_accessed=False,official_source_commit=json.loads((ROOT/'configs/yolov12_official_local_source_lock_v1.json').read_text())['commit'])
        def boundary(epoch):return dict(model=cpu(m.state_dict()),optimizer=cpu(opt.state_dict()),rng=rng(),
            sampler=sampler.generator.get_state(),loader=loader.generator.get_state(),val_loader=val.generator.get_state(),
            epoch=epoch,best=best,bad=bad,history=history,runtime_hashes=locked,config=CONFIG,seed=seed)
        def restore(state):
            m.load_state_dict(state['model'],strict=True);opt.load_state_dict(state['optimizer'])
            sampler.generator.set_state(state['sampler']);loader.generator.set_state(state['loader']);val.generator.set_state(state['val_loader']);restore_rng(state['rng'])
        if a.resume:
            state=torch.load(out/'last.pt',map_location='cpu',weights_only=False)
            if state['runtime_hashes']!=locked or state['config']!=CONFIG or state['seed']!=seed:raise RuntimeError('resume provenance mismatch')
            restore(state);best=state['best'];bad=state['bad'];history=state['history'];start_epoch=state['epoch']+1
        else:
            write(out/'metadata.json',provenance)
            for rel in locked:
                snapshot=out/'implementation_snapshot'/rel
                snapshot.parent.mkdir(parents=True,exist_ok=True)
                snapshot.write_bytes((ROOT/rel).read_bytes())
        print(f'START phase={a.phase} seed={seed} GPU={env["gpu"]} train=14561 val=4853 batch=16 parameters=1504834',flush=True)
        torch.cuda.reset_peak_memory_stats();started=time.monotonic()
        if a.phase=='health':
            baseline=boundary(0);probe={}
            for sensor in ('S1','S2'):
                indices=[i for i,r in enumerate(datasets['train'].rows) if r['sensor']==sensor][:16]
                batch=grouped_sensor_collate([datasets['train'][i] for i in indices])
                t0=time.monotonic()
                probe[sensor]=train_one_epoch(m,[batch],criterion,opt,device=device,grad_clip_norm=5,freeze_batchnorm=True,model_variant='m1_degraded')
                torch.cuda.synchronize();probe[sensor]['seconds']=time.monotonic()-t0
                probe[sensor]['peak_reserved_mib']=torch.cuda.max_memory_reserved()/2**20
                restore(baseline)
            write(out/'batch16_hardware_probe.json',probe)
            print('HARDWARE homogeneous batch16 S1/S2 passed '+json.dumps({s:x['peak_reserved_mib'] for s,x in probe.items()}),flush=True)
        limit=2 if a.phase=='health' else 50
        for epoch in range(start_epoch,limit+1):
            if runtime()!=locked:raise RuntimeError('runtime/data changed during run')
            t0=time.monotonic()
            stats=train_one_epoch(m,Progress(loader,'train',out,epoch,True),criterion,opt,device=device,grad_clip_norm=5,freeze_batchnorm=True,model_variant='m1_degraded')
            assert stats['steps']==len(loader)==911
            train_seconds=time.monotonic()-t0
            metrics=evaluate_clean_and_degraded(m,Progress(val,'validation',out,epoch),criterion,device,seed=seed,tag=1)
            candidate=dict(epoch=epoch,degraded_flood_f1=metrics['degraded']['overall']['flood_f1'],clean_flood_f1=metrics['clean']['overall']['flood_f1'])
            improved=is_better_checkpoint(candidate,best)
            if improved:best=candidate;bad=0;save(out/'best.pt',dict(model=cpu(m.state_dict()),selection=best,runtime_hashes=locked,config=CONFIG,seed=seed))
            else:bad+=1
            row=dict(epoch=epoch,train=stats,validation=metrics,selection=candidate,is_best=improved,train_seconds=train_seconds,epoch_seconds=time.monotonic()-t0)
            history.append(row);write(out/'history.json',history);save(out/'last.pt',boundary(epoch))
            print(f'EPOCH {epoch:02d}/{limit} | loss {stats["loss"]:.4f} | F1 clean {candidate["clean_flood_f1"]:.4f} degraded {candidate["degraded_flood_f1"]:.4f} | S1 {metrics["degraded"]["per_sensor"]["S1"]["flood_f1"]:.4f} S2 {metrics["degraded"]["per_sensor"]["S2"]["flood_f1"]:.4f} | {row["epoch_seconds"]:.1f}s | best={best["epoch"]} patience={bad}/10',flush=True)
            if bad>=10:break
        resume_report=None
        if a.phase=='health':
            state=torch.load(out/'last.pt',map_location='cpu',weights_only=False)
            # Regenerate the next epoch's actual sampler batches from an epoch boundary.
            def branch():
                traces=[]
                def first_two():
                    for b in itertools.islice(iter(loader),2):
                        traces.append({s:x['tile_id'] for s,x in b.items()});yield b
                t=train_one_epoch(m,first_two(),criterion,opt,device=device,grad_clip_norm=5,freeze_batchnorm=True,model_variant='m1_degraded')
                return dict(model=cpu(m.state_dict()),optimizer=cpu(opt.state_dict()),stats=t,trace=traces)
            one=branch()
            # Fresh model and optimizer, then restore CPU-staged persisted checkpoint.
            del m,opt
            m,_=build(weight);m=m.to(device);opt=torch.optim.AdamW(m.parameters(),lr=3e-5,weight_decay=1e-4)
            restore(state);two=branch()
            resume_report={k:exact(one[k],two[k]) for k in one}
            if not all(resume_report.values()):raise RuntimeError('epoch-boundary restoration differs')
            restore(state)
        selected=torch.load(out/'best.pt',map_location='cpu',weights_only=False)
        m.load_state_dict(selected['model'],strict=True)
        replay=evaluate_clean_and_degraded(m,Progress(val,'selected_replay',out,best['epoch']),criterion,device,seed=seed,tag=1)
        if not exact(replay,history[best['epoch']-1]['validation']):raise RuntimeError('full selected validation differs')
        write(out/'selected_full_val.json',replay)
        summary=dict(status='SERVER_FULL_HEALTH_PASSED' if a.phase=='health' else 'FORMAL_TRAINING_COMPLETED_TEST_PENDING',
            config=CONFIG,runtime_hashes=locked,environment=env,seed=seed,best=best,epochs=len(history),updates=sum(r['train']['steps'] for r in history),
            full_val_replay_exact=True,epoch_boundary_resume=resume_report,test_images_accessed=False,
            peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20,peak_reserved_mib=torch.cuda.max_memory_reserved()/2**20,
            elapsed_seconds=time.monotonic()-started,first_epoch_seconds=history[0]['epoch_seconds'])
        if a.phase=='health' and any(r['selection']['degraded_flood_f1']<=0 for r in history):raise RuntimeError('no positive-class learning under full health')
        write(out/'summary.json',summary);print(json.dumps(summary,ensure_ascii=False),flush=True)
    except BaseException as error:
        write(out/'failure.json',dict(type=type(error).__name__,message=str(error),test_images_accessed=False));raise
    finally:mutex.unlink()

if __name__=='__main__':main()
