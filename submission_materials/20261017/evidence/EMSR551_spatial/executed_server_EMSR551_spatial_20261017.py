"""Bounded fixed12026 EMSR551 cache + spatial counts. Explicit launch only."""
from pathlib import Path
import argparse, csv, hashlib, importlib.util, json, sys, time, traceback, os, shutil
from collections import defaultdict

B=Path('/root/autodl-tmp/sturm_codex_20260919')
W=B/'experiments/followup_server_20261008_v1/workspace'
C=B/'experiments/EMSR551_spatial_20261017_v1'
O=B/'results/EMSR551_spatial_20261017_v1'
sys.path.insert(0,str(C));sys.path.insert(0,str(W));sys.path.append(str(B))
def read(p): return json.loads(p.read_text())
def dump(p,v): p.write_text(json.dumps(v,indent=2,allow_nan=False)+'\n')
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for block in iter(lambda:f.read(1048576),b''): h.update(block)
    return h.hexdigest()
def load(p,name):
    spec=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
def csvwrite(p,rr):
    with p.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rr[0]));w.writeheader();w.writerows(rr)

def main():
    import numpy as np
    import torch
    import torch.nn.functional as F
    from src.dataset.sturm_flood import SturmFloodDataset,grouped_sensor_collate
    from spatial_error_partition_20261017 import partition_counts
    assert not O.exists(), 'Existing attempt: do not repeat or overwrite'
    assert shutil.disk_usage(B).free>=1024**3
    active=[]
    for p in Path('/proc').iterdir():
        if not p.name.isdigit() or int(p.name) in (os.getpid(),os.getppid()):continue
        try:cmd=(p/'cmdline').read_bytes().replace(b'\0',b' ').decode()
        except (OSError,UnicodeError):continue
        if str(B) in cmd and 'python' in cmd and '-c ' not in cmd:active.append(dict(pid=int(p.name),command=cmd))
    assert not active,active
    assert torch.__version__=='2.1.0+cu121' and torch.cuda.get_device_name()=='NVIDIA GeForce RTX 4090 D'
    torch.set_num_threads(4);torch.backends.cudnn.allow_tf32=True;torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=False;torch.use_deterministic_algorithms(False)
    protocol=read(C/'EMSR551_protocol_prepared.json')
    assert protocol['event']=='EMSR551' and protocol['evaluation_seed']==12026 and protocol['primary_radius']==2
    records=read(C/'frozen_assets_full.json')
    assert {(q['system'],q['seed']) for q in records}=={(s,n) for s in ('main_only','decoder','fpn') for n in range(2026,2031)}
    with (C/'EMSR551_expected_five_seed_counts.csv').open(encoding='utf-8-sig') as f:
        expected={(q['system'],int(q['seed']),q['view']):q for q in csv.DictReader(f)}
    assert len(expected)==30
    admission=[]
    for q in records:
        assert sha(Path(q['checkpoint']))==q['checkpoint_sha256'],q['checkpoint']
        admission.append(dict(path=q['checkpoint'],sha256=q['checkpoint_sha256'],purpose='New spatial inference checkpoint handoff',conclusion='MATCH'))
    ds=SturmFloodDataset(W/'data/manifests/manifest.csv',W/'data/splits/event_split_v1.csv','test')
    assert len(ds)==4863
    ds.rows.sort(key=lambda r:(r['sensor'],r['event_id'],r['tile_id'],r['image_path']))
    target_indices=[i for i,r in enumerate(ds.rows) if r['event_id']=='EMSR551']
    assert len(target_indices)==9
    with (C/'EMSR551_tile_inventory.csv').open(encoding='utf-8-sig') as f:inventory=list(csv.DictReader(f))
    assert [r['tile_id'] for r in inventory]==[ds.rows[i]['tile_id'] for i in target_indices]
    batch_ids=sorted({i//16 for i in target_indices})
    assert batch_ids==protocol['original_batch_ids']==[192,193]
    O.mkdir();(O/'cache').mkdir()
    dump(O/'test_access_claim.json',dict(status='CLAIMED_ONCE',attempt=1,scope='EMSR551 spatial diagnostic only; two original batches as context',
         models=15,views=2,target_tiles=9,context_batch_slots=32,training=False,checkpoint_reselection=False,time_unix=time.time()))
    dump(O/'protocol.json',protocol);dump(O/'checkpoint_admission.json',admission)
    dump(O/'environment.json',dict(torch=torch.__version__,GPU=torch.cuda.get_device_name(),threads=4,
         cudnn_tf32=True,matmul_tf32=False,benchmark=False,deterministic=False,precision='FP32'))
    adapter=load(B/'experiments/same_output_efficiency_20261004_v1/server_efficiency_same_output_adapter_v1.py','spatial_adapter')
    fpn=load(W/'src/models/yolo26_fpn_reference.py','src.models.yolo26_fpn_reference')
    def generator(identifier,stream):
        seed=int.from_bytes(hashlib.sha256(f'12026|{identifier}|{stream}'.encode()).digest()[:8],'little')%(2**63-1)
        return torch.Generator(device='cpu').manual_seed(seed)
    # Freeze transformed tensors once, shared by all models. Original batch
    # membership/order is retained; only target-event samples are saved/scored.
    batches=[];input_receipts=[]
    for bi in batch_ids:
        ids=list(range(bi*16,min((bi+1)*16,len(ds))))
        samples=[ds[i] for i in ids]
        groups=grouped_sensor_collate(samples)
        assert set(groups)=={'S1'} and len(samples)==16
        batch=groups['S1'];x=batch['image'];valid=batch['valid_mask'].bool()
        degraded=[];keeps=[]
        for j,i in enumerate(ids):
            row=ds.rows[i];identifier='/'.join(row[k] for k in ('sensor','event_id','tile_id','image_path'))
            severity=.5+.5*torch.rand((),generator=generator(identifier,'severity'))
            noise=torch.randn(x[j].shape,generator=generator(identifier,'gaussian'))
            xx=x[j]*(1+.12*severity*noise)
            deletion=(torch.rand(valid[j].shape,generator=generator(identifier,'deletion'))<(.05+.10*severity))&valid[j]
            keep=valid[j]&~deletion;xx=torch.where(keep[None],xx,torch.zeros_like(xx))
            degraded.append(xx);keeps.append(keep)
            if i in target_indices:
                tile=row['tile_id'];target=batch['mask'][j].numpy().astype(np.uint8)
                for suffix,array in [('target',target),('original_valid',valid[j].numpy()),('degraded_valid',keep.numpy())]:
                    np.save(O/'cache'/f'{tile}.{suffix}.npy',array,allow_pickle=False)
                input_receipts.append(dict(tile_id=tile,identifier=identifier,original_batch=bi,
                    degraded_input_valid_sha256=hashlib.sha256(xx.contiguous().numpy().tobytes()+keep.numpy().tobytes()).hexdigest()))
        batches.append((ids,batch,x,valid,torch.stack(degraded),torch.stack(keeps)))
    dump(O/'target_input_receipts.json',input_receipts)
    tile_rows=[];event_rows=[];completed=[];start=time.time()
    for q in records:
        system,seed=q['system'],q['seed'];key=f'{system}_{seed}'
        model=fpn.Yolo26FPNReference(common_channels=16) if system=='fpn' else adapter.construct(q['unit'],q['registration']['constructor_config'])
        checkpoint=torch.load(q['checkpoint'],map_location='cpu',weights_only=False)
        state_key=q['registration']['checkpoint_state_key'] if q['registration'] else 'model_state_dict'
        model.load_state_dict(checkpoint[state_key],strict=True);del checkpoint
        model=model.eval().cuda()
        for view in ('clean','degraded'):
            totals=dict(TP=0,FP=0,FN=0);aggregates=defaultdict(lambda:dict(TP=0,FP=0,FN=0,valid_pixels=0,reference_foreground=0,signed_imbalance=0))
            with torch.inference_mode():
                for ids,batch,x,valid,degraded,keep in batches:
                    xx,vv=(x,valid) if view=='clean' else (degraded,keep)
                    logits=model.forward_native(xx.cuda(),'S1')
                    if logits.shape[-2:]!=(128,128): logits=F.interpolate(logits,size=(128,128),mode='bilinear',align_corners=False)
                    assert torch.isfinite(logits).all()
                    pred=logits.argmax(1).cpu().numpy().astype(np.uint8)
                    for j,i in enumerate(ids):
                        if i not in target_indices: continue
                        tile=ds.rows[i]['tile_id'];target=batch['mask'][j].numpy().astype(np.uint8)
                        np.save(O/'cache'/f'{key}.{view}.{tile}.prediction.npy',pred[j],allow_pickle=False)
                        for radius in (2,1,3):
                            parts,tt=partition_counts(target,pred[j],valid[j].numpy(),vv[j].numpy(),radius)
                            if radius==2:
                                for k in totals: totals[k]+=tt[k]
                            for part in parts:
                                tile_rows.append(dict(system=system,seed=seed,condition=view,tile_id=tile,**part))
                                dst=aggregates[(radius,part['region'])]
                                for k in dst: dst[k]+=part[k]
                    del logits,pred
            baseline=expected[(system,seed,view)]
            assert all(totals[k]==int(baseline[k]) for k in totals),dict(model=key,view=view,observed=totals,expected=baseline)
            for (radius,region),counts in aggregates.items():
                event_rows.append(dict(system=system,seed=seed,condition=view,radius=radius,region=region,**counts,
                    FP_share=counts['FP']/totals['FP'] if totals['FP'] else None,
                    FN_share=counts['FN']/totals['FN'] if totals['FN'] else None))
            completed.append(dict(model=key,condition=view,tiles=9,**totals,baseline_counts_equal=True))
            dump(O/'progress.json',dict(status='RUNNING',completed_model_conditions=len(completed),total_model_conditions=30))
            print('SPATIAL',key,view,len(completed),30,flush=True)
        del model;torch.cuda.empty_cache()
    assert len(completed)==30 and len(tile_rows)==2430 and len(event_rows)==270
    csvwrite(O/'tile_region_counts.csv',tile_rows);csvwrite(O/'event_region_counts.csv',event_rows)
    dump(O/'baseline_count_reconciliation.json',completed)
    dump(O/'completion.json',dict(status='EMSR551_SPATIAL_COMPLETE',models=15,conditions=2,target_tiles=9,
        target_tile_forwards=270,context_preserving_tile_forwards=960,baseline_counts_equal=True,
        training=False,checkpoint_reselection=False,other_event_analysis=False,elapsed_seconds=time.time()-start))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--execute-authorized',action='store_true',required=True);p.parse_args()
    try:main()
    except BaseException as e:
        if O.exists():dump(O/'failure.json',dict(type=type(e).__name__,message=str(e),traceback=traceback.format_exc(),automatic_retry=False))
        raise
