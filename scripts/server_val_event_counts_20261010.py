# Copyright (C) 2026 STURM output-path study contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
# Source-release notice added 2026-10-11; numerical code unchanged.
"""Authorized frozen val event-count supplementation only; no test or training."""
from pathlib import Path
import sys,json,csv,hashlib,importlib.util,time,gc,statistics,traceback,os
B=Path(__file__).resolve().parents[1]
W=B/'experiments/followup_server_20261008_v1/workspace'
S=B/'experiments/same_output_efficiency_20261004_v1'
O=B/'results/val_event_counts_20261010_v1'
sys.path.insert(0,str(W));sys.path.append(str(B))
def dump(p,v):
 t=p.with_suffix('.tmp');t.write_text(json.dumps(v,indent=2,allow_nan=False)+'\n');t.replace(p)
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for block in iter(lambda:f.read(1048576),b''):h.update(block)
 return h.hexdigest()
def load(p,name):
 spec=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
def main():
 O.mkdir(exist_ok=False)
 dump(O/'authorization.json',dict(human_authorized=True,source='User 2026-10-10: server opened, continue inference then revise; scoped to frozen val event counts',split='val',training=False,test_access=False,checkpoint_reselection=False,evaluation_seed=12026))
 import torch
 import torch.nn.functional as F
 from src.dataset.sturm_flood import SturmFloodDataset,grouped_sensor_collate
 from src.training.masked_segmentation import MaskedSegmentationMeter
 assert torch.__version__=='2.1.0+cu121' and torch.cuda.get_device_name()=='NVIDIA GeForce RTX 4090 D'
 torch.set_num_threads(4);torch.backends.cudnn.allow_tf32=True;torch.backends.cuda.matmul.allow_tf32=False
 torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=False
 dump(O/'environment.json',dict(torch=torch.__version__,GPU=torch.cuda.get_device_name(),cudnn_tf32=True,matmul_tf32=False,benchmark=False,deterministic=False,threads=4,script_sha256=sha(Path(__file__))))
 regs=json.loads((S/'registrations_90.json').read_text())['units']
 units=list(csv.DictReader((S/'units_90.csv').open()))
 adapter=load(S/'server_efficiency_same_output_adapter_v1.py','old_cost_adapter')
 fpnmod=load(W/'src/models/yolo26_fpn_reference.py','src.models.yolo26_fpn_reference')
 records=[]
 for seed in [2026,2027,2028]:
  for system in ['main_only','decoder','fpn']:
   if system=='fpn':
    cp=W/f'results/fpn_reference_formal_20261008_v1/seed{seed}/best.pt'
    done=json.loads((cp.parent/'completion.json').read_text());expected=done['best_sha256'];reg=None;u=None
   else:
    u=next(q for q in units if q['system']==system and int(q['seed'])==seed)
    cp=Path(u['remote_checkpoint']);expected=u['expected_sha256'];reg=regs[u['unit_id']]
   assert sha(cp)==expected,str(cp)
   records.append(dict(system=system,seed=seed,checkpoint=str(cp),checkpoint_sha256=expected,unit=u,registration=reg))
 dump(O/'frozen_assets.json',records)
 def build(q,device):
  if q['system']=='fpn':m=fpnmod.Yolo26FPNReference(common_channels=16)
  else:m=adapter.construct(q['unit'],q['registration']['constructor_config'])
  ck=torch.load(q['checkpoint'],map_location='cpu',weights_only=False)
  key=q['registration']['checkpoint_state_key'] if q['registration'] else 'model_state_dict'
  m.load_state_dict(ck[key],strict=True);del ck;return m.eval().to(device)
 models={f"{q['system']}_{q['seed']}":build(q,'cuda') for q in records}
 meter=lambda:MaskedSegmentationMeter(2,255)
 meters={k:{v:{'overall':meter(),'per_sensor':{},'per_event':{}} for v in ['clean','degraded']} for k in models}
 ds=SturmFloodDataset(W/'data/manifests/manifest.csv',W/'data/splits/event_split_v1.csv','val')
 assert len(ds)==4853
 ds.rows.sort(key=lambda r:(r['sensor'],r['event_id'],r['tile_id'],r['image_path']))
 dump(O/'evaluation_protocol.json',dict(global_evaluation_seed=12026,identifier='sensor/event_id/tile_id/image_path',sample_seed_rule='little-endian first8 SHA256 bytes of 12026|identifier|stream, modulo 2**63-1',streams=['severity','gaussian','deletion'],RNG='torch CPU Generator; float32',batch=16,order='sensor,event_id,tile_id,image_path',degradation_coefficients_unchanged=True,shared_transformed_tensors=True,loss_not_recomputed=True))
 def generator(identifier,stream):
  seed=int.from_bytes(hashlib.sha256(f'12026|{identifier}|{stream}'.encode()).digest()[:8],'little')%(2**63-1)
  return torch.Generator(device='cpu').manual_seed(seed)
 loader=torch.utils.data.DataLoader(ds,batch_size=16,shuffle=False,num_workers=0,collate_fn=grouped_sensor_collate)
 manifest=(O/'sample_perturbation_receipts.jsonl').open('x');start=time.time()
 streamhash=hashlib.sha256()
 with torch.inference_mode():
  for bi,group in enumerate(loader):
   for sensor,batch in group.items():
    x=batch['image'];valid=batch['valid_mask'].bool();images=[];valids=[]
    for i in range(len(x)):
     # Resolve the identity from immutable manifest rows in the fixed loader order.
     row=ds.rows[bi*16+i]
     if row['sensor']!=sensor:
      row=next(r for r in ds.rows[bi*16:bi*16+16] if r['sensor']==sensor and r['tile_id']==batch['tile_id'][i])
     identifier='/'.join(row[k] for k in ['sensor','event_id','tile_id','image_path'])
     severity=.5+.5*torch.rand((),generator=generator(identifier,'severity'))
     noise=torch.randn(x[i].shape,generator=generator(identifier,'gaussian'))
     if sensor=='S1':degraded=x[i]*(1+.12*severity*noise)
     else:degraded=x[i]+.20*severity*x[i].flatten(1).std(1,unbiased=False).clamp_min(.001)[:,None,None]*noise
     deletion=(torch.rand(valid[i].shape,generator=generator(identifier,'deletion'))<(.05+.10*severity))&valid[i]
     keep=valid[i]&~deletion;degraded=torch.where(keep[None],degraded,torch.zeros_like(degraded))
     h=hashlib.sha256(degraded.contiguous().numpy().tobytes()+keep.numpy().tobytes()).hexdigest();streamhash.update(identifier.encode()+h.encode())
     manifest.write(json.dumps(dict(identifier=identifier,severity=float(severity),noise_sha256=hashlib.sha256(noise.numpy().tobytes()).hexdigest(),deletion_sha256=hashlib.sha256(deletion.numpy().tobytes()).hexdigest(),input_valid_sha256=h))+'\n')
     images.append(degraded);valids.append(keep)
    for view,xx,vv in [('clean',x,valid),('degraded',torch.stack(images),torch.stack(valids))]:
     xx=xx.cuda();vv=vv.cuda();target=batch['mask'].cuda()
     for key,m in models.items():
      logits=m.forward_native(xx,sensor)
      if logits.shape[-2:]!=(128,128):logits=F.interpolate(logits,size=(128,128),mode='bilinear',align_corners=False)
      assert torch.isfinite(logits).all()
      d=meters[key][view];d['overall'].update(logits,target,vv)
      d['per_sensor'].setdefault(sensor,meter()).update(logits,target,vv)
      for i,event in enumerate(batch['event_id']):d['per_event'].setdefault(event,meter()).update(logits[i:i+1],target[i:i+1],vv[i:i+1])
    if bi%10==0:
     dump(O/'progress.json',dict(stage='frozen_val_event_counts',batches_done=bi+1,batches_total=len(loader),elapsed_seconds=time.time()-start));print('EVAL',bi+1,len(loader),flush=True)
 manifest.close()
 for key,views in meters.items():
  result={v:{k:(z.compute() if k=='overall' else {n:t.compute() for n,t in z.items()}) for k,z in d.items()} for v,d in views.items()}
  dump(O/f'{key}_reevaluation.json',dict(metrics=result,shared_input_stream_sha256=streamhash.hexdigest(),evaluation_seed=12026))
 # Gate against the original spatial-val sensor totals before accepting new event counts.
 old=json.loads((B/'results/spatial_val_20261010_v1/summary.json').read_text())
 for key,views in meters.items():
  for view,d in views.items():
   for sensor,mm in d['per_sensor'].items():
    cm=mm.compute()['confusion_matrix'];actual=dict(TP=cm[1][1],FP=cm[0][1],FN=cm[1][0]);expected=old[f'{key}/{view}/{sensor}']
    assert actual=={k:expected[k] for k in actual},(key,view,sensor,actual,expected)
 for q in records:assert sha(Path(q['checkpoint']))==q['checkpoint_sha256']
 dump(O/'reevaluation_complete.json',dict(status='COMPLETE',elapsed_seconds=time.time()-start,input_stream_sha256=streamhash.hexdigest(),models=9,tiles=4853,split='val',old_spatial_counts_equal=True,training=False,test_access=False,checkpoint_reselection=False))
if __name__=='__main__':
 try:main()
 except Exception:
  if O.exists():dump(O/'failure.json',dict(status='FAILED',traceback=traceback.format_exc()))
  raise
