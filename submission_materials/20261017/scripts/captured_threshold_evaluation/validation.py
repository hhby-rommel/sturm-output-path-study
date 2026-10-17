"""Authorized frozen val event-count supplementation only; no test or training."""
from pathlib import Path
import sys,json,csv,hashlib,importlib.util,time,gc,statistics,traceback,os
B=Path('/root/autodl-tmp/sturm_codex_20260919')
W=B/'experiments/followup_server_20261008_v1/workspace'
S=B/'experiments/same_output_efficiency_20261004_v1'
C=Path('/root/autodl-tmp/sturm_codex_20260919/experiments/checkpoint_threshold_validation_20261016_v1')
O=Path('/root/autodl-tmp/sturm_codex_20260919/results/checkpoint_threshold_validation_20261016_v1/val')
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
 dump(O/'authorization.json',dict(human_authorized=True,source='User20261016 checkpoint-only extension; local preparation now, server execution after GPU availability',split='val',training=False,test_access=False,checkpoint_reselection=False,evaluation_seed=12026))
 import torch
 import torch.nn.functional as F
 from src.dataset.sturm_flood import SturmFloodDataset,grouped_sensor_collate
 from src.training.masked_segmentation import MaskedSegmentationMeter
 assert torch.__version__=='2.1.0+cu121' and torch.cuda.get_device_name()=='NVIDIA GeForce RTX 4090 D'
 torch.set_num_threads(4);torch.backends.cudnn.allow_tf32=True;torch.backends.cuda.matmul.allow_tf32=False
 torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=False;torch.use_deterministic_algorithms(False)
 dump(O/'environment.json',dict(torch=torch.__version__,GPU=torch.cuda.get_device_name(),cudnn_tf32=True,matmul_tf32=False,benchmark=False,deterministic=False,threads=4,script_sha256=sha(Path(__file__))))
 records=json.loads((C/'frozen_assets.json').read_text())
 assert len(records)==6 and {(q['system'],q['seed']) for q in records}=={(s,n) for s in ['main_only','decoder','fpn'] for n in [2029,2030]}
 # All six checkpoint hashes are checked once at deployment/admission; unchanged files use stat receipts between stages.
 receipt=json.loads((C/'checkpoint_admission.json').read_text())
 for q in records:
  st=Path(q['checkpoint']).stat();expected=receipt['checkpoints'][q['checkpoint']]
  assert st.st_size==expected['size'] and st.st_mtime_ns==expected['mtime_ns'],q['checkpoint']
 adapter=load(S/'server_efficiency_same_output_adapter_v1.py','old_cost_adapter')
 fpnmod=load(W/'src/models/yolo26_fpn_reference.py','src.models.yolo26_fpn_reference')
 dump(O/'frozen_assets.json',records)
 def build(q,device):
  if q['system']=='fpn':m=fpnmod.Yolo26FPNReference(common_channels=16)
  else:m=adapter.construct(q['unit'],q['registration']['constructor_config'])
  ck=torch.load(q['checkpoint'],map_location='cpu',weights_only=False)
  key=q['registration']['checkpoint_state_key'] if q['registration'] else 'model_state_dict'
  m.load_state_dict(ck[key],strict=True);del ck;return m.eval().to(device)
 models={f"{q['system']}_{q['seed']}":build(q,'cuda') for q in records}
 from math import log
 taus=[i/20 for i in range(1,20)]
 boundaries=torch.tensor([log(t/(1-t)) for t in taus],dtype=torch.float32,device='cuda')
 histogram={k:{v:{} for v in ['clean','degraded']} for k in models}
 dump(O/'selection_protocol_frozen.json',dict(status='FIXED_BEFORE_VAL_CALIBRATION',taus=taus,threshold_rule='logit_fg-logit_bg > log(tau/(1-tau)); tie background; threshold after output interpolation',primary='per sensor event-equal absolute relative extent error; GT-zero excluded with count',smoothing='within minimum plus1pp; closest to0.5 then larger tau',secondary='per sensor pooled F1 maximum; exact ties closest0.5 then larger tau',group='system,seed,sensor,clean_or_degraded',split='val',evaluation_seed=12026,test_access=False,training=False,global_event_aggregation='sum S1/S2 confusion counts within event before event equal aggregation',controlled_perturbation_only=True))
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
      # One bucketization per forward; suffix sums give all19 threshold counts.
      margin=(logits[:,1]-logits[:,0]).contiguous()
      bucket=torch.bucketize(margin,boundaries,right=False)
      accepted=vv & target.ne(255)
      batch_ids=torch.arange(len(xx),device='cuda')[:,None,None].expand_as(target)
      gt=target.eq(1)
      encoded=(batch_ids*40+gt.long()*20+bucket)[accepted]
      counts=torch.bincount(encoded,minlength=len(xx)*40).reshape(len(xx),2,20).cpu().numpy()
      for i,event in enumerate(batch['event_id']):
       event_key=sensor+'|'+event
       if event_key not in histogram[key][view]:histogram[key][view][event_key]=counts[i].copy()
       else:histogram[key][view][event_key]+=counts[i]

      for i,event in enumerate(batch['event_id']):d['per_event'].setdefault(event,meter()).update(logits[i:i+1],target[i:i+1],vv[i:i+1])
    if bi%10==0:
     dump(O/'progress.json',dict(stage='val_threshold_histograms',batches_done=bi+1,batches_total=len(loader),elapsed_seconds=time.time()-start));print('EVAL',bi+1,len(loader),flush=True)
 manifest.close()
 for key,views in meters.items():
  result={v:{k:(z.compute() if k=='overall' else {n:t.compute() for n,t in z.items()}) for k,z in d.items()} for v,d in views.items()}
  dump(O/f'{key}_reevaluation.json',dict(metrics=result,shared_input_stream_sha256=streamhash.hexdigest(),evaluation_seed=12026))
 # Six new model identities have no prior fixed12026 val counts. Validate threshold0.5 via independent bucket and argmax meters.
 import numpy as np
 all_counts={};selected=[]
 previous=B/'results/val_event_counts_20261010_v1'
 for key,views in histogram.items():
  all_counts[key]={}
  prior={v:{'per_event':{e:mm.compute() for e,mm in meters[key][v]['per_event'].items()}} for v in ['clean','degraded']}
  for view,hists in views.items():
   all_counts[key][view]={};baseline_events={}
   for event_key,h in hists.items():
    sensor,event=event_key.split('|');suffix=np.flip(np.cumsum(np.flip(h,axis=1),axis=1),axis=1)
    rows=[dict(tau=t,TP=int(suffix[1,j+1]),FP=int(suffix[0,j+1]),FN=int(h[1].sum()-suffix[1,j+1])) for j,t in enumerate(taus)]
    all_counts[key][view][event_key]=rows
    baseline_events.setdefault(event,dict(TP=0,FP=0,FN=0))
    for k in ['TP','FP','FN']:baseline_events[event][k]+=rows[9][k]
   for event,c in baseline_events.items():
    cm=prior[view]['per_event'][event]['confusion_matrix'];assert c==dict(TP=cm[1][1],FP=cm[0][1],FN=cm[1][0]),(key,view,event,c,cm)
   for sensor in ['S1','S2']:
    es={e:rows for e,rows in all_counts[key][view].items() if e.startswith(sensor+'|')};scores=[]
    for j,t in enumerate(taus):
     rs=[rows[j] for rows in es.values()];valid_rs=[r for r in rs if r['TP']+r['FN']>0]
     assert valid_rs,(key,view,sensor)
     err=statistics.mean(100*abs(r['FP']-r['FN'])/(r['TP']+r['FN']) for r in valid_rs)
     tp=sum(r['TP'] for r in rs);fp=sum(r['FP'] for r in rs);fn=sum(r['FN'] for r in rs)
     scores.append(dict(tau=t,MARE_percent=err,pooled_F1=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0,events=len(rs),relative_error_events=len(valid_rs)))
    minimum=min(s['MARE_percent'] for s in scores);best=max(s['pooled_F1'] for s in scores)
    closest=lambda a:min(a,key=lambda s:(abs(round(20*s['tau'])-10),-s['tau']))
    primary=closest([s for s in scores if s['MARE_percent']<=minimum+1]);secondary=closest([s for s in scores if s['pooled_F1']==best])
    selected.append(dict(model=key,view=view,sensor=sensor,extent_selected=primary,F1_selected=secondary,baseline=scores[9],curve=scores))
 dump(O/'event_threshold_counts.json',dict(taus=taus,counts=all_counts,scope='val only; same inputs for six extension checkpoints'))
 dump(O/'selected_thresholds_frozen.json',dict(status='FROZEN_AFTER_VAL_BEFORE_ANY_CALIBRATION_TEST',thresholds=selected,training=False,test_access=False))
 dump(O/'threshold_freeze_receipt.json',dict(selection_sha256=sha(O/'selected_thresholds_frozen.json'),counts_sha256=sha(O/'event_threshold_counts.json'),protocol_sha256=sha(O/'selection_protocol_frozen.json'),baseline_val_event_counts_equal=True,expected_input_stream_sha256=json.loads((previous/'reevaluation_complete.json').read_text())['input_stream_sha256']))
 assert streamhash.hexdigest()==json.loads((previous/'reevaluation_complete.json').read_text())['input_stream_sha256']

 for q in records:
  st=Path(q['checkpoint']).stat();expected=receipt['checkpoints'][q['checkpoint']]
  assert st.st_size==expected['size'] and st.st_mtime_ns==expected['mtime_ns'],q['checkpoint']
 dump(O/'reevaluation_complete.json',dict(status='COMPLETE',elapsed_seconds=time.time()-start,input_stream_sha256=streamhash.hexdigest(),models=6,tiles=4853,split='val',old_spatial_counts_equal=None,threshold_count=19,frozen_threshold_units=24,baseline_val_event_counts_equal=True,training=False,test_access=False,checkpoint_reselection=False))
if __name__=='__main__':
 try:main()
 except Exception:
  if O.exists():dump(O/'failure.json',dict(status='FAILED',traceback=traceback.format_exc()))
  raise
