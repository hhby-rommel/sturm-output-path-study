# Copyright (C) 2026 STURM output-path study contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
# Source-release notice added 2026-10-11; numerical code unchanged.
"""Frozen val boundary/geometry study; thresholds from train/val GT only."""
from pathlib import Path
import sys,json,hashlib,time,gc,traceback,statistics
import numpy as np
from scipy import ndimage as ndi
B=Path(__file__).resolve().parents[1];W=B/'experiments/followup_server_20261008_v1/workspace'
O=B/'results/spatial_val_20261010_v1';OLD=B/'results/output_path_followup_20261010_v1'
sys.path.insert(0,str(W));sys.path.append(str(B))
from scripts.server_output_path_followup_20261010 import load,sha,dump
CONNECT=np.ones((3,3),bool)
def labels(gt,valid):return ndi.label(gt&valid,CONNECT)
def component_scores(gt,pred,valid,thresholds):
 g,ng=labels(gt,valid);p,npred=labels(pred,valid)
 ga=np.bincount(g.ravel(),minlength=ng+1);pa=np.bincount(p.ravel(),minlength=npred+1)
 overlap=(g>0)&(p>0);pairs,counts=np.unique(g[overlap].astype(np.int64)*(npred+1)+p[overlap],return_counts=True)
 best=np.zeros(ng+1)
 for pair,intersection in zip(pairs,counts):
  gi=int(pair//(npred+1));pi=int(pair%(npred+1));best[gi]=max(best[gi],intersection/(ga[gi]+pa[pi]-intersection))
 rows=[]
 for i in range(1,ng+1):
  bucket='small' if ga[i]<=thresholds[0] else ('medium' if ga[i]<=thresholds[1] else 'large')
  rows.append((bucket,int(ga[i]),float(best[i]),int(best[i]>=.5)))
 return rows
def boundaries(gt,pred,valid,tolerance):
 # Exclude tile-border and invalid-neighbour artificial contours.
 interior=ndi.binary_erosion(valid,CONNECT,border_value=0)
 bg=(gt&~ndi.binary_erosion(gt,CONNECT,border_value=0))&interior
 bp=(pred&~ndi.binary_erosion(pred,CONNECT,border_value=0))&interior
 # Disk tolerance in pixel units; prevent matches across invalid holes.
 yy,xx=np.ogrid[-tolerance:tolerance+1,-tolerance:tolerance+1];disk=xx*xx+yy*yy<=tolerance*tolerance
 domain=ndi.binary_erosion(valid,disk,border_value=0)
 bg&=domain;bp&=domain
 mg=int((bp&ndi.binary_dilation(bg,disk)).sum());mp=int((bg&ndi.binary_dilation(bp,disk)).sum())
 return [mg,int(bp.sum()),mp,int(bg.sum())]
def main():
 O.mkdir(exist_ok=False);start=time.time()
 import torch
 import torch.nn.functional as F
 from src.dataset.sturm_flood import SturmFloodDataset,grouped_sensor_collate
 # This protocol is written before loading any model predictions.
 protocol=dict(split='fixed_full_val',test_access=False,training=False,seeds=[2026,2027,2028],systems=['main_only','decoder','fpn'],views=['clean','degraded'],main_boundary_tolerance=2,sensitivity_boundary_tolerance=1,boundary='water-side 8-neighbour morphological contour; exclude invalid/tile-edge tolerance neighbourhood',matching='Euclidean disk dilation',zero_denominator='zero precision/recall/F1; aggregate matched and total contours before F1',component_connectivity=8,threshold_source='all train+val GT components, per sensor; numpy linear 33rd/67th quantiles',component_metric='GT-centric best IoU against any overlapping predicted component; union includes entire predicted component valid area and hence nearby FP',component_recall_IoU_threshold=.5,matching_not_one_to_one=True,unmatched_predictions='overall FP reported separately; no size-bin attribution of unmatched FP',objects='tile-contained connected components, not complete physical water bodies',global_eval_seed=12026,scope='spatial behavior, not physical mechanism identification')
 dump(O/'protocol_frozen_before_predictions.json',protocol)
 areas={'S1':[],'S2':[]};total=0
 for split in ['train','val']:
  ds=SturmFloodDataset(W/'data/manifests/manifest.csv',W/'data/splits/event_split_v1.csv',split)
  for i in range(len(ds)):
   item=ds[i];g,n=labels(item['mask'].numpy()==1,item['valid_mask'].numpy().astype(bool));areas[item['sensor']].extend(np.bincount(g.ravel(),minlength=n+1)[1:].astype(int).tolist());total+=1
   if total%200==0:
    dump(O/'progress.json',dict(stage='train_val_GT_thresholds',tiles_done=total,tiles_total=19414,elapsed_seconds=time.time()-start));print('GT',total,19414,flush=True)
 thresholds={s:np.quantile(v,[.33,.67],method='linear').tolist() for s,v in areas.items()}
 dump(O/'thresholds_frozen.json',dict(thresholds=thresholds,component_counts={s:len(v) for s,v in areas.items()},GT_tiles=total,model_predictions_seen=False,protocol_sha256=sha(O/'protocol_frozen_before_predictions.json')))
 np.savez_compressed(O/'train_val_GT_component_areas.npz',**{s:np.array(v,dtype=np.int64) for s,v in areas.items()});del areas
 torch.set_num_threads(4);torch.backends.cudnn.allow_tf32=True;torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=False
 assert torch.__version__=='2.1.0+cu121' and torch.cuda.get_device_name()=='NVIDIA GeForce RTX 4090 D'
 adapter=load(B/'experiments/same_output_efficiency_20261004_v1/server_efficiency_same_output_adapter_v1.py','spatial_old_adapter')
 fpn=load(W/'src/models/yolo26_fpn_reference.py','src.models.yolo26_fpn_reference')
 records=json.loads((OLD/'frozen_assets.json').read_text());models={}
 for q in records:
  assert sha(Path(q['checkpoint']))==q['checkpoint_sha256']
  m=fpn.Yolo26FPNReference(common_channels=16) if q['system']=='fpn' else adapter.construct(q['unit'],q['registration']['constructor_config'])
  ck=torch.load(q['checkpoint'],map_location='cpu',weights_only=False);key=q['registration']['checkpoint_state_key'] if q['registration'] else 'model_state_dict';m.load_state_dict(ck[key],strict=True);del ck
  models[f"{q['system']}_{q['seed']}"]=m.eval().cuda()
 ds=SturmFloodDataset(W/'data/manifests/manifest.csv',W/'data/splits/event_split_v1.csv','val');ds.rows.sort(key=lambda r:(r['sensor'],r['event_id'],r['tile_id'],r['image_path']))
 loader=torch.utils.data.DataLoader(ds,batch_size=16,shuffle=False,num_workers=0,collate_fn=grouped_sensor_collate)
 receipts=(O/'sample_stream_sha.jsonl').open('x');sums={}
 def rng(identifier,stream):
  seed=int.from_bytes(hashlib.sha256(f'12026|{identifier}|{stream}'.encode()).digest()[:8],'little')%(2**63-1)
  return torch.Generator().manual_seed(seed)
 with torch.inference_mode():
  for bi,group in enumerate(loader):
   for sensor,batch in group.items():
    x=batch['image'];valid=batch['valid_mask'].bool();changed=[];keeps=[]
    for i in range(len(x)):
     row=next(r for r in ds.rows[bi*16:bi*16+16] if r['sensor']==sensor and r['tile_id']==batch['tile_id'][i]);identifier='/'.join(row[k] for k in ['sensor','event_id','tile_id','image_path'])
     sev=.5+.5*torch.rand((),generator=rng(identifier,'severity'));noise=torch.randn(x[i].shape,generator=rng(identifier,'gaussian'))
     z=x[i]*(1+.12*sev*noise) if sensor=='S1' else x[i]+.20*sev*x[i].flatten(1).std(1,unbiased=False).clamp_min(.001)[:,None,None]*noise
     keep=valid[i]&~((torch.rand(valid[i].shape,generator=rng(identifier,'deletion'))<(.05+.10*sev))&valid[i]);z=torch.where(keep[None],z,torch.zeros_like(z))
     changed.append(z);keeps.append(keep);receipts.write(json.dumps(dict(identifier=identifier,input_valid_sha256=hashlib.sha256(z.numpy().tobytes()+keep.numpy().tobytes()).hexdigest()))+'\n')
    for view,xx,vv in [('clean',x,valid),('degraded',torch.stack(changed),torch.stack(keeps))]:
     xx=xx.cuda();vv=vv.numpy();gt=batch['mask'].numpy()==1
     for key,m in models.items():
      logits=m.forward_native(xx,sensor)
      if logits.shape[-2:]!=(128,128):logits=F.interpolate(logits,size=(128,128),mode='bilinear',align_corners=False)
      assert torch.isfinite(logits).all();pred=logits.argmax(1).cpu().numpy()==1
      for i in range(len(pred)):
       groupkey=f'{key}/{view}/{sensor}';d=sums.setdefault(groupkey,dict(boundary={'1':[0]*4,'2':[0]*4},components={b:[0,0.,0,0] for b in ['small','medium','large']},TP=0,FP=0,FN=0))
       for tol in [1,2]:d['boundary'][str(tol)]=[a+b for a,b in zip(d['boundary'][str(tol)],boundaries(gt[i],pred[i],vv[i],tol))]
       for bucket,area,iou,recall in component_scores(gt[i],pred[i],vv[i],thresholds[sensor]):
        v=d['components'][bucket];v[0]+=1;v[1]+=iou;v[2]+=recall;v[3]+=area
       d['TP']+=int((gt[i]&pred[i]&vv[i]).sum());d['FP']+=int((~gt[i]&pred[i]&vv[i]).sum());d['FN']+=int((gt[i]&~pred[i]&vv[i]).sum())
   if bi%5==0:
    dump(O/'partial_counts.json',sums);dump(O/'progress.json',dict(stage='fixed_val_spatial_inference',batches_done=bi+1,batches_total=len(loader),elapsed_seconds=time.time()-start));print('VAL',bi+1,len(loader),flush=True)
 receipts.close();dump(O/'spatial_counts.json',sums)
 summary={}
 for key,d in sums.items():
  out=dict(boundary={},components={},TP=d['TP'],FP=d['FP'],FN=d['FN'])
  for tol,(mp,npred,mg,ng) in d['boundary'].items():
   precision=mp/max(npred,1);recall=mg/max(ng,1);out['boundary'][tol]=dict(precision=precision,recall=recall,F1=2*precision*recall/max(precision+recall,1e-12))
  for bucket,(n,iou,recall,area) in d['components'].items():out['components'][bucket]=dict(n=n,mean_best_IoU=iou/max(n,1),recall_at_05=recall/max(n,1),GT_pixels=area)
  summary[key]=out
 dump(O/'summary.json',summary)
 for q in records:assert sha(Path(q['checkpoint']))==q['checkpoint_sha256']
 dump(O/'completion.json',dict(status='FIXED_VAL_BOUNDARY_AND_COMPONENT_COMPLETE',elapsed_seconds=time.time()-start,test_access=False,training=False,thresholds=thresholds,models=9,val_tiles=len(ds)))
if __name__=='__main__':
 try:main()
 except Exception as e:
  if O.exists():dump(O/'failure.json',dict(type=type(e).__name__,message=str(e),traceback=traceback.format_exc(),automatic_retry=False))
  raise
