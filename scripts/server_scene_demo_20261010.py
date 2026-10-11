# Copyright (C) 2026 STURM output-path study contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
# Source-release notice added 2026-10-11; numerical code unchanged.
"""Preselected clean scene workflow demonstration, frozen seed2027; no training."""
from pathlib import Path
import sys,json,hashlib,time,gc,traceback
import numpy as np
import rasterio
from rasterio.transform import Affine
B=Path(__file__).resolve().parents[1];W=B/'experiments/followup_server_20261008_v1/workspace';O=B/'results/scene_demo_20261010_v1'
sys.path.insert(0,str(W));sys.path.append(str(B))
from scripts.server_output_path_followup_20261010 import load,sha,dump
def main():
 O.mkdir(exist_ok=False);start=time.time()
 dump(O/'protocol_frozen_before_predictions.json',dict(scope='one S1 and one S2 available event/AOI tile mosaic; workflow illustration, not new generalization evidence',selection='per sensor choose lexicographically first test event; then AOI/CRS group with most tiles, lexicographic tie break',seed=2027,systems=['main_only','decoder','fpn'],condition='clean',overlap='arithmetic mean of two-class logits over valid contributions',GT_overlap='reject conflicting valid labels',outside_coverage='nodata255',training=False,checkpoint_reselection=False,test_access=True,scene_identity_frozen_before_inference=True,maximum_canvas_pixels=40000000))
 import torch
 import torch.nn.functional as F
 from src.dataset.sturm_flood import SturmFloodDataset
 torch.set_num_threads(4);torch.backends.cudnn.allow_tf32=True;torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=False
 assert torch.__version__=='2.1.0+cu121' and torch.cuda.get_device_name()=='NVIDIA GeForce RTX 4090 D'
 ds=SturmFloodDataset(W/'data/manifests/manifest.csv',W/'data/splits/event_split_v1.csv','test');selected={}
 for sensor in ['S1','S2']:
  event=min(r['event_id'] for r in ds.rows if r['sensor']==sensor);groups={}
  for i,row in enumerate(ds.rows):
   if row['sensor']==sensor and row['event_id']==event:groups.setdefault((row['aoi_code'],row['image_crs']),[]).append(i)
  group=min(groups,key=lambda k:(-len(groups[k]),k));selected[sensor]=dict(event=event,AOI=group[0],CRS=group[1],indices=groups[group],tiles=len(groups[group]))
 dump(O/'selection_frozen.json',dict(scenes=selected,test_manifest_sha256=sha(W/'data/manifests/manifest.csv'),split_sha256=sha(W/'data/splits/event_split_v1.csv'),no_model_results_used=True))
 adapter=load(B/'experiments/same_output_efficiency_20261004_v1/server_efficiency_same_output_adapter_v1.py','scene_old_adapter');fpn=load(W/'src/models/yolo26_fpn_reference.py','src.models.yolo26_fpn_reference')
 records=[q for q in json.loads((B/'results/output_path_followup_20261010_v1/frozen_assets.json').read_text()) if q['seed']==2027];models={}
 for q in records:
  assert sha(Path(q['checkpoint']))==q['checkpoint_sha256'];m=fpn.Yolo26FPNReference(common_channels=16) if q['system']=='fpn' else adapter.construct(q['unit'],q['registration']['constructor_config'])
  ck=torch.load(q['checkpoint'],map_location='cpu',weights_only=False);m.load_state_dict(ck[q['registration']['checkpoint_state_key'] if q['registration'] else 'model_state_dict'],strict=True);del ck;models[q['system']]=m.eval().cuda()
 dump(O/'frozen_assets.json',records)
 for sensor,selection in selected.items():
  D=O/sensor;D.mkdir();headers=[]
  for i in selection['indices']:
   row=ds.rows[i]
   with rasterio.open(ds._path(row['image_path'])) as r:headers.append(dict(index=i,transform=r.transform,width=r.width,height=r.height,crs=r.crs,bounds=r.bounds))
   with rasterio.open(ds._path(row['mask_path'])) as r:assert r.crs==headers[-1]['crs'] and r.transform==headers[-1]['transform'],'GT/image georeference mismatch'
  first=headers[0];a=first['transform'];assert a.b==0 and a.d==0 and a.a>0 and a.e<0,'Unsupported rotated/nonstandard raster'
  left=min(h['bounds'].left for h in headers);top=max(h['bounds'].top for h in headers);right=max(h['bounds'].right for h in headers);bottom=min(h['bounds'].bottom for h in headers)
  width=int(round((right-left)/a.a));height=int(round((top-bottom)/(-a.e)));assert width*height<=40000000,'Canvas exceeds frozen resource bound; no alternate event selection'
  transform=Affine(a.a,0,left,0,a.e,top);count=np.zeros((height,width),np.uint16);gt=np.full((height,width),255,np.uint8);display=np.zeros((height,width),np.float32);cached=[]
  io_start=time.perf_counter()
  for h in headers:
   t=h['transform'];assert t.a==a.a and t.e==a.e
   col=(t.c-left)/a.a;row=(top-t.f)/(-a.e);assert abs(col-round(col))<1e-5 and abs(row-round(row))<1e-5,'Nonaligned tile grid'
   col=int(round(col));row=int(round(row));item=ds[h['index']];valid=item['valid_mask'].numpy().astype(bool);target=item['mask'].numpy().astype(np.uint8);window=np.s_[row:row+h['height'],col:col+h['width']]
   prior=gt[window];assert not np.any(valid&(prior!=255)&(prior!=target)),'Conflicting overlap GT'
   prior[valid]=target[valid];gt[window]=prior;count[window]+=valid;display[window]+=item['image'][0].numpy()*valid;cached.append((item['image'],valid,window))
  data_seconds=time.perf_counter()-io_start;covered=count>0;display/=np.maximum(count,1)
  profile=dict(driver='GTiff',height=height,width=width,count=1,crs=first['crs'],transform=transform,compress='deflate')
  def save(name,array,nodata):
   with rasterio.open(D/name,'w',dtype=array.dtype,nodata=nodata,**profile) as f:f.write(array,1)
  save('GT.tif',gt,255);save('valid_coverage.tif',covered.astype(np.uint8),0);display[~covered]=np.nan;save('input_channel0.tif',display,np.nan)
  reports={}
  for system,m in models.items():
   scores=np.zeros((2,height,width),np.float32);raw_ms=[];host_start=time.perf_counter()
   with torch.inference_mode():
    # Separate warmup from inference-only CUDA event timing.
    fixture=cached[0][0][None].cuda()
    for _ in range(10):m.forward_native(fixture,sensor)
    torch.cuda.synchronize();del fixture
    for j in range(0,len(cached),16):
     tiles=cached[j:j+16];x=torch.stack([v[0] for v in tiles]).cuda();t0=torch.cuda.Event(enable_timing=True);t1=torch.cuda.Event(enable_timing=True)
     t0.record();logits=m.forward_native(x,sensor)
     if logits.shape[-2:]!=(128,128):logits=F.interpolate(logits,size=(128,128),mode='bilinear',align_corners=False)
     t1.record();torch.cuda.synchronize();raw_ms.append(t0.elapsed_time(t1));assert torch.isfinite(logits).all();arr=logits.cpu().numpy()
     for k,(_,valid,window) in enumerate(tiles):scores[:,window[0],window[1]]+=arr[k]*valid[None]
     dump(O/'progress.json',dict(stage='scene_inference',sensor=sensor,event=selection['event'],system=system,tiles_done=min(j+16,len(cached)),tiles_total=len(cached)));print('SCENE',sensor,system,min(j+16,len(cached)),len(cached),flush=True)
   scores/=np.maximum(count,1)[None];pred=scores.argmax(0).astype(np.uint8);pred[~covered]=255
   tp=covered&(gt==1)&(pred==1);fp=covered&(gt==0)&(pred==1);fn=covered&(gt==1)&(pred==0);error=np.zeros_like(gt);error[tp]=1;error[fp]=2;error[fn]=3;error[~covered]=255
   save(f'{system}_prediction.tif',pred,255);save(f'{system}_FP_FN.tif',error,255)
   TP=int(tp.sum());FP=int(fp.sum());FN=int(fn.sum());reports[system]=dict(TP=TP,FP=FP,FN=FN,F1=2*TP/max(2*TP+FP+FN,1),inference_only_CUDA_ms=sum(raw_ms),raw_batch_CUDA_ms=raw_ms,forward_copy_and_stitch_wall_seconds=time.perf_counter()-host_start)
   del scores,pred,error,logits,arr,x;gc.collect()
  dump(D/'scene_report.json',dict(selection=selection,width=width,height=height,valid_covered_pixels=int(covered.sum()),covered_area_CRS_units_squared=float(covered.sum()*abs(a.a*a.e)),CRS=str(first['crs']),overlap_pixels=int((count>1).sum()),max_overlap=int(count.max()),data_loading_seconds=data_seconds,models=reports,original_full_scene_coverage_not_asserted=True))
  del cached,count,gt,display,covered;gc.collect()
 for q in records:assert sha(Path(q['checkpoint']))==q['checkpoint_sha256']
 dump(O/'completion.json',dict(status='TWO_SENSOR_SELECTED_AOI_SCENE_DEMOS_COMPLETE',elapsed_seconds=time.time()-start,training=False,seed=2027,scene_selection=selected,statistical_generalization_claim=False))
if __name__=='__main__':
 try:main()
 except Exception as e:
  if O.exists():dump(O/'failure.json',dict(type=type(e).__name__,message=str(e),traceback=traceback.format_exc(),automatic_retry=False))
  raise
