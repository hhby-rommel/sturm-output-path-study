"""One authorized AOI extension; original per-seed numeric implementation retained."""
from pathlib import Path
from datetime import datetime,timezone
import argparse,csv,hashlib,json,os,shutil,subprocess,sys,time,traceback
B=Path('/root/autodl-tmp/sturm_codex_20260919');C=B/'experiments/AOI_five_seed_20261021_v1';O=B/'results/AOI_five_seed_20261021_v1'
def read(p):return json.loads(p.read_text())
def dump(p,x):p.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for block in iter(lambda:f.read(1048576),b''):h.update(block)
 return h.hexdigest()
def preflight():
 import torch,rasterio
 assert not O.exists(),'Existing attempt; do not retry or overwrite'
 assert torch.__version__=='2.1.0+cu121' and torch.cuda.get_device_name()=='NVIDIA GeForce RTX 4090 D'
 assert shutil.disk_usage(B).free>2*1024**3
 for q in read(C/'deployment_identity.json'):assert sha(C/q['name'])==q['sha256'],q['name']
 sources=[]
 for q in read(C/'source_identity.json'):
  assert sha(Path(q['path']))==q['sha256'],q['path'];sources.append(dict(**q,conclusion='MATCH'))
 frozen=read(C/'selection_frozen.json');W=B/'experiments/followup_server_20261008_v1/workspace'
 for name,key in [('data/manifests/manifest.csv','test_manifest_sha256'),('data/splits/event_split_v1.csv','split_sha256')]:assert sha(W/name)==frozen[key],name
 assets=read(C/'frozen_assets_full.json');assert len(assets)==15
 admission=[]
 for q in assets:
  assert sha(Path(q['checkpoint']))==q['checkpoint_sha256'],q['checkpoint']
  admission.append(dict(system=q['system'],seed=q['seed'],path=q['checkpoint'],sha256=q['checkpoint_sha256'],purpose='Formal checkpoint identity at new AOI inference handoff',time_utc=datetime.now(timezone.utc).isoformat(),conclusion='MATCH'))
 assert {(q['system'],q['seed']) for q in assets}=={(s,n) for s in ('main_only','decoder','fpn') for n in range(2026,2031)}
 for sensor in ('S1','S2'):
  d=B/'results/scene_demo_20261010_v1'/sensor
  assert read(d/'scene_report.json')['selection']==frozen['scenes'][sensor]
  for name in ['GT.tif','valid_coverage.tif','main_only_prediction.tif','decoder_prediction.tif','fpn_prediction.tif']:assert (d/name).is_file()
 result=dict(status='PREFLIGHT_PASS',time_utc=datetime.now(timezone.utc).isoformat(),torch=torch.__version__,rasterio=rasterio.__version__,GPU=torch.cuda.get_device_name(),checkpoint_admission=admission,original_source_admission=sources)
 dump(C/'preflight.json',result);print(json.dumps({k:v for k,v in result.items() if 'admission' not in k}),flush=True)
 return result
def collect():
 import numpy as np,rasterio
 rows=[];reused=O/'seed2027';reused.mkdir()
 for seed in range(2026,2031):
  for sensor in ('S1','S2'):
   source=B/'results/scene_demo_20261010_v1'/sensor if seed==2027 else O/f'seed{seed}'/sensor
   report=read(source/'scene_report.json');assert report['overlap_pixels']==0
   if seed==2027:
    dest=reused/sensor;dest.mkdir()
    for name in ['GT.tif','valid_coverage.tif','main_only_prediction.tif','decoder_prediction.tif','fpn_prediction.tif','scene_report.json']:shutil.copy2(source/name,dest/name)
   with rasterio.open(source/'GT.tif') as f:gt=f.read(1);transform=f.transform;assert str(f.crs)=='EPSG:32635' and abs(transform.a*transform.e)==100
   with rasterio.open(source/'valid_coverage.tif') as f:covered=f.read(1)>0
   assert int(covered.sum())==report['valid_covered_pixels']
   for system in ('main_only','decoder','fpn'):
    with rasterio.open(source/f'{system}_prediction.tif') as f:
     assert f.transform==transform;pred=f.read(1)
    TP=int(np.sum(covered&(gt==1)&(pred==1)));FP=int(np.sum(covered&(gt==0)&(pred==1)));FN=int(np.sum(covered&(gt==1)&(pred==0)))
    assert all(v==report['models'][system][k] for k,v in dict(TP=TP,FP=FP,FN=FN).items())
    ref=TP+FN;positive=TP+FP;bias=(positive-ref)*.0001
    rows.append(dict(sensor=sensor,event_id='EMSR292',AOI='01CHRISOUPOLI',system=system,seed=seed,condition='clean',origin='reused_original_scene' if seed==2027 else 'new_frozen_checkpoint_inference',TP=TP,FP=FP,FN=FN,valid_pixels=int(covered.sum()),reference_water_pixels=ref,predicted_water_pixels=positive,pixel_area_m2=100,valid_coverage_km2=float(covered.sum()*.0001),reference_water_area_km2=ref*.0001,predicted_water_area_km2=positive*.0001,signed_area_bias_km2=bias,absolute_area_error_km2=abs(bias),signed_area_bias_percent=100*(positive-ref)/ref,absolute_area_error_percent=100*abs(positive-ref)/ref,F1=2*TP/(2*TP+FP+FN)))
 assert len(rows)==30
 summary=[]
 metrics=['predicted_water_area_km2','signed_area_bias_km2','absolute_area_error_km2','signed_area_bias_percent','absolute_area_error_percent','F1']
 for sensor in ('S1','S2'):
  group=[q for q in rows if q['sensor']==sensor]
  assert len({q['valid_pixels'] for q in group})==len({q['reference_water_pixels'] for q in group})==1
  for system in ('main_only','decoder','fpn'):
   rr=[q for q in group if q['system']==system];assert len(rr)==5
   s=dict(sensor=sensor,system=system,seeds='2026;2027;2028;2029;2030',n=5,valid_coverage_km2=rr[0]['valid_coverage_km2'],reference_water_area_km2=rr[0]['reference_water_area_km2'])
   for metric in metrics:
    x=np.array([q[metric] for q in rr]);s[metric+'_mean']=float(x.mean());s[metric+'_sample_SD']=float(x.std(ddof=1))
   summary.append(s)
 for name,rr in [('AOI_area_all_five_seeds.csv',rows),('AOI_area_five_seed_summary.csv',summary)]:
  with (O/name).open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(rr[0]));w.writeheader();w.writerows(rr)
 dump(O/'area_verification.json',dict(status='PASS',rows=30,groups=6,source='Original-grid prediction GeoTIFFs and reference masks; pixel counts reconciled with each scene report',seed2027='Reused, no new inference',same_coverage_across_paths_and_seeds=True))
def execute():
 assert read(C/'protocol.json')['human_authorized']
 assert (C/'preflight.json').is_file() and read(C/'preflight.json')['status']=='PREFLIGHT_PASS'
 assert not O.exists();O.mkdir()
 shutil.copy2(C/'preflight.json',O/'preflight.json');shutil.copy2(C/'protocol.json',O/'protocol.json')
 start=time.time()
 try:
  for seed in (2026,2028,2029,2030):
   dump(O/'progress.json',dict(stage='inference',current_seed=seed,completed_new_seeds=[n for n in (2026,2028,2029,2030) if n<seed]))
   subprocess.run([sys.executable,'-u',str(C/f'seed{seed}.py')],cwd=B,check=True)
  collect()
  dump(O/'completion.json',dict(status='FIVE_SEED_AOI_AREA_COMPLETE',new_seeds=[2026,2028,2029,2030],reused_seed=2027,new_model_sensor_rows=24,final_rows=30,summary_rows=6,training=False,checkpoint_reselection=False,elapsed_seconds=time.time()-start))
 except BaseException as e:
  dump(O/'failure.json',dict(type=type(e).__name__,message=str(e),traceback=traceback.format_exc(),automatic_retry=False));raise
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--mode',choices=['preflight','execute'],required=True);a=p.parse_args()
 preflight() if a.mode=='preflight' else execute()
