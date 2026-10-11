# Copyright (C) 2026 STURM output-path study contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
# Source-release notice added 2026-10-11; numerical code unchanged.
"""Same-session CPU repeated forward measurement; no datasets or training."""
from pathlib import Path
import sys,json,time,statistics,gc,subprocess,traceback,importlib.util,hashlib
B=Path(__file__).resolve().parents[1];W=B/'experiments/followup_server_20261008_v1/workspace';S=B/'experiments/same_output_efficiency_20261004_v1';O=B/'results/cpu_matched_repeat_20261011_v1'
sys.path.insert(0,str(W));sys.path.append(str(B))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def dump(p,q):p.write_text(json.dumps(q,indent=2,allow_nan=False)+'\n')
def load(p,n):
 s=importlib.util.spec_from_file_location(n,p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
def main():
 O.mkdir(exist_ok=False);start=time.time()
 import torch
 import torch.nn.functional as F
 assert torch.__version__=='2.1.0+cu121';torch.set_num_threads(4);torch.set_num_interop_threads(1)
 records=json.loads((B/'results/val_event_counts_20261010_v1/frozen_assets.json').read_text())
 adapter=load(S/'server_efficiency_same_output_adapter_v1.py','cpu_cost_adapter');fpn=load(W/'src/models/yolo26_fpn_reference.py','src.models.yolo26_fpn_reference')
 dump(O/'protocol_frozen.json',dict(authorization='User20261011: fast supplemental experiments',models=9,units=18,threads=4,interop_threads=1,batch=1,warmup=20,blocks=5,repeats_per_block=50,fixture='zero float32 [1,2/9,128,128]',output='128x128 2-class logits; native bilinear interpolation included',training=False,dataset_access=False,test_access=False,selection='all units and all repeats retained',process='single session; one loaded model at a time'))
 dump(O/'environment.json',dict(torch=torch.__version__,CPU=subprocess.check_output(['lscpu'],text=True),script_sha256=sha(Path(__file__)),mkldnn_enabled=torch.backends.mkldnn.enabled,python=sys.version))
 units=[]
 for q in records:
  assert sha(Path(q['checkpoint']))==q['checkpoint_sha256']
  m=fpn.Yolo26FPNReference(common_channels=16) if q['system']=='fpn' else adapter.construct(q['unit'],q['registration']['constructor_config'])
  ck=torch.load(q['checkpoint'],map_location='cpu',weights_only=False);key=q['registration']['checkpoint_state_key'] if q['registration'] else 'model_state_dict';m.load_state_dict(ck[key],strict=True);del ck;m=m.eval().cpu()
  for sensor,channels in [('S1',2),('S2',9)]:
   x=torch.zeros(1,channels,128,128)
   def forward():
    z=m.forward_native(x,sensor);return F.interpolate(z,size=(128,128),mode='bilinear',align_corners=False) if z.shape[-2:]!=(128,128) else z
   with torch.inference_mode():
    z=forward();assert z.shape==(1,2,128,128) and torch.isfinite(z).all();del z
    for _ in range(20):z=forward();del z
    blocks=[]
    for block in range(5):
     raw=[]
     for _ in range(50):
      t=time.perf_counter();z=forward();raw.append(1000*(time.perf_counter()-t));del z
     blocks.append(raw)
   times=sum(blocks,[]);unit=dict(system=q['system'],seed=q['seed'],sensor=sensor,mean_ms=statistics.mean(times),repeat_SD_ms=statistics.stdev(times),block_means_ms=[statistics.mean(x) for x in blocks],blocks_raw_ms=blocks,checkpoint_sha256=q['checkpoint_sha256'])
   units.append(unit);dump(O/f"{q['system']}_{q['seed']}_{sensor}.json",unit)
   dump(O/'progress.json',dict(units_done=len(units),units_total=18,elapsed_seconds=time.time()-start));print('CPU',len(units),18,flush=True)
  del m;gc.collect();assert sha(Path(q['checkpoint']))==q['checkpoint_sha256']
 summary={}
 for system in ['main_only','decoder','fpn']:
  for sensor in ['S1','S2']:
   r=[u for u in units if u['system']==system and u['sensor']==sensor];assert len(r)==3
   a=[u['mean_ms'] for u in r];summary[system+'/'+sensor]=dict(mean_ms=statistics.mean(a),weight_sample_SD_ms=statistics.stdev(a),weight_means_ms=a,repeat_SD_ms=[u['repeat_SD_ms'] for u in r])
 dump(O/'summary.json',summary);dump(O/'completion.json',dict(status='COMPLETE',units=18,raw_times=4500,elapsed_seconds=time.time()-start,training=False,dataset_access=False,test_access=False))
if __name__=='__main__':
 try:main()
 except Exception:
  if O.exists():dump(O/'failure.json',dict(traceback=traceback.format_exc()))
  raise
