# Copyright (C) 2026 STURM output-path study contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
# Source-release notice added 2026-10-11; numerical code unchanged.
"""Same-session four-system frozen forward costs; no dataset access or training."""
from pathlib import Path
import sys,json,csv,time,gc,statistics,subprocess,traceback,platform
B=Path(__file__).resolve().parents[1];W=B/'experiments/followup_server_20261008_v1/workspace';S=B/'experiments/same_output_efficiency_20261004_v1';O=B/'results/runtime_matched_session_20261010_v1'
sys.path.insert(0,str(W));sys.path.append(str(B))
from scripts.server_output_path_followup_20261010 import load,sha,dump
def main():
 O.mkdir(exist_ok=False);start=time.time()
 import torch
 import torch.nn.functional as F
 assert torch.__version__=='2.1.0+cu121' and torch.cuda.get_device_name()=='NVIDIA GeForce RTX 4090 D'
 torch.set_num_threads(4);torch.backends.cudnn.allow_tf32=True;torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=False
 def gpu():return subprocess.check_output(['nvidia-smi','--query-gpu=name,uuid,driver_version','--format=csv,noheader'],text=True).strip()
 identity=gpu()
 dump(O/'environment.json',dict(GPU_identity=identity,torch=torch.__version__,python=sys.version,cudnn_version=torch.backends.cudnn.version(),platform=platform.platform(),threads=4,cudnn_tf32=True,matmul_tf32=False,benchmark=False,deterministic=False,script_sha256=sha(Path(__file__))))
 records=json.loads((B/'results/output_path_followup_20261010_v1/frozen_assets.json').read_text())
 regs=json.loads((S/'registrations_90.json').read_text())['units'];units=list(csv.DictReader((S/'units_90.csv').open()))
 for seed in [2026,2027,2028]:
  u=next(q for q in units if q['system']=='unet' and int(q['seed'])==seed); records.append(dict(system='unet',seed=seed,checkpoint=u['remote_checkpoint'],checkpoint_sha256=u['expected_sha256'],unit=u,registration=regs[u['unit_id']]))
 records.sort(key=lambda q:(q['seed'],['main_only','decoder','fpn','unet'].index(q['system'])))
 for q in records:assert sha(Path(q['checkpoint']))==q['checkpoint_sha256']
 dump(O/'frozen_assets.json',records)
 plan=[dict(system=q['system'],seed=q['seed'],sensor=s,batch=b) for q in records for s in ['S1','S2'] for b in [1,2,4,8,16]]
 dump(O/'protocol_frozen.json',dict(authorization='User20261010: follow revision plan and reopened GPU server for checking',test_access=False,dataset_access=False,training=False,checkpoint_reselection=False,fixtures='fixed zero FP32 [B,2/9,128,128]',outputs='two-class128x128; native bilinear align_cornersFalse included',GPU_identity=identity,process='one controller process; one loaded model at a time; allocator cache cleared before each unit',warmup=50,blocks=3,repeats_per_block=100,order=plan,units=120,selection='all prespecified units/blocks retained',scope='device forward only; excludes IO,H2D,checkpoint load and external processing'))
 adapter=load(S/'server_efficiency_same_output_adapter_v1.py','matched_runtime_adapter');fpn=load(W/'src/models/yolo26_fpn_reference.py','src.models.yolo26_fpn_reference')
 done=0
 for q in records:
  m=fpn.Yolo26FPNReference(common_channels=16) if q['system']=='fpn' else adapter.construct(q['unit'],q['registration']['constructor_config'])
  ck=torch.load(q['checkpoint'],map_location='cpu',weights_only=False);key=q['registration']['checkpoint_state_key'] if q['registration'] else 'model_state_dict';m.load_state_dict(ck[key],strict=True);del ck
  m=m.eval().cuda()
  def forward(x,s):
   z=m.forward_native(x,s)
   return F.interpolate(z,size=(128,128),mode='bilinear',align_corners=False) if z.shape[-2:]!=(128,128) else z
  with torch.inference_mode():
   for sensor,c in [('S1',2),('S2',9)]:
    for batch in [1,2,4,8,16]:
     gc.collect();torch.cuda.empty_cache();x=torch.zeros(batch,c,128,128,device='cuda');z=forward(x,sensor);assert z.shape==(batch,2,128,128) and torch.isfinite(z).all();del z
     for _ in range(50):z=forward(x,sensor);del z
     torch.cuda.synchronize();blocks=[]
     for block in range(3):
      starts=[torch.cuda.Event(enable_timing=True) for _ in range(100)];ends=[torch.cuda.Event(enable_timing=True) for _ in range(100)]
      torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();base_a=torch.cuda.memory_allocated();base_r=torch.cuda.memory_reserved()
      for i in range(100):starts[i].record();z=forward(x,sensor);ends[i].record();del z
      torch.cuda.synchronize();blocks.append(dict(raw_ms=[a.elapsed_time(b) for a,b in zip(starts,ends)],baseline_allocated=base_a,baseline_reserved=base_r,peak_allocated=torch.cuda.max_memory_allocated(),peak_reserved=torch.cuda.max_memory_reserved()))
     assert gpu()==identity
     raw=sum([b['raw_ms'] for b in blocks],[]);unit=f"{q['system']}_{q['seed']}_{sensor}_b{batch}"
     dump(O/(unit+'.json'),dict(unit=unit,system=q['system'],seed=q['seed'],sensor=sensor,batch=batch,checkpoint_sha256=q['checkpoint_sha256'],mean_ms=statistics.mean(raw),throughput=1000*batch/statistics.mean(raw),peak_allocated_MiB=max(b['peak_allocated'] for b in blocks)/1048576,peak_reserved_MiB=max(b['peak_reserved'] for b in blocks)/1048576,blocks=blocks,GPU_identity=identity))
     done+=1;dump(O/'progress.json',dict(completed_units=done,total_units=120,last_completed=unit,elapsed_seconds=time.time()-start));print('RUNTIME',done,120,unit,flush=True);del x,starts,ends;gc.collect();torch.cuda.empty_cache()
  del m,forward;gc.collect();torch.cuda.empty_cache()
 for q in records:assert sha(Path(q['checkpoint']))==q['checkpoint_sha256']
 dump(O/'completion.json',dict(status='SAME_GPU_SESSION_120_UNITS_COMPLETE',elapsed_seconds=time.time()-start,GPU_identity=identity,units=120,raw_times=36000,test_access=False,dataset_access=False,training=False,checkpoint_reselection=False))
if __name__=='__main__':
 try:main()
 except Exception as e:
  if O.exists():dump(O/'failure.json',dict(type=type(e).__name__,message=str(e),traceback=traceback.format_exc(),automatic_retry=False))
  raise
