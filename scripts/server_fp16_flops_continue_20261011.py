# Copyright (C) 2026 STURM output-path study contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
# Source-release notice added 2026-10-11; numerical code unchanged.
"""Matched FP32/FP16 synthetic forward costs and declared major-operator FLOPs."""
from pathlib import Path
import sys,json,time,statistics,gc,subprocess,traceback,importlib.util,hashlib
B=Path(__file__).resolve().parents[1];W=B/'experiments/followup_server_20261008_v1/workspace';S=B/'experiments/same_output_efficiency_20261004_v1';O=B/'results/fp16_flops_continue_20261011_v1';OLD=B/'results/fp16_flops_20261011_v1'
sys.path.insert(0,str(W));sys.path.append(str(B))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def dump(p,q):p.write_text(json.dumps(q,indent=2,allow_nan=False)+'\n')
def load(p,n):
 s=importlib.util.spec_from_file_location(n,p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
def main():
 O.mkdir(exist_ok=False);start=time.time()
 import torch
 import torch.nn.functional as F
 assert torch.__version__=='2.1.0+cu121' and torch.cuda.get_device_name()=='NVIDIA GeForce RTX 4090 D'
 torch.set_num_threads(4);torch.backends.cudnn.allow_tf32=True;torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.benchmark=False
 records=json.loads((B/'results/val_event_counts_20261010_v1/frozen_assets.json').read_text())
 adapter=load(S/'server_efficiency_same_output_adapter_v1.py','half_cost_adapter');fpn=load(W/'src/models/yolo26_fpn_reference.py','src.models.yolo26_fpn_reference')
 dump(O/'protocol_frozen.json',dict(units=72,seeds=[2026,2027,2028],sensors=['S1','S2'],batches=[1,16],precision=['FP32','FP16_autocast'],warmup=20,blocks=3,repeats=50,fixture='zeros float32; autocast float16 for half case',output='128x128 logits with native interpolation included',dataset_access=False,test_access=False,training=False,accuracy_claim=False,flops_scope='conv2d/convtranspose2d/linear MACs x2 plus profiler mm/bmm/addmm FLOPs; excludes BN,activation,pooling,addition,softmax,bilinear interpolation; major-operator count not total FLOPs',flops_batch=1,flops_seed=2026))
 dump(O/'environment.json',dict(torch=torch.__version__,GPU=subprocess.check_output(['nvidia-smi','--query-gpu=name,uuid,driver_version','--format=csv,noheader'],text=True),script_sha256=sha(Path(__file__))))
 units=[];invalid=[]
 for q in records:
  assert sha(Path(q['checkpoint']))==q['checkpoint_sha256']
  m=fpn.Yolo26FPNReference(common_channels=16) if q['system']=='fpn' else adapter.construct(q['unit'],q['registration']['constructor_config'])
  ck=torch.load(q['checkpoint'],map_location='cpu',weights_only=False);key=q['registration']['checkpoint_state_key'] if q['registration'] else 'model_state_dict';m.load_state_dict(ck[key],strict=True);del ck;m=m.eval().cuda()
  def forward(x,sensor):
   z=m.forward_native(x,sensor);return F.interpolate(z,size=(128,128),mode='bilinear',align_corners=False) if z.shape[-2:]!=(128,128) else z
  for sensor,c in [('S1',2),('S2',9)]:
   if False:
    x=torch.zeros(1,c,128,128,device='cuda');macs=[0];module_counts=[]
    def hook(mod,inputs,out):
     if isinstance(mod,torch.nn.ConvTranspose2d):n=inputs[0].numel()*mod.out_channels//mod.groups*mod.kernel_size[0]*mod.kernel_size[1]
     elif isinstance(mod,torch.nn.Conv2d):n=out.numel()*mod.in_channels//mod.groups*mod.kernel_size[0]*mod.kernel_size[1]
     else:n=out.numel()*mod.in_features
     macs[0]+=n;module_counts.append(dict(type=type(mod).__name__,MACs=n))
    hooks=[mod.register_forward_hook(hook) for mod in m.modules() if isinstance(mod,(torch.nn.Conv2d,torch.nn.ConvTranspose2d,torch.nn.Linear))]
    with torch.inference_mode():z=forward(x,sensor);del z
    for h in hooks:h.remove()
    with torch.inference_mode(),torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA],record_shapes=True,with_flops=True) as prof:
     z=forward(x,sensor);torch.cuda.synchronize();del z
    ops=[dict(name=e.key,flops=e.flops or 0,calls=e.count) for e in prof.key_averages()]
    matrix=sum(e['flops'] for e in ops if e['name'] in ['aten::mm','aten::bmm','aten::addmm'])
    dump(O/f"flops_{q['system']}_{sensor}.json",dict(system=q['system'],sensor=sensor,conv_linear_MACs=macs[0],matrix_FLOPs=matrix,major_operator_GFLOPs=(2*macs[0]+matrix)/1e9,module_counts=module_counts,profiler_ops=ops,scope='major operators only; exclusions in protocol; no double count of conv profiler values'))
    del x
   for batch in [1,16]:
    x=torch.zeros(batch,c,128,128,device='cuda')
    with torch.inference_mode():reference=forward(x,sensor).float()
    for precision in ['FP32','FP16_autocast']:
     unit_id=f"{q['system']}_{q['seed']}_{sensor}_b{batch}_{precision}"
     if (OLD/(unit_id+'.json')).exists():
      units.append(json.loads((OLD/(unit_id+'.json')).read_text()));continue
     enabled=precision!='FP32';gc.collect();torch.cuda.empty_cache()
     with torch.inference_mode(),torch.autocast('cuda',dtype=torch.float16,enabled=enabled):
      z=forward(x,sensor);assert z.shape==(batch,2,128,128)
      known_failed=unit_id=='fpn_2028_S1_b1_FP16_autocast'
      finite=bool(torch.isfinite(z).all())
      if not finite or known_failed:
       failure=dict(unit=unit_id,system=q['system'],seed=q['seed'],sensor=sensor,batch=batch,precision=precision,status='INVALID_NUMERICS',nan_count=int(torch.isnan(z).sum()),inf_count=int(torch.isinf(z).sum()),replay_finite=finite,first_session_failure=known_failed,FP32_max_abs=float(reference.abs().max()),scope='same zero fixture; no timing accepted')
       dump(O/(unit_id+'_invalid.json'),failure);invalid.append(failure);del z;continue
      delta=(z.float()-reference).abs();diagnostic=dict(max_abs=float(delta.max()),relative_to_max_reference=float(delta.max()/reference.abs().max().clamp_min(1e-8)))
      del z,delta
      for _ in range(20):z=forward(x,sensor);del z
      torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();blocks=[]
      for b in range(3):
       starts=[torch.cuda.Event(enable_timing=True) for _ in range(50)];ends=[torch.cuda.Event(enable_timing=True) for _ in range(50)]
       for i in range(50):starts[i].record();z=forward(x,sensor);ends[i].record();del z
       torch.cuda.synchronize();blocks.append([a.elapsed_time(b) for a,b in zip(starts,ends)])
     raw=sum(blocks,[]);unit=dict(system=q['system'],seed=q['seed'],sensor=sensor,batch=batch,precision=precision,mean_ms=statistics.mean(raw),blocks_raw_ms=blocks,peak_allocated_MiB=torch.cuda.max_memory_allocated()/1048576,zero_fixture_logit_diagnostic=diagnostic,reference_tensor_in_memory=True)
     dump(O/f"{q['system']}_{q['seed']}_{sensor}_b{batch}_{precision}.json",unit);units.append(unit)
     dump(O/'progress.json',dict(units_done=len(units),units_total=72,elapsed_seconds=time.time()-start));print('FP16_COST',len(units),72,flush=True)
    del x,reference
  del m;gc.collect();torch.cuda.empty_cache();assert sha(Path(q['checkpoint']))==q['checkpoint_sha256']
 summary={}
 for system in ['main_only','decoder','fpn']:
  for sensor in ['S1','S2']:
   for batch in [1,16]:
    for precision in ['FP32','FP16_autocast']:
     rs=[u for u in units if (u['system'],u['sensor'],u['batch'],u['precision'])==(system,sensor,batch,precision)];assert 0<=len(rs)<=3
     a=[u['mean_ms'] for u in rs];summary[f'{system}/{sensor}/b{batch}/{precision}']=dict(mean_ms=statistics.mean(a) if a else None,weight_sample_SD_ms=statistics.stdev(a) if len(a)>1 else None,weight_means_ms=a,valid_weights=len(a),scope='first session measurements retained; remaining FP32 units continued in separate session; invalid FP16 weights excluded explicitly')
 dump(O/'invalid_units.json',invalid);dump(O/'summary.json',summary);dump(O/'completion.json',dict(status='COMPLETE_WITH_INVALID_FP16_UNITS',units_accounted=len(units)+len(invalid),valid_units=len(units),invalid_units=len(invalid),raw_times=150*len(units),flops_units=6,elapsed_seconds=time.time()-start,training=False,dataset_access=False,test_access=False))
if __name__=='__main__':
 try:main()
 except Exception:
  if O.exists():dump(O/'failure.json',dict(traceback=traceback.format_exc()))
  raise
