# Copyright (C) 2026 STURM output-path study contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
# Source-release notice added 2026-10-11; numerical code unchanged.
"""Frozen val numeric audit, independent fvcore counts and size/batch runtime. No test."""
from pathlib import Path
import sys,json,time,statistics,gc,hashlib,traceback,importlib.util,subprocess
B=Path(__file__).resolve().parents[1];W=B/'experiments/followup_server_20261008_v1/workspace';S=B/'experiments/same_output_efficiency_20261004_v1';O=B/'results/quick_diagnostics_20261011_v1'
sys.path.insert(0,str(W));sys.path.append(str(B))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def dump(p,q):p.write_text(json.dumps(q,indent=2,allow_nan=False)+'\n')
def load(p,n):
 s=importlib.util.spec_from_file_location(n,p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
def main():
 O.mkdir(exist_ok=False);start=time.time()
 import torch
 import torch.nn.functional as F
 from src.dataset.sturm_flood import SturmFloodDataset
 assert torch.__version__=='2.1.0+cu121' and torch.cuda.get_device_name()=='NVIDIA GeForce RTX 4090 D'
 torch.set_num_threads(4);torch.backends.cudnn.allow_tf32=True;torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.benchmark=False
 records=json.loads((B/'results/val_event_counts_20261010_v1/frozen_assets.json').read_text())
 adapter=load(S/'server_efficiency_same_output_adapter_v1.py','diag_cost_adapter');fpn=load(W/'src/models/yolo26_fpn_reference.py','src.models.yolo26_fpn_reference')
 def build(q):
  assert sha(Path(q['checkpoint']))==q['checkpoint_sha256']
  m=fpn.Yolo26FPNReference(common_channels=16) if q['system']=='fpn' else adapter.construct(q['unit'],q['registration']['constructor_config'])
  ck=torch.load(q['checkpoint'],map_location='cpu',weights_only=False);key=q['registration']['checkpoint_state_key'] if q['registration'] else 'model_state_dict';m.load_state_dict(ck[key],strict=True);del ck;return m.eval().cuda()
 def forward(m,x,sensor):
  z=m.forward_native(x,sensor);return F.interpolate(z,size=x.shape[-2:],mode='bilinear',align_corners=False) if z.shape[-2:]!=x.shape[-2:] else z
 ds=SturmFloodDataset(W/'data/manifests/manifest.csv',W/'data/splits/event_split_v1.csv','val');assert len(ds)==4853
 ds.rows.sort(key=lambda r:(r['sensor'],r['event_id'],r['tile_id'],r['image_path']))
 fixtures=[];chosen=[]
 for sensor,c in [('S1',2),('S2',9)]:
  fixtures.append((sensor,'zero',torch.zeros(1,c,128,128),torch.ones(128,128,dtype=torch.bool)))
  seen=set()
  for i,r in enumerate(ds.rows):
   if r['sensor']==sensor and r['event_id'] not in seen:
    item=ds[i];identifier='/'.join(r[k] for k in ['sensor','event_id','tile_id','image_path'])
    fixtures.append((sensor,identifier,item['image'][None],item['valid_mask']));chosen.append(dict(sensor=sensor,event=r['event_id'],tile=r['tile_id'],identifier=identifier,input_sha256=hashlib.sha256(item['image'].numpy().tobytes()).hexdigest()))
    seen.add(r['event_id'])
    if len(seen)==4:break
 sizes=[(128,1),(128,16),(128,32),(128,64),(256,1),(256,8),(512,1),(512,4)]
 dump(O/'protocol_frozen.json',dict(training=False,test_access=False,checkpoint_reselection=False,val_sample_rule='first sorted tile from first four distinct events per sensor; no prediction-based selection',val_fixture_count=8,zero_fixture_count=2,precisions=['FP32','FP16_autocast','BF16_autocast'],weights=9,size_batch_plan=sizes,runtime_precision='FP32',runtime_warmup=10,runtime_blocks=3,runtime_repeats_per_block=20,runtime_fixture='zeros; logits resized to input HxW; no stitched scenes or accuracy inference',flops='fvcore independent JIT operator counts; MAC convention multiplied2, retain conv/linear/matmul/bmm only for common comparison; other operators separately recorded',fvcore_dependency_scope='isolated target directory; training environment packages unchanged'))
 dump(O/'val_sample_manifest.json',chosen);dump(O/'frozen_assets.json',records)
 dump(O/'environment.json',dict(torch=torch.__version__,GPU=subprocess.check_output(['nvidia-smi','--query-gpu=name,uuid,driver_version','--format=csv,noheader'],text=True),script_sha256=sha(Path(__file__)),BF16_supported=torch.cuda.is_bf16_supported()))
 numerics=[]
 def tensors(x):
  if isinstance(x,torch.Tensor):return [x]
  if isinstance(x,(list,tuple)):return sum([tensors(y) for y in x],[])
  if isinstance(x,dict):return sum([tensors(y) for y in x.values()],[])
  return []
 for q in records:
  m=build(q)
  for sensor,identifier,cpu_x,valid in fixtures:
   x=cpu_x.cuda()
   with torch.inference_mode():ref=forward(m,x,sensor).float()
   assert torch.isfinite(ref).all(),('FP32 failure',q['system'],q['seed'],identifier)
   for precision,dtype in [('FP32',torch.float32),('FP16_autocast',torch.float16),('BF16_autocast',torch.bfloat16)]:
    with torch.inference_mode(),torch.autocast('cuda',dtype=dtype if precision!='FP32' else torch.float16,enabled=precision!='FP32'):z=forward(m,x,sensor)
    finite=bool(torch.isfinite(z).all());row=dict(system=q['system'],seed=q['seed'],sensor=sensor,fixture=identifier,precision=precision,finite=finite,nan_count=int(torch.isnan(z).sum()),inf_count=int(torch.isinf(z).sum()),valid_domain_nonfinite=int((~torch.isfinite(z))[0,:,valid.cuda()].sum()),FP32_max_abs=float(ref.abs().max()))
    if finite:row['max_abs_vs_FP32']=float((z.float()-ref).abs().max())
    else:
     found=[]
     def observer(name,mod,inp,out):
      ts=tensors(out)
      if found or not any(t.is_floating_point() and not bool(torch.isfinite(t).all()) for t in ts):return
      ins=tensors(inp);item=dict(name=name,module=type(mod).__name__,input_finite=all(not t.is_floating_point() or bool(torch.isfinite(t).all()) for t in ins),output_shapes=[list(t.shape) for t in ts],input_max_abs=[float(t.float().abs().max()) if t.numel() and torch.isfinite(t).all() else None for t in ins])
      if isinstance(mod,torch.nn.BatchNorm2d):item['running_var_min']=float(mod.running_var.min());item['eps']=mod.eps
      found.append(item)
     hooks=[mod.register_forward_hook(lambda mod,inp,out,n=name:observer(n,mod,inp,out)) for name,mod in m.named_modules()]
     with torch.inference_mode(),torch.autocast('cuda',dtype=dtype):probe=forward(m,x,sensor)
     for h in hooks:h.remove()
     row['first_nonfinite_module']=found[0] if found else None;row['hook_replay_nonfinite']=not bool(torch.isfinite(probe).all());del probe
    numerics.append(row);del z
   del x,ref
  dump(O/'numerics.json',numerics);print('NUMERIC_MODEL',q['system'],q['seed'],flush=True);del m;gc.collect();torch.cuda.empty_cache()
 dump(O/'numeric_completion.json',dict(status='COMPLETE',units=len(numerics),elapsed_seconds=time.time()-start,test_access=False))
 # External counter uses a separate dependency path, retaining version and unsupported ops.
 sys.path.insert(0,str(B/'experiments/fvcore_crosscheck_20261011_v1/deps'))
 from fvcore.nn import FlopCountAnalysis
 import importlib.metadata
 dump(O/'counter_versions.json',{name:importlib.metadata.version(name) for name in ['fvcore','iopath','yacs']})
 cross=[]
 for q in [q for q in records if q['seed']==2026]:
  m=build(q)
  for sensor,c in [('S1',2),('S2',9)]:
   class Wrapped(torch.nn.Module):
    def __init__(self,inner):super().__init__();self.inner=inner
    def forward(self,x):return forward(self.inner,x,sensor)
   x=torch.zeros(1,c,128,128,device='cuda');wrapper=Wrapped(m)
   with torch.inference_mode():analysis=FlopCountAnalysis(wrapper,(x,));counts=dict(analysis.by_operator())
   allowed=['conv','linear','matmul','bmm','mm','addmm'];macs=sum(counts.get(k,0) for k in allowed)
   original=json.loads((B/'results/fp16_flops_20261011_v1'/f"flops_{q['system']}_{sensor}.json").read_text())
   row=dict(system=q['system'],sensor=sensor,fvcore_operator_MACs=counts,major_operator_GFLOPs=2*macs/1e9,previous_major_operator_GFLOPs=original['major_operator_GFLOPs'],relative_difference=(2*macs/1e9-original['major_operator_GFLOPs'])/original['major_operator_GFLOPs'],unsupported_ops=dict(analysis.unsupported_ops()),scope='1 fvcore multiply-add counted as1 MAC; normalize2; BN and other handled operators excluded from common total')
   cross.append(row);dump(O/f"fvcore_{q['system']}_{sensor}.json",row);print('FVCORE',q['system'],sensor,row['relative_difference'],flush=True);del x,wrapper
  del m;gc.collect();torch.cuda.empty_cache()
 dump(O/'fvcore_crosscheck.json',cross)
 costs=[]
 for q in records:
  m=build(q)
  for sensor,c in [('S1',2),('S2',9)]:
   for size,batch in sizes:
    unit=dict(system=q['system'],seed=q['seed'],sensor=sensor,size=size,batch=batch)
    try:
     gc.collect();torch.cuda.empty_cache();x=torch.zeros(batch,c,size,size,device='cuda')
     with torch.inference_mode():
      z=forward(m,x,sensor);assert z.shape==(batch,2,size,size) and torch.isfinite(z).all();del z
      for _ in range(10):z=forward(m,x,sensor);del z
      torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();blocks=[]
      for block in range(3):
       starts=[torch.cuda.Event(enable_timing=True) for _ in range(20)];ends=[torch.cuda.Event(enable_timing=True) for _ in range(20)]
       for i in range(20):starts[i].record();z=forward(m,x,sensor);ends[i].record();del z
       torch.cuda.synchronize();blocks.append([a.elapsed_time(b) for a,b in zip(starts,ends)])
     unit.update(status='COMPLETE',mean_ms=statistics.mean(sum(blocks,[])),blocks_raw_ms=blocks,peak_allocated_MiB=torch.cuda.max_memory_allocated()/1048576);del x,starts,ends
    except (RuntimeError,AssertionError) as e:
     unit.update(status='FAILED',type=type(e).__name__,message=str(e));gc.collect();torch.cuda.empty_cache()
    costs.append(unit);dump(O/f"cost_{q['system']}_{q['seed']}_{sensor}_{size}_b{batch}.json",unit)
    dump(O/'progress.json',dict(stage='size_batch_cost',units_done=len(costs),units_total=144,elapsed_seconds=time.time()-start));print('SIZE_COST',len(costs),144,flush=True)
  del m;gc.collect();torch.cuda.empty_cache();assert sha(Path(q['checkpoint']))==q['checkpoint_sha256']
 dump(O/'size_batch_costs.json',costs);dump(O/'completion.json',dict(status='COMPLETE',numeric_units=len(numerics),fvcore_units=len(cross),runtime_units=len(costs),runtime_failed=sum(q['status']!='COMPLETE' for q in costs),elapsed_seconds=time.time()-start,training=False,test_access=False))
if __name__=='__main__':
 try:main()
 except Exception:
  if O.exists():dump(O/'failure.json',dict(traceback=traceback.format_exc()))
  raise
