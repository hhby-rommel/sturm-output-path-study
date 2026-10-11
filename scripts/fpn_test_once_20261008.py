# Copyright (C) 2026 STURM output-path study contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
# Source-release notice added 2026-10-11; numerical code unchanged.
"""One claimed full STURM test of a successful FPN val-best; no training."""
from pathlib import Path
import argparse,hashlib,json,sys,time
R=Path(__file__).resolve().parents[1]
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def write(p,v):
 tmp=p.with_suffix(p.suffix+'.tmp');assert not tmp.exists();tmp.write_text(json.dumps(v,indent=2,allow_nan=False)+'\n');tmp.replace(p)
def main():
 p=argparse.ArgumentParser();p.add_argument('--seed',type=int,choices=[2026,2027,2028],required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 authpath=R/'prepared_controls/followup_one_shot_20261008_v1/test_authorization.json';auth=json.loads(authpath.read_text());out=a.output.resolve()
 assert auth['approved'] and auth['entries']['fpn']==sha(__file__) and a.seed in auth['fpn_seeds']
 assert out.is_relative_to((R/'results').resolve())
 assert not any((out/n).exists() for n in ['test_access_claim.json','test_evaluation.json','test_failure.json','failure.json']),'Already claimed/failed; no repeated test'
 done=json.loads((out/'completion.json').read_text());assert done['status']=='FPN_TRAIN_VAL_COMPLETE_NO_TEST' and done['identity']['seed']==a.seed
 best=out/'best.pt';bestsha=sha(best);assert bestsha==done['best_sha256']
 scope=auth['sturm_scope'];assert sha(R/'data/manifests/manifest.csv')==scope['manifest_sha256'] and sha(R/'data/splits/event_split_v1.csv')==scope['split_sha256']
 from fpn_reference_task_20261008 import build
 import torch
 from scripts.server_main import make_loader
 from src.dataset.sturm_flood import SturmFloodDataset
 from src.training.masked_segmentation import MaskedSemanticLoss
 from src.evaluation.formal import evaluate_clean_and_degraded
 assert torch.__version__=='2.1.0+cu121' and torch.cuda.get_device_name()=='NVIDIA GeForce RTX 4090 D'
 torch.set_num_threads(4);torch.backends.cudnn.allow_tf32=True;torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=False;torch.use_deterministic_algorithms(False)
 identity=dict(seed=a.seed,selection=done['best'],checkpoint_sha256=bestsha,authorization_sha256=sha(authpath),test_entry_sha256=sha(__file__),training_completion_sha256=sha(out/'completion.json'),fixed_tag=1,test_loader_seed=a.seed+3,once=True,scope=scope)
 with (out/'test_access_claim.json').open('x') as f:json.dump(identity,f,indent=2)
 try:
  start=time.time();model,_=build(a.seed,torch.device('cuda'));ck=torch.load(best,map_location='cpu',weights_only=False)
  assert ck['seed']==a.seed and ck['selection']==done['best'] and ck['epoch']==done['best']['epoch']
  model.load_state_dict(ck['model_state_dict'],strict=True);del ck
  ds=SturmFloodDataset(R/'data/manifests/manifest.csv',R/'data/splits/event_split_v1.csv','test');assert len(ds)==4863 and len({q['event_id'] for q in ds.rows})==12
  loader=make_loader(ds,batch_size=16,train=False,seed=a.seed+3)
  class Observer(torch.nn.Module):
   def __init__(self,m):super().__init__();self.model=m;self.hashes={};self.batches={}
   def forward(self,batch):
    sensor=batch['sensor'];sensor=sensor[0] if isinstance(sensor,list) else sensor;key=str(sensor)
    h=self.hashes.setdefault(key,hashlib.sha256());h.update(batch['image'].detach().cpu().contiguous().numpy().tobytes());h.update(batch['valid_mask'].detach().cpu().contiguous().numpy().tobytes())
    self.batches[key]=self.batches.get(key,0)+1
    return self.model(batch)
   def forward_native(self,x,sensor):
    key=str(sensor);h=self.hashes.setdefault(key,hashlib.sha256());h.update(x.detach().cpu().contiguous().numpy().tobytes());self.batches[key]=self.batches.get(key,0)+1
    return self.model.forward_native(x,sensor)
  observed=Observer(model)
  metrics=evaluate_clean_and_degraded(observed,loader,MaskedSemanticLoss(aux_weight=.4).cuda(),torch.device('cuda'),seed=a.seed,tag=1)
  assert sha(best)==bestsha
  write(out/'test_evaluation.json',dict(**identity,status='FPN_ONE_SHOT_TEST_COMPLETE',metrics=metrics,test_tiles=len(ds),test_events=12,input_stream_sha256={k:v.hexdigest() for k,v in observed.hashes.items()},batches=observed.batches,elapsed_seconds=time.time()-start,test_access=True))
 except Exception as e:write(out/'test_failure.json',dict(type=type(e).__name__,message=str(e),claim_exists=True,no_retry=True));raise
if __name__=='__main__':main()
