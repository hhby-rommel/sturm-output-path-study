# Copyright (C) 2026 STURM output-path study contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
# Source-release notice added 2026-10-11; numerical code unchanged.
"""Frozen decoder attribution train/val entry. No test code or automatic resume."""
from pathlib import Path
import argparse,sys,json,hashlib,time,copy,platform
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import torch
from scripts.server_main import make_loader,RowView,save_checkpoint,_restore_rng_state
from src.dataset.sturm_flood import SturmFloodDataset
from src.models.decoder_attribution import DecoderAttribution,ARMS,decoder_parameter_count
from src.training.single_sensor_decoder_control import build_original_decoder,LocalSelectionState
from src.training.engine import train_one_epoch
from src.training.masked_segmentation import MaskedSemanticLoss
from src.training.independent_decoder_preflight import compare_states,rng_state
from src.evaluation.formal import seed_everything,fixed_rng,evaluate_clean_and_degraded

def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def write(p,v):
 p=Path(p);t=p.with_suffix(p.suffix+'.tmp');t.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n',encoding='utf-8');t.replace(p)
def verify(root,lock):
 for rel,digest in lock['source_files'].items():
  p=(ROOT/rel).resolve()
  if not p.is_relative_to(ROOT.resolve()) or sha(p)!=digest:raise RuntimeError('source drift: '+rel)
 for rel,digest in lock['assets'].items():
  p=(root/rel).resolve()
  if not p.is_relative_to(root.resolve()) or sha(p)!=digest:raise RuntimeError('asset drift: '+rel)
 cfg=lock['protocol']
 if cfg!=dict(lr=.0003,weight_decay=.0001,batch=16,clip=5,max_epochs=50,patience=10,aux_weight=.4,tag=1,scheduler=None,freeze_BN=True):raise RuntimeError('numeric protocol altered')
def build(arm,seed,root,device):
 seed_everything(seed);base,report=build_original_decoder(root/'models/pretrained/yolo26n-sem.pt')
 with fixed_rng(seed,42):m=DecoderAttribution(base,arm)
 return m.to(device),report
def loader_pair(root,seed,fixture=None):
 sets={s:SturmFloodDataset(root/'data/manifests/manifest.csv',root/'data/splits/event_split_v1.csv',s) for s in ('train','val')}
 if fixture:
  for s in sets:sets[s]=RowView(sets[s],fixture[s])
 return tuple(make_loader(sets[s],batch_size=16,train=s=='train',seed=seed+(1 if s=='train' else 2)) for s in ('train','val'))
class EvaluationStream(torch.nn.Module):
 """Input hashes only; no extra forwards, RNG calls or metric modifications."""
 def __init__(self,model):
  super().__init__();self.model=model;self.hashes={v:{k:hashlib.sha256() for k in ('image','mask','valid_mask')} for v in ('clean','degraded')}
 def forward(self,batch):
  view='degraded' if 'degradation_severity' in batch else 'clean'
  for k,h in self.hashes[view].items():h.update(batch[k].detach().cpu().contiguous().numpy().tobytes())
  return self.model(batch)
 def summaries(self):return {v:{k:h.hexdigest() for k,h in x.items()} for v,x in self.hashes.items()}
def run(args,lock):
 root=args.project_root.resolve();verify(root,lock)
 if args.phase=='audit':return {'status':'STATIC_SOURCE_ASSET_GATE_PASSED','new_training':False,'new_test':False}
 if args.phase=='formal':
  if args.arm=='full' or args.seed not in (2026,2027,2028):raise PermissionError('formal scope is three registered variants and seeds')
  approval=read(args.approval) if args.approval else {}
  if approval.get('approved') is not True or approval.get('action')!='train_val' or approval.get('lock_sha256')!=sha(args.lock) or {'arm':args.arm,'seed':args.seed} not in approval.get('tasks',[]):raise PermissionError('exact frozen train/val authorization missing')
  health=read(args.health_receipt) if args.health_receipt else {}
  if health.get('status')!='SERVER_GPU_ALL_FOUR_ARM_HEALTH_PASSED' or health.get('lock_sha256')!=sha(args.lock):raise PermissionError('server GPU compatibility receipt missing')
  if health.get('full_val') is not True or health.get('val_tiles')!=4853:raise PermissionError('full-val GPU closure receipt missing')
  if not torch.cuda.is_available():raise RuntimeError('formal stage requires checked CUDA device')
  if health.get('torch')!=torch.__version__ or health.get('GPU')!=torch.cuda.get_device_name():raise RuntimeError('health/runtime environment drift')
  identity=root/'raw_train_val_identity.json'
  if sha(identity)!=lock['full_raw_train_val_index_sha256']:raise RuntimeError('raw index drift')
  for item in read(identity):
   p=(root/item['path']).resolve()
   if not p.is_relative_to((root/'data/raw/Dataset').resolve()) or sha(p)!=item['sha256']:raise RuntimeError('raw train/val data drift: '+item['path'])
 elif not (10000<=args.seed<20000):raise ValueError('health uses non-formal seed')
 out=args.output_dir.resolve()
 if out==root or out.is_relative_to(root/'data') or out.is_relative_to(root/'models'):raise ValueError('unsafe output directory')
 out.mkdir(parents=True,exist_ok=False)
 args._claimed_output=True
 device=torch.device(args.device)
 fixture=read(root/'health_fixture.json') if args.phase=='health' else None
 if fixture:
  if sha(root/'health_fixture.json')!=lock['health_fixture_sha256']:raise RuntimeError('fixture selection drift')
  for rel,digest in lock['fixture_assets'].items():
   if sha(root/rel)!=digest:raise RuntimeError('fixture raw bytes changed')
 model,init=build(args.arm,args.seed,root,device);opt=torch.optim.AdamW(model.parameters(),lr=.0003,weight_decay=.0001);criterion=MaskedSemanticLoss(aux_weight=.4).to(device)
 train,val=loader_pair(root,args.seed,fixture)
 if not fixture and (len(train.dataset)!=14561 or len(val.dataset)!=4853 or len(train)!=911):raise RuntimeError('formal dataset/batch budget mismatch')
 selector=LocalSelectionState();history=[];start=time.monotonic();torch.set_num_threads(4)
 write(out/'metadata.json',dict(phase=args.phase,arm=args.arm,seed=args.seed,lock_sha256=sha(args.lock),protocol=lock['protocol'],initialization=init,
  train_tiles=len(train.dataset),val_tiles=len(val.dataset),test_enabled=False,automatic_resume=False,python=sys.version,platform=platform.platform(),torch=torch.__version__,device=str(device),
  registered_parameters=sum(p.numel() for p in model.parameters()),decoder_parameters=decoder_parameter_count(model.base),capacity=model.capacity_design))
 bn={n:v.detach().cpu().clone() for n,v in model.named_buffers() if 'running_' in n or 'num_batches_tracked' in n}
 for epoch in range(1,2 if fixture else 51):
  tick=time.monotonic();stats=train_one_epoch(model,train,criterion,opt,device=device,model_variant='m1_degraded',grad_clip_norm=5,freeze_batchnorm=True)
  observer=EvaluationStream(model);ev=evaluate_clean_and_degraded(observer,val,criterion,device,seed=args.seed,tag=1)
  ev['evaluation_input_stream_sha256']=observer.summaries()
  stop=selector.observe(epoch,ev);history.append(dict(epoch=epoch,training=stats,validation=ev,selection=copy.deepcopy(selector.best),stale=selector.stale,seconds=time.monotonic()-tick))
  for n,v in model.named_buffers():
   if n in bn and not torch.equal(v.cpu(),bn[n]):raise RuntimeError('BN running buffer changed: '+n)
  for name in (['last.pt','best.pt'] if selector.best['epoch']==epoch else ['last.pt']):
   save_checkpoint(out/name,model,opt,None,epoch=epoch,seed=args.seed,selection=selector.best,stale_epochs=selector.stale,history=history,max_epochs=50,patience=10,optimizer_spec_version='optimizer_v1',train_loader=train)
  write(out/'history.json',history);print('ATTRIBUTION_TRAIN_VAL',args.phase,args.arm,args.seed,epoch,ev['degraded']['overall']['flood_f1'],flush=True)
  if stop:break
 if fixture:
  ck=torch.load(out/'last.pt',map_location='cpu');results=[]
  for _ in range(2):
   recovered,_=build(args.arm,args.seed,root,device);recovered.load_state_dict(ck['model_state_dict'],strict=True);ropt=torch.optim.AdamW(recovered.parameters(),lr=.0003,weight_decay=.0001);ropt.load_state_dict(copy.deepcopy(ck['optimizer_state_dict']))
   rtrain,_=loader_pair(root,args.seed,fixture);_restore_rng_state(ck['rng_state'],rtrain)
   train_one_epoch(recovered,rtrain,criterion,ropt,device=device,model_variant='m1_degraded',grad_clip_norm=5,freeze_batchnorm=True)
   results.append((copy.deepcopy(recovered.state_dict()),copy.deepcopy(ropt.state_dict()),rtrain.sampler.generator.get_state(),rng_state()))
  for a,b in zip(*results):compare_states(a,b)
  write(out/'saved_restore_probe.json',dict(model_optimizer_tolerance_pass=True,sampler_generator_exact=True,atol=1e-6,rtol=1e-5,scope='next small fixture epoch from saved sampler/RNG; no unrestricted resume claim',extra_updates=2))
 verify(root,lock)
 result=dict(status=('SERVER_GPU_HEALTH_ARM_PASSED' if device.type=='cuda' else 'LOCAL_INDEPENDENT_CPU_FIXTURE_HEALTH_PASSED') if fixture else 'FORMAL_TRAIN_VAL_COMPLETED_TEST_PENDING',arm=args.arm,seed=args.seed,
  lock_sha256=sha(args.lock),epochs=len(history),best=selector.best,stale=selector.stale,elapsed_seconds=time.monotonic()-start,
  optimizer_updates=sum(h['training']['steps'] for h in history),exposure_budget=sum(h['training']['steps'] for h in history)*16 if fixture else len(train.dataset)*len(history),
  best_sha256=sha(out/'best.pt'),last_sha256=sha(out/'last.pt'),test_enabled=False,automatic_resume=False,
  device=str(device),GPU=torch.cuda.get_device_name() if device.type=='cuda' else None,torch=torch.__version__,cuda=torch.version.cuda)
 write(out/'completion.json',result);return result
def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--phase',choices=('audit','health','formal'),required=True);p.add_argument('--arm',choices=ARMS,required=True);p.add_argument('--seed',type=int,required=True);p.add_argument('--project-root',type=Path,required=True);p.add_argument('--output-dir',type=Path,required=True);p.add_argument('--lock',type=Path,required=True);p.add_argument('--approval',type=Path);p.add_argument('--health-receipt',type=Path);p.add_argument('--device',default='cpu');a=p.parse_args()
 if a.phase=='formal' and a.device!='cuda':p.error('formal device must be cuda')
 try:print(json.dumps(run(a,read(a.lock)),ensure_ascii=False),flush=True)
 except Exception as exc:
  if getattr(a,'_claimed_output',False):write(a.output_dir/'failure.json',dict(error=type(exc).__name__,detail=str(exc),no_retry=True,no_test=True))
  raise
if __name__=='__main__':main()
