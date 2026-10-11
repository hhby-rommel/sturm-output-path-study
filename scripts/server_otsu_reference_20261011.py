# Copyright (C) 2026 STURM output-path study contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
# Source-release notice added 2026-10-11; numerical code unchanged.
"""Classical S1 anchor: predeclared val selection then one new-baseline test claim."""
from pathlib import Path
import sys,json,time,hashlib,statistics,traceback
import numpy as np
B=Path(__file__).resolve().parents[1];W=B/'experiments/followup_server_20261008_v1/workspace';O=B/'results/otsu_reference_20261011_v1'
sys.path.insert(0,str(W))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def dump(p,q):
 t=p.with_suffix('.tmp');t.write_text(json.dumps(q,indent=2,allow_nan=False));t.replace(p)
def score(c):
 tp,fp,fn,tn=map(int,c);gt=tp+fn;den=2*tp+fp+fn
 return dict(TP=tp,FP=fp,FN=fn,TN=tn,F1=2*tp/den if den else 0,precision=tp/(tp+fp) if tp+fp else 0,recall=tp/gt if gt else 0,signed_bias_percent=100*(fp-fn)/gt if gt else None,absolute_relative_error_percent=100*abs(fp-fn)/gt if gt else None)
def otsu(x,valid):
 values=x[valid]
 if values.size==0 or float(values.min())==float(values.max()):return np.zeros_like(valid),None
 counts,edges=np.histogram(values,bins=256,range=(float(values.min()),float(values.max())));centers=(edges[:-1]+edges[1:])/2
 w=np.cumsum(counts,dtype=np.float64);total=float(w[-1]);mu=np.cumsum(counts*centers,dtype=np.float64);den=w*(total-w)
 between=np.full(256,-np.inf);good=den>0;between[good]=(mu[-1]*w[good]-mu[good]*total)**2/den[good]
 j=int(np.argmax(between));threshold=float(edges[j+1]);return x<=threshold,threshold
def main():
 import torch
 from src.dataset.sturm_flood import SturmFloodDataset
 torch.set_num_threads(1);O.mkdir(exist_ok=False);start=time.time();methods=['band1','band2','AND','OR']
 dump(O/'protocol_frozen_before_val.json',dict(scope='S1 only, stored channel indices; dark-pixel per-tile Otsu, 256 uniform raw-value bins',selection='max val perturbed pooled F1, exact clean tie, then fixed band1/band2/AND/OR order',methods=methods,constant_or_empty='all background',threshold='first between-class-variance maximizing bin; dark side <= upper edge',histogram_domain='common valid evaluation domain; no water/background labels used to set per-tile Otsu threshold',degradation='existing sample-specific S1 stream12026; original coefficients',training=False,test_scope='one new deterministic baseline, not repeated neural model test',Bolivia=False))
 def generator(identifier,stream):return torch.Generator(device='cpu').manual_seed(int.from_bytes(hashlib.sha256(f'12026|{identifier}|{stream}'.encode()).digest()[:8],'little')%(2**63-1))
 def evaluate(split,active):
  ds=SturmFloodDataset(W/'data/manifests/manifest.csv',W/'data/splits/event_split_v1.csv',split,sensor='S1');assert len(ds)==(4315 if split=='val' else 4331)
  ds.rows.sort(key=lambda r:(r['sensor'],r['event_id'],r['tile_id'],r['image_path']))
  counts={v:{m:{} for m in active} for v in ['clean','degraded']};constant=0;streamhash=hashlib.sha256()
  for i,row in enumerate(ds.rows):
   item=ds[i];x=item['image'];valid=item['valid_mask'].bool();y=item['mask'].numpy();identifier='/'.join(row[k] for k in ['sensor','event_id','tile_id','image_path'])
   severity=.5+.5*torch.rand((),generator=generator(identifier,'severity'));noise=torch.randn(x.shape,generator=generator(identifier,'gaussian'));degraded=x*(1+.12*severity*noise)
   deletion=(torch.rand(valid.shape,generator=generator(identifier,'deletion'))<(.05+.10*severity))&valid;keep=valid&~deletion;degraded=torch.where(keep[None],degraded,torch.zeros_like(degraded))
   streamhash.update(identifier.encode()+hashlib.sha256(degraded.numpy().tobytes()+keep.numpy().tobytes()).hexdigest().encode())
   for view,z,v in [('clean',x.numpy(),valid.numpy()),('degraded',degraded.numpy(),keep.numpy())]:
    p1,t1=otsu(z[0],v);p2,t2=otsu(z[1],v);constant+=int(t1 is None)+int(t2 is None)
    pred={'band1':p1,'band2':p2,'AND':p1&p2,'OR':p1|p2};fg=y==1
    for method in active:
     p=pred[method];c=np.array([np.count_nonzero(v&p&fg),np.count_nonzero(v&p&~fg),np.count_nonzero(v&~p&fg),np.count_nonzero(v&~p&~fg)],dtype=np.int64)
     counts[view][method].setdefault(row['event_id'],np.zeros(4,dtype=np.int64));counts[view][method][row['event_id']]+=c
   if i%100==0:dump(O/'progress.json',dict(split=split,tiles_done=i+1,tiles_total=len(ds),elapsed_seconds=time.time()-start));print('OTSU',split,i+1,len(ds),flush=True)
  result={}
  for view,ms in counts.items():
   result[view]={}
   for method,es in ms.items():
    event_metrics={e:score(c) for e,c in es.items()};pooled=score(sum(es.values()));areas=[c['absolute_relative_error_percent'] for c in event_metrics.values() if c['absolute_relative_error_percent'] is not None]
    result[view][method]=dict(overall=pooled,per_event=event_metrics,event_macro_F1=statistics.mean(c['F1'] for c in event_metrics.values()),event_MARE_percent=statistics.mean(areas),events=len(es))
  return dict(metrics=result,tiles=len(ds),split=split,S1_input_stream_sha256=streamhash.hexdigest(),constant_channel_tile_views=constant)
 val=evaluate('val',methods);dump(O/'val_results.json',val)
 chosen=max(methods,key=lambda m:(val['metrics']['degraded'][m]['overall']['F1'],val['metrics']['clean'][m]['overall']['F1'],-methods.index(m)))
 dump(O/'chosen_method_frozen.json',dict(method=chosen,selection_split='val',selection_role='four predeclared classical variants; not trained seed repetitions',val_results_sha256=sha(O/'val_results.json'),protocol_sha256=sha(O/'protocol_frozen_before_val.json')))
 frozen=sha(O/'chosen_method_frozen.json')
 with (O/'test_access_claim.json').open('x') as f:json.dump(dict(scope='New S1 Otsu baseline only; one clean and one fixed-perturbed pass',chosen_method=chosen,selection_sha256=frozen,user_authorization='User20261011 permits follow-up experiments; classical baseline extension',no_neural_model_reevaluation=True),f)
 test=evaluate('test',[chosen]);dump(O/'test_result.json',test)
 # Same-domain label/count gate against saved neural baseline without new inference.
 prior=json.loads((B/'results/output_path_followup_20261010_v1/main_only_2026_reevaluation.json').read_text())['metrics']
 for view in ['clean','degraded']:
  q=test['metrics'][view][chosen]['overall'];c=prior[view]['per_sensor']['S1']['confusion_matrix']
  assert q['TP']+q['FN']==c[1][1]+c[1][0]
  assert sum(q[k] for k in ['TP','FP','FN','TN'])==sum(sum(r) for r in c)
 assert sha(O/'chosen_method_frozen.json')==frozen
 dump(O/'completion.json',dict(status='CLASSICAL_S1_OTSU_VAL_SELECTION_TEST_ONCE_COMPLETE',chosen_method=chosen,selection_sha256=frozen,elapsed_seconds=time.time()-start,training=False,neural_model_test_reevaluation=False,test_domain_matches_saved_core=True))
if __name__=='__main__':
 try:main()
 except Exception:
  if O.exists():dump(O/'failure.json',dict(traceback=traceback.format_exc(),no_retry_test=True))
  raise
