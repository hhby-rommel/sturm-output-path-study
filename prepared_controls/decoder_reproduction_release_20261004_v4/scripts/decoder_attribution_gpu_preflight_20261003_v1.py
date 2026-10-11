# Copyright (C) 2026 STURM output-path study contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
# Source-release notice added 2026-10-11; numerical code unchanged.
"""Server GPU fixture chain plus full-val closure; no test and no formal launch."""
from pathlib import Path
import argparse,sys,json,subprocess
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R))
import torch
from scripts import decoder_attribution_train_val_20261003_v1 as runner
from src.training.masked_segmentation import MaskedSemanticLoss
from src.evaluation.formal import evaluate_clean_and_degraded
def main():
 p=argparse.ArgumentParser();p.add_argument('--project-root',type=Path,required=True);p.add_argument('--lock',type=Path,required=True);p.add_argument('--output-root',type=Path,required=True);a=p.parse_args()
 if not torch.cuda.is_available():raise RuntimeError('GPU unavailable; no CPU substitution')
 root=a.project_root.resolve();lock=runner.read(a.lock);runner.verify(root,lock)
 index=root/'raw_train_val_identity.json'
 if runner.sha(index)!=lock['full_raw_train_val_index_sha256']:raise RuntimeError('raw index changed')
 for item in runner.read(index):
  if runner.sha(root/item['path'])!=item['sha256']:raise RuntimeError('raw train/val bytes changed')
 out=a.output_root.resolve();out.mkdir(parents=True,exist_ok=False)
 cmd=[sys.executable,'-I',str(R/'scripts/decoder_attribution_health_20261003_v1.py'),'--project-root',str(root),'--lock',str(a.lock.resolve()),'--output-root',str(out/'fixture_health'),'--device','cuda']
 subprocess.run(cmd,cwd=R,check=True)
 receipt=runner.read(out/'fixture_health/all_arm_receipt.json');streams=None;torch.set_num_threads(4)
 for arm in ('full','no_skip_capacity','no_input','no_p1'):
  model,_=runner.build(arm,10501,root,torch.device('cuda'));ck=torch.load(out/'fixture_health'/arm/'best.pt',map_location='cpu');model.load_state_dict(ck['model_state_dict'],strict=True)
  _,val=runner.loader_pair(root,10501,None);assert len(val.dataset)==4853
  observer=runner.EvaluationStream(model);metrics=evaluate_clean_and_degraded(observer,val,MaskedSemanticLoss(aux_weight=.4).cuda(),torch.device('cuda'),seed=10501,tag=1)
  if streams is None:streams=observer.summaries()
  if streams!=observer.summaries():raise RuntimeError('full-val tensor pairing drift')
  runner.write(out/(arm+'_full_val.json'),dict(metrics=metrics,input_stream_sha256=observer.summaries(),diagnostic_only=True))
  del model,ck,observer;torch.cuda.empty_cache();print('GPU_FULL_VAL_PASSED',arm,flush=True)
 runner.verify(root,lock);receipt.update(full_val=True,val_tiles=4853,GPU=torch.cuda.get_device_name(),torch=torch.__version__,cuda=torch.version.cuda,
  GPU_full_val_scope='original full validation, no learning/performance threshold, no test',raw_train_val_checked=True)
 runner.write(out/'server_gpu_admission_receipt.json',receipt)
 print('GPU_COMPATIBILITY_AND_FULL_VAL_COMPLETE_FORMAL_NOT_STARTED',flush=True)
if __name__=='__main__':main()
