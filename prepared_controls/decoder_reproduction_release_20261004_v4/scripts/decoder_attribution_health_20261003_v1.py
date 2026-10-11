# Copyright (C) 2026 STURM output-path study contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
# Source-release notice added 2026-10-11; numerical code unchanged.
"""All-arm fixture health, distinct local CPU and server GPU receipts."""
from pathlib import Path
import argparse,sys,json,hashlib,subprocess
R=Path(__file__).resolve().parents[1]
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 p=argparse.ArgumentParser();p.add_argument('--project-root',type=Path,required=True);p.add_argument('--lock',type=Path,required=True);p.add_argument('--output-root',type=Path,required=True);p.add_argument('--device',choices=('cpu','cuda'),required=True);a=p.parse_args();out=a.output_root.resolve();out.mkdir(parents=True,exist_ok=False);rows=[];expected_stream=None
 for arm in ('full','no_skip_capacity','no_input','no_p1'):
  dest=out/arm;cmd=[sys.executable,'-u',str(R/'scripts/decoder_attribution_train_val_20261003_v1.py'),'--phase','health','--arm',arm,'--seed','10501','--project-root',str(a.project_root.resolve()),'--output-dir',str(dest),'--lock',str(a.lock.resolve()),'--device',a.device]
  with (out/(arm+'.log')).open('x') as f:proc=subprocess.run(cmd,cwd=R,stdout=f,stderr=subprocess.STDOUT)
  if proc.returncode:
   (out/'failure.json').write_text(json.dumps(dict(arm=arm,returncode=proc.returncode,stop_on_first_failure=True)));raise RuntimeError('health arm failed: '+arm)
  row=read(dest/'completion.json');probe=read(dest/'saved_restore_probe.json');streams=read(dest/'history.json')[0]['validation']['evaluation_input_stream_sha256']
  if expected_stream is None:expected_stream=streams
  if expected_stream!=streams or not probe['model_optimizer_tolerance_pass'] or row['test_enabled']:raise RuntimeError('health pairing/recovery guard failed')
  rows.append(row);print('HEALTH_ARM_PASSED',arm,flush=True)
 result=dict(status='SERVER_GPU_ALL_FOUR_ARM_HEALTH_PASSED' if a.device=='cuda' else 'INDEPENDENT_LOCAL_CPU_ALL_FOUR_ARM_HEALTH_PASSED',lock_sha256=sha(a.lock),arms=rows,paired_input_streams=True,fixture_only=True,full_val=False,test_enabled=False,formal_training_started=False)
 (out/'all_arm_receipt.json').write_text(json.dumps(result,indent=2)+'\n')
 print(result['status'],flush=True)
if __name__=='__main__':main()
