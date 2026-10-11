# Copyright (C) 2026 STURM output-path study contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
# Source-release notice added 2026-10-11; numerical code unchanged.
"""Nine train/val-only tasks, fail-stop, new directories, no auto-resume."""
from pathlib import Path
import argparse,json,subprocess,sys,hashlib,os,time
R=Path(__file__).resolve().parents[1]
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def main():
 p=argparse.ArgumentParser();p.add_argument('--project-root',type=Path,required=True);p.add_argument('--output-root',type=Path,required=True);p.add_argument('--lock',type=Path,required=True);p.add_argument('--approval',type=Path,required=True);p.add_argument('--health-receipt',type=Path,required=True);a=p.parse_args()
 lock=read(a.lock);approval=read(a.approval);receipt=read(a.health_receipt);digest=sha(a.lock)
 tasks=[dict(arm=arm,seed=seed) for seed in (2026,2027,2028) for arm in ('no_skip_capacity','no_input','no_p1')]
 if approval.get('approved') is not True or approval.get('action')!='train_val' or approval.get('lock_sha256')!=digest or approval.get('tasks')!=tasks:raise PermissionError('exact nine-task authorization missing')
 if receipt.get('status')!='SERVER_GPU_ALL_FOUR_ARM_HEALTH_PASSED' or receipt.get('lock_sha256')!=digest:raise PermissionError('server GPU receipt missing')
 out=a.output_root.resolve();out.mkdir(parents=True,exist_ok=False)
 with (out/'serial_claim.json').open('x') as f:json.dump(dict(pid=os.getpid(),time=time.time(),lock_sha256=digest),f)
 for task in tasks:
  target=out/task['arm']/str(task['seed']);cmd=[sys.executable,'-u',str(R/'scripts/decoder_attribution_train_val_20261003_v1.py'),'--phase','formal','--arm',task['arm'],'--seed',str(task['seed']),'--project-root',str(a.project_root.resolve()),'--output-dir',str(target),'--lock',str(a.lock.resolve()),'--approval',str(a.approval.resolve()),'--health-receipt',str(a.health_receipt.resolve()),'--device','cuda']
  with (out/f"{task['seed']}_{task['arm']}.log").open('x') as f:code=subprocess.call(cmd,cwd=R,stdout=f,stderr=subprocess.STDOUT)
  if code:
   (out/'serial_failure.json').write_text(json.dumps(dict(task=task,returncode=code,no_retry=True,no_test=True),indent=2));raise RuntimeError('first failed task; queue stopped')
  result=read(target/'completion.json')
  if result.get('status')!='FORMAL_TRAIN_VAL_COMPLETED_TEST_PENDING' or result.get('lock_sha256')!=digest or result.get('test_enabled') is not False:raise RuntimeError('completed task contract mismatch; queue stopped')
 (out/'serial_completion.json').write_text(json.dumps(dict(status='NINE_TRAIN_VAL_TASKS_COMPLETED_TEST_PENDING',tasks=tasks,no_automatic_test=True),indent=2))
if __name__=='__main__':main()
