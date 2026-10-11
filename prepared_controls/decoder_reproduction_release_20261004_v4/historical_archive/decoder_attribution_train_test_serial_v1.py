# Copyright (C) 2026 STURM output-path study contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
# Source-release notice added 2026-10-11; numerical code unchanged.
"""Adopt the live first training, then nine strictly serial train/test tasks."""
from pathlib import Path
import argparse,sys,json,os,signal,time,subprocess,hashlib
sys.path.insert(0,str(Path(__file__).resolve().parent))
from decoder_attribution_test_once_v1 import TASKS,read,sha,write,valid_existing_test

def proc(pid):
    try:
        raw=Path(f'/proc/{pid}/stat').read_text();fields=raw[raw.rfind(')')+2:].split()
        return dict(state=fields[0],starttime=fields[19],exit_status=int(fields[49]))
    except FileNotFoundError:return None

def execute(cmd,log,state_path,phase,task):
    with log.open('x') as f:
        child=subprocess.Popen(cmd,stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT)
        write(state_path,dict(status='RUNNING',phase=phase,task=task,child_pid=child.pid,no_retry=True))
        code=child.wait()
    if code:raise RuntimeError(f'{phase} {task} exit={code}; queue stopped')

def completed_training(target,task,digest):
    done=read(target/'completion.json')
    if (target/'failure.json').exists() or done['status']!='FORMAL_TRAIN_VAL_COMPLETED_TEST_PENDING' or done['arm']!=task['arm'] or done['seed']!=task['seed'] or done['lock_sha256']!=digest or done['best_sha256']!=sha(target/'best.pt'):
        raise RuntimeError('training completion contract mismatch')
    return done

def main():
    p=argparse.ArgumentParser();p.add_argument('--plan',type=Path,required=True);p.add_argument('--authorization',type=Path,required=True)
    a=p.parse_args();plan=read(a.plan);auth=read(a.authorization)
    root=Path(plan['project_root']);out=Path(plan['output_root']);lock=Path(plan['lock']);approval=Path(plan['train_approval']);health=Path(plan['health_receipt'])
    digest=sha(lock);old_pid=plan['old_controller_pid'];adopt_pid=plan['adopt_child_pid'];state=out/'train_test_status.json'
    if auth.get('approved') is not True or auth.get('action')!='one_shot_test_after_each_task' or auth['tasks']!=TASKS or auth['lock_sha256']!=digest or auth['serial_entry_sha256']!=sha(__file__) or auth['plan_sha256']!=sha(a.plan):
        raise PermissionError('train/test scope authorization mismatch')
    if sha(Path(__file__).with_name('decoder_attribution_test_once_v1.py'))!=auth['test_entry_sha256']:raise RuntimeError('test entry drift')
    previous=read(approval);receipt=read(health)
    if previous['approved'] is not True or previous['tasks']!=TASKS or previous['lock_sha256']!=digest or receipt['full_val'] is not True or receipt['lock_sha256']!=digest:raise RuntimeError('original training admission mismatch')
    with (out/'train_test_controller_claim.json').open('x') as f:json.dump(dict(pid=os.getpid(),plan_sha256=sha(a.plan),authorization_sha256=sha(a.authorization)),f)
    completed=[]
    try:
        old=proc(old_pid);child=proc(adopt_pid)
        if old is None or old['state'] not in ('T','t') or old['starttime']!=plan['old_controller_starttime']:raise RuntimeError('old controller not held/identity drift')
        if child is not None and child['starttime']!=plan['adopt_child_starttime']:raise RuntimeError('adopted process identity drift')
        write(state,dict(status='RUNNING',phase='ADOPT_UNINTERRUPTED_TRAIN',task=TASKS[0],child_pid=adopt_pid,old_controller_held=True))
        while child is not None and child['state'] not in ('Z','X'):
            time.sleep(5);child=proc(adopt_pid)
            if child is not None and child['starttime']!=plan['adopt_child_starttime']:raise RuntimeError('adopted pid reused')
        if child is not None and child['exit_status']!=0:raise RuntimeError('adopted training failed; no retry')
        completed_training(out/TASKS[0]['arm']/str(TASKS[0]['seed']),TASKS[0],digest)
        # Retire only the held old controller, after its child has finished.
        old=proc(old_pid)
        if old is None or old['starttime']!=plan['old_controller_starttime'] or old['state'] not in ('T','t'):raise RuntimeError('old controller identity drift')
        os.kill(old_pid,signal.SIGKILL)
        write(out/'train_test_handoff_completed.json',dict(adopted_task=TASKS[0],training_restarted=False,old_controller_retired=old_pid,retired_after_child_completed=True))
        for index,task in enumerate(TASKS):
            target=out/task['arm']/str(task['seed'])
            if index:
                if target.exists():raise RuntimeError('unexpected existing task; no resume')
                cmd=[sys.executable,'-u','-I',str(root/'scripts/decoder_attribution_train_val_20261003_v1.py'),'--phase','formal','--arm',task['arm'],'--seed',str(task['seed']),'--project-root',str(root),'--output-dir',str(target),'--lock',str(lock),'--approval',str(approval),'--health-receipt',str(health),'--device','cuda']
                execute(cmd,out/f"{task['seed']}_{task['arm']}.log",state,'TRAIN',task)
            completed_training(target,task,digest)
            cmd=[sys.executable,'-u','-I',str(Path(__file__).with_name('decoder_attribution_test_once_v1.py')),'--project-root',str(root),'--task-dir',str(target),'--lock',str(lock),'--authorization',str(a.authorization),'--arm',task['arm'],'--seed',str(task['seed'])]
            execute(cmd,out/f"{task['seed']}_{task['arm']}_test.log",state,'TEST_ONCE',task)
            result=valid_existing_test(target,task,digest,sha(a.authorization))
            completed.append(dict(task=task,test_sha256=sha(target/'test_evaluation.json'),checkpoint_sha256=result['checkpoint_sha256']))
            write(out/'train_test_completed_tasks.json',completed)
        write(out/'train_test_serial_completion.json',dict(status='ALL_NINE_TRAIN_AND_ONE_SHOT_TEST_COMPLETED',tasks=completed))
        write(state,dict(status='COMPLETED',completed=completed))
    except BaseException as exc:
        write(out/'train_test_serial_failure.json',dict(error=type(exc).__name__,detail=str(exc),no_retry=True,no_auto_resume=True))
        write(state,dict(status='FAILED_STOPPED',error=str(exc),completed=completed))
        raise

if __name__=='__main__':main()
