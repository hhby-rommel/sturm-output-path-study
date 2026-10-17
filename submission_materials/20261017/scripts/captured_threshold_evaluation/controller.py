"""Explicit server admission and serial val->threshold-freeze->one-pass-test; no training."""
from pathlib import Path
from datetime import datetime,timezone
import argparse,hashlib,json,os,shutil,subprocess,sys,time
B=Path('/root/autodl-tmp/sturm_codex_20260919')
C=B/'experiments/checkpoint_threshold_validation_20261016_v1'
O=B/'results/checkpoint_threshold_validation_20261016_v1'
PY=B/'.venv/bin/python'
def read(p):return json.loads(p.read_text())
def dump(p,v):
    t=p.with_suffix('.tmp');t.write_text(json.dumps(v,indent=2,allow_nan=False));t.replace(p)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def identity(pid):
    p=Path('/proc')/str(pid)
    return dict(pid=pid,command=(p/'cmdline').read_bytes().replace(b'\0',b' ').decode(),
                start_ticks=(p/'stat').read_text().rsplit(')',1)[1].split()[19])
def preflight():
    import torch
    assert torch.__version__=='2.1.0+cu121' and torch.cuda.is_available()
    assert torch.cuda.get_device_name()=='NVIDIA GeForce RTX 4090 D'
    assert shutil.disk_usage(B).free>=1024**3
    for q in read(C/'deployment_identity.json')['files']:
        assert sha(C/q['relative_path'])==q['sha256'],q['relative_path']
    active=[]
    for p in Path('/proc').iterdir():
        if not p.name.isdigit() or int(p.name) in [os.getpid(),os.getppid()]:continue
        try:cmd=(p/'cmdline').read_bytes().replace(b'\0',b' ').decode()
        except (OSError,UnicodeError):continue
        if str(B) in cmd and 'python' in cmd and '-c ' not in cmd:active.append(dict(pid=int(p.name),command=cmd))
    assert not active,active
    records=read(C/'frozen_assets.json');assert len(records)==6
    checkpoints={}
    for q in records:
        p=Path(q['checkpoint']);actual=sha(p);assert actual==q['checkpoint_sha256'],str(p)
        st=p.stat();checkpoints[str(p)]=dict(sha256=actual,size=st.st_size,mtime_ns=st.st_mtime_ns,
            purpose='Frozen checkpoint handoff to new val/test inference scope',time_utc=datetime.now(timezone.utc).isoformat(),conclusion='MATCH')
    prior_val=B/'results/val_event_counts_20261010_v1/reevaluation_complete.json'
    prior_test=B/'results/eval12026_extension_20261015_v1/completion.json'
    assert read(prior_val)['status']=='COMPLETE'
    assert read(prior_test)['status']=='SIX_EXTENSION_CHECKPOINTS_EVAL12026_COMPLETE'
    for q in records:
        assert (prior_test.parent/f"{q['system']}_{q['seed']}_reevaluation.json").exists()
    dump(C/'checkpoint_admission.json',dict(status='PASS',checkpoints=checkpoints,
        torch=torch.__version__,GPU=torch.cuda.get_device_name(),new_source_identity=str(C/'deployment_identity.json'),
        data_identity='Reuse unchanged historical data/split receipts; no raw-data hashing'))
def phase(stage):
    entry=C/('validation.py' if stage=='val' else 'test_frozen_thresholds.py')
    with (C/f'{stage}.log').open('x') as log:
        p=subprocess.Popen([str(PY),'-u',str(entry)],cwd=B,stdout=log,stderr=subprocess.STDOUT)
        dump(C/'status.json',dict(status='RUNNING',stage=stage,child=identity(p.pid),controller=identity(os.getpid()),time_unix=time.time()))
        rc=p.wait()
    assert rc==0,dict(stage=stage,returncode=rc)
def serial():
    try:
        assert not O.exists();O.mkdir()
        phase('val')
        v=read(O/'val/reevaluation_complete.json')
        assert v['status']=='COMPLETE' and v['models']==6 and v['tiles']==4853
        assert v['threshold_count']==19 and v['frozen_threshold_units']==24
        assert v['baseline_val_event_counts_equal'] and not v['test_access']
        prior=read(B/'results/val_event_counts_20261010_v1/reevaluation_complete.json')
        assert v['input_stream_sha256']==prior['input_stream_sha256']
        selection=O/'val/selected_thresholds_frozen.json'
        freeze=read(O/'val/threshold_freeze_receipt.json')
        assert sha(selection)==freeze['selection_sha256']
        units=read(selection)['thresholds']
        assert len(units)==24 and len({(q['model'],q['view'],q['sensor']) for q in units})==24
        models=sorted(f"{q['system']}_{q['seed']}" for q in read(C/'frozen_assets.json'))
        claim=dict(status='CLAIMED_ONCE',attempt=1,models=models,selection_sha256=freeze['selection_sha256'],
            validation_completion=v,scope='One new test traversal at frozen selected thresholds only',time_unix=time.time())
        with (C/'test_access_claim.json').open('x') as f:json.dump(claim,f,indent=2)
        phase('test')
        t=read(O/'test/reevaluation_complete.json')
        assert t['status']=='COMPLETE' and t['models']==6 and t['tiles']==4863
        assert t['baseline_fixed_test_counts_equal'] and t['selection_sha256']==claim['selection_sha256']
        assert sha(selection)==claim['selection_sha256']
        dump(C/'completion.json',dict(status='SIX_CHECKPOINT_THRESHOLD_AND_VAL_EXTENSION_COMPLETE',training=False,
            validation_models=6,threshold_units=24,test_traversals=1,checkpoint_reselection=False,
            selection_sha256=claim['selection_sha256'],time_unix=time.time(),paper_changed=False,release_published=False))
        dump(C/'status.json',dict(status='COMPLETE_NO_OTHER_JOBS_QUEUED'))
    except BaseException:
        dump(C/'status.json',dict(status='STOPPED',automatic_retry=False,test_claim_preserved=(C/'test_access_claim.json').exists()))
        raise
def launch():
    assert not O.exists() and not (C/'launch.json').exists() and not (C/'test_access_claim.json').exists()
    preflight()
    with (C/'controller.log').open('x') as log:
        p=subprocess.Popen([str(PY),'-u',str(C/'controller.py'),'--serial'],cwd=B,
            stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    receipt=dict(controller=identity(p.pid),time_unix=time.time(),scope='Six original checkpoints2029/2030; val/freeze/test only')
    dump(C/'launch.json',receipt);print(json.dumps(receipt))
if __name__=='__main__':
    p=argparse.ArgumentParser();g=p.add_mutually_exclusive_group(required=True)
    g.add_argument('--execute-authorized',action='store_true');g.add_argument('--serial',action='store_true')
    a=p.parse_args();serial() if a.serial else launch()
