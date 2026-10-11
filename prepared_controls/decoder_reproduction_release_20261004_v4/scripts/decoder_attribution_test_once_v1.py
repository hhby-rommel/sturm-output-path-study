# Copyright (C) 2026 STURM output-path study contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
# Source-release notice added 2026-10-11; numerical code unchanged.
"""One claimed full-test evaluation of a completed task's frozen val-best."""
from pathlib import Path
import argparse,hashlib,json,sys,time

TASKS=[dict(arm=arm,seed=seed) for seed in (2026,2027,2028) for arm in ('no_skip_capacity','no_input','no_p1')]

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()

def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))

def write(p,obj):
    p=Path(p);tmp=p.with_suffix(p.suffix+'.tmp')
    tmp.write_text(json.dumps(obj,ensure_ascii=False,indent=2)+'\n',encoding='utf-8');tmp.replace(p)

def claim_once(out,identity):
    if any((out/name).exists() for name in ('test_evaluation.json','test_access_claim.json','test_failure.json')):
        raise PermissionError('test already claimed or completed; no second access')
    with (out/'test_access_claim.json').open('x',encoding='utf-8') as f:
        json.dump(identity,f,ensure_ascii=False,indent=2)

def valid_existing_test(out,task,lock_sha,authorization_sha):
    """Verify recorded result without constructing a dataset or running a model."""
    result=read(out/'test_evaluation.json');claim=read(out/'test_access_claim.json');done=read(out/'completion.json')
    if (out/'test_failure.json').exists():raise RuntimeError('previous test failure; no retry')
    for obj in (result,claim):
        if obj['arm']!=task['arm'] or obj['seed']!=task['seed'] or obj['lock_sha256']!=lock_sha or obj['authorization_sha256']!=authorization_sha:
            raise RuntimeError('test identity mismatch')
        if obj['checkpoint_sha256']!=done['best_sha256'] or obj['checkpoint_sha256']!=sha(out/'best.pt') or obj['selection']!=done['best']:
            raise RuntimeError('frozen best changed')
    if result['status']!='FORMAL_ONE_SHOT_TEST_COMPLETED' or result['test_tiles']!=4863 or result['test_events']!=12 or result['real_test_split_used'] is not True:
        raise RuntimeError('test completion mismatch')
    return result

def run(a):
    task=dict(arm=a.arm,seed=a.seed);auth=read(a.authorization);lock_sha=sha(a.lock)
    if auth.get('approved') is not True or auth.get('action')!='one_shot_test_after_each_task' or auth.get('tasks')!=TASKS or auth.get('lock_sha256')!=lock_sha or auth.get('test_entry_sha256')!=sha(__file__) or task not in TASKS:
        raise PermissionError('exact test authorization missing')
    root=a.project_root.resolve();out=a.task_dir.resolve();done=read(out/'completion.json');meta=read(out/'metadata.json')
    if done.get('status')!='FORMAL_TRAIN_VAL_COMPLETED_TEST_PENDING' or done['arm']!=a.arm or done['seed']!=a.seed or done['lock_sha256']!=lock_sha or meta['phase']!='formal' or (out/'failure.json').exists():
        raise RuntimeError('successful training completion required')
    if meta['arm']!=a.arm or meta['seed']!=a.seed or meta['lock_sha256']!=lock_sha or meta['test_enabled'] is not False:
        raise RuntimeError('training metadata mismatch')
    best=out/'best.pt';checkpoint_sha=sha(best)
    if checkpoint_sha!=done['best_sha256']:raise RuntimeError('best bytes changed')
    # Imports use frozen original implementation, never a reconstructed substitute.
    sys.path.insert(0,str(root))
    import torch
    from scripts import decoder_attribution_train_val_20261003_v1 as runner
    from scripts.server_main import make_loader
    from src.dataset.sturm_flood import SturmFloodDataset
    from src.training.masked_segmentation import MaskedSemanticLoss
    from src.evaluation.formal import evaluate_clean_and_degraded
    runner.verify(root,read(a.lock))
    if not torch.cuda.is_available() or done['GPU']!=torch.cuda.get_device_name() or done['torch']!=torch.__version__:
        raise RuntimeError('test runtime differs from admitted training environment')
    identity=dict(arm=a.arm,seed=a.seed,lock_sha256=lock_sha,authorization_sha256=sha(a.authorization),
        test_entry_sha256=sha(__file__),checkpoint_sha256=checkpoint_sha,selection=done['best'],
        train_completion_sha256=sha(out/'completion.json'),once=True,real_test_split_used=True,
        fixed_degraded_tag=1,test_loader_seed=a.seed+3,access_claim_unix=time.time())
    claim_once(out,identity)
    try:
        start=time.monotonic();torch.set_num_threads(4);device=torch.device('cuda')
        model,_=runner.build(a.arm,a.seed,root,device)
        ck=torch.load(best,map_location='cpu')
        if ck['seed']!=a.seed or ck['epoch']!=done['best']['epoch'] or ck['selection']!=done['best']:
            raise RuntimeError('checkpoint selection mismatch')
        model.load_state_dict(ck['model_state_dict'],strict=True);del ck
        data=SturmFloodDataset(root/'data/manifests/manifest.csv',root/'data/splits/event_split_v1.csv','test')
        if len(data)!=4863 or len({r['event_id'] for r in data.rows})!=12:raise RuntimeError('test split scope drift')
        loader=make_loader(data,batch_size=16,train=False,seed=a.seed+3)
        observer=runner.EvaluationStream(model)
        metrics=evaluate_clean_and_degraded(observer,loader,MaskedSemanticLoss(aux_weight=.4).to(device),device,seed=a.seed,tag=1)
        if sha(best)!=checkpoint_sha:raise RuntimeError('best changed during test')
        runner.verify(root,read(a.lock))
        payload=dict(identity,status='FORMAL_ONE_SHOT_TEST_COMPLETED',test_tiles=len(data),test_events=12,
            metrics=metrics,evaluation_input_stream_sha256=observer.summaries(),
            manifest_sha256=sha(root/'data/manifests/manifest.csv'),split_sha256=sha(root/'data/splits/event_split_v1.csv'),
            seconds=time.monotonic()-start,torch=torch.__version__,cuda=torch.version.cuda,GPU=torch.cuda.get_device_name(),
            best_reselected=False,optimizer_loaded=False,training_modified=False)
        write(out/'test_evaluation.json',payload)
        print('ONE_SHOT_TEST_COMPLETED',a.arm,a.seed,metrics['degraded']['overall']['flood_f1'],flush=True)
        return payload
    except BaseException as exc:
        write(out/'test_failure.json',dict(error=type(exc).__name__,detail=str(exc),claim_retained=True,no_retry=True))
        raise

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--project-root',type=Path,required=True);p.add_argument('--task-dir',type=Path,required=True)
    p.add_argument('--lock',type=Path,required=True);p.add_argument('--authorization',type=Path,required=True)
    p.add_argument('--arm',choices=('no_skip_capacity','no_input','no_p1'),required=True);p.add_argument('--seed',type=int,required=True)
    a=p.parse_args();run(a)

if __name__=='__main__':main()
