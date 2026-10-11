# Copyright (C) 2026 STURM output-path study contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
# Source-release notice added 2026-10-11; numerical code unchanged.
"""Exclusive, separately authorized seed2028 YOLOv12 light-head test after training.

Use the frozen v1 runtime. Recheck selected full validation before claiming any
test access; preserve the claim on failure and never retry a claimed test.
"""
import os
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
import argparse, hashlib, importlib.util, json, platform, subprocess, time
from pathlib import Path
from importlib.metadata import version

ROOT=Path(__file__).resolve().parents[1]
PROJECT=Path(__file__).resolve().parents[1]

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--seed',type=int,choices=(2028,),required=True)
    p.add_argument('--approved',action='store_true')
    a=p.parse_args()
    if not a.approved:raise RuntimeError('explicit test authorization required')
    if ROOT!=PROJECT/'.codex_deployments/yolov12_light_server_20261001_v1':raise RuntimeError('wrong deployment')
    out=PROJECT/'results/formal/yolov12n_turbo_light_semantic_v1'/str(a.seed)
    if (out/'test_access_claim.json').exists() or (out/'test_evaluation.json').exists():raise RuntimeError('test already claimed/evaluated; do not repeat')
    busy=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
    if busy:raise RuntimeError('GPU job already active: '+busy)
    spec=importlib.util.spec_from_file_location('frozen_yolov12_train',ROOT/'scripts/server_yolov12_light_formal_v3.py')
    frozen=importlib.util.module_from_spec(spec);spec.loader.exec_module(frozen)
    torch=frozen.torch
    torch.set_num_threads(2);torch.set_num_interop_threads(2);torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    complete=json.loads((out/'summary.json').read_text());metadata=json.loads((out/'metadata.json').read_text())
    locked=frozen.runtime();weight=frozen.assets()
    if complete['status']!='FORMAL_TRAINING_COMPLETED_TEST_PENDING' or not complete['full_val_replay_exact']:raise RuntimeError('completed training with full replay required')
    if complete['config']!=frozen.CONFIG or complete['runtime_hashes']!=locked or metadata['runtime_hashes']!=locked:raise RuntimeError('frozen runtime/config/data mismatch')
    environment=dict(torch=torch.__version__,cuda=torch.version.cuda,gpu=torch.cuda.get_device_name(0),python=platform.python_version(),packages={n:version(n) for n in ('numpy','torchvision','rasterio','scipy')})
    if environment!=complete['environment']:raise RuntimeError('training environment changed')
    history=json.loads((out/'history.json').read_text())
    chosen=max(history,key=lambda r:(r['selection']['degraded_flood_f1'],r['selection']['clean_flood_f1'],-r['epoch']))
    if chosen['selection']!=complete['best']:raise RuntimeError('selection cannot be rebuilt from history')
    checkpoint=out/'best.pt';digest=frozen.sha(checkpoint)
    state=torch.load(checkpoint,map_location='cpu',weights_only=False)
    if state['seed']!=a.seed or state['selection']!=chosen['selection'] or state['config']!=frozen.CONFIG or state['runtime_hashes']!=locked:raise RuntimeError('selected checkpoint identity differs')
    mutex=PROJECT/'experiments/yolov12_light_gpu.lock'
    fd=os.open(mutex,os.O_CREAT|os.O_EXCL|os.O_WRONLY);os.write(fd,str(os.getpid()).encode());os.close(fd)
    try:
        frozen.seed_everything(a.seed)
        model,init=frozen.build(weight);model=model.cuda();model.load_state_dict(state['model'],strict=True)
        del state
        criterion=frozen.MaskedSemanticLoss();device=torch.device('cuda')
        val=frozen.SturmFloodDataset(ROOT/'data/manifests/manifest.csv',ROOT/'data/splits/event_split_v1.csv','val')
        loader=frozen.DataLoader(val,batch_size=16,num_workers=0,collate_fn=frozen.grouped_sensor_collate,generator=torch.Generator().manual_seed(a.seed+1002))
        print(f'PRETEST seed={a.seed} selected_epoch={chosen["epoch"]} full_val_replay starting',flush=True)
        replay=frozen.evaluate_clean_and_degraded(model,loader,criterion,device,seed=a.seed,tag=1)
        if not frozen.exact(replay,chosen['validation']):raise RuntimeError('loaded selected checkpoint full validation differs')
        if frozen.sha(checkpoint)!=digest:raise RuntimeError('checkpoint changed during pretest replay')
        pretest=dict(exact_metrics_equal=True,checkpoint_sha256=digest,selection=chosen['selection'],environment=environment)
        with (out/'pretest_best_val_recheck_v1.json').open('x',encoding='utf-8') as f:json.dump(pretest,f,indent=2)
        claim=dict(seed=a.seed,selected_checkpoint_sha256=digest,selection=chosen['selection'],runtime_hashes=locked,
            test_runner_sha256=frozen.sha(Path(__file__)),environment=environment,time_unix=time.time(),
            scope='one exclusive claimed test invocation; claim retained on failure',authorization='user: 检查当前yolov12的2028实验以及一次性实验进度，如果结束则开启下一个实验的服务器准备和正式实验')
        with (out/'test_access_claim.json').open('x',encoding='utf-8') as f:json.dump(claim,f,ensure_ascii=False,indent=2)
        # This constructor and all test image reads occur only after the claim.
        test=frozen.SturmFloodDataset(ROOT/'data/manifests/manifest.csv',ROOT/'data/splits/event_split_v1.csv','test')
        if len(test)!=4863:raise RuntimeError('test tile count differs from frozen split')
        events={r['event_id'] for r in test.rows}
        if events&{r['event_id'] for r in val.rows}:raise RuntimeError('test/val event overlap')
        loader=frozen.DataLoader(test,batch_size=16,num_workers=0,collate_fn=frozen.grouped_sensor_collate,generator=torch.Generator().manual_seed(a.seed+1003))
        print(f'TEST seed={a.seed} tiles={len(test)} events={len(events)} starting',flush=True)
        metrics=frozen.evaluate_clean_and_degraded(model,loader,criterion,device,seed=a.seed,tag=1)
        if frozen.sha(checkpoint)!=digest:raise RuntimeError('checkpoint changed during test')
        record=dict(status='FORMAL_SEED2028_CLAIMED_TEST_COMPLETED',metrics=metrics,real_test_split_used=True,
            test_tiles=len(test),test_events=len(events),**claim)
        with (out/'test_evaluation.json').open('x',encoding='utf-8') as f:json.dump(record,f,ensure_ascii=False,indent=2)
        print(f'TEST_COMPLETED seed={a.seed} clean={metrics["clean"]["overall"]["flood_f1"]:.6f} degraded={metrics["degraded"]["overall"]["flood_f1"]:.6f}',flush=True)
    finally:mutex.unlink()

if __name__=='__main__':main()
