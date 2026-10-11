# Copyright (C) 2026 STURM output-path study contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
# Source-release notice added 2026-10-11; numerical code unchanged.
"""Original-protocol independent control plumbing and fail-closed admission."""
import copy
import hashlib
import json
import random
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from src.dataset.sturm_flood import SturmFloodDataset,grouped_sensor_collate
from src.training.exposure_matched_decoder_control import ExposureMatchedSampler


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save_json(path,value):
    Path(path).write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf8')


def check_authorization(path,*,action,seed=None,sensor=None,bundle_digest=None):
    if action not in {'train','final_evaluation'}:raise ValueError('unknown action')
    approval=json.loads(Path(path).read_text(encoding='utf8')) if path else {}
    if approval.get('approved') is not True or approval.get('action')!=action:
        raise PermissionError('explicit matching authorization required')
    if seed not in approval.get('seeds',[]):raise PermissionError('seed not authorized')
    if action=='train' and sensor not in approval.get('sensors',[]):raise PermissionError('sensor not authorized')
    if not bundle_digest or approval.get('bundle_sha256')!=bundle_digest:
        raise PermissionError('authorization must identify exact frozen bundle')
    return approval


def loader(root,split,sensor,seed,epoch=1):
    if split not in {'train','val'}:raise ValueError('train/val loader only; test uses separate protected path')
    ds=SturmFloodDataset(Path(root)/'data/manifests/manifest.csv',Path(root)/'data/splits/event_split_v1.csv',split,sensor=sensor)
    sampler=ExposureMatchedSampler(len(ds),14561,sensor,seed) if split=='train' else None
    if sampler:sampler.set_epoch(epoch)
    return DataLoader(ds,batch_size=16,sampler=sampler,shuffle=False,num_workers=0,
        pin_memory=torch.cuda.is_available(),collate_fn=grouped_sensor_collate)


def rng_state():
    return dict(python=random.getstate(),numpy=np.random.get_state(),torch=torch.random.get_rng_state(),
        cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [])


def restore_rng(state):
    random.setstate(state['python']);np.random.set_state(state['numpy']);torch.random.set_rng_state(state['torch'].cpu())
    if torch.cuda.is_available():torch.cuda.set_rng_state_all([s.cpu() for s in state['cuda']])


def checkpoint(model,optimizer,sampler,*,epoch,step,selection,stale,trace=None,offset=None):
    return copy.deepcopy(dict(model=model.state_dict(),optimizer=optimizer.state_dict(),sampler=sampler.state_dict(),
        rng=rng_state(),epoch=epoch,step=step,selection=selection,stale=stale,trace=trace,offset=offset))


def restore_checkpoint(model,optimizer,sampler,state):
    model.load_state_dict(state['model'],strict=True);optimizer.load_state_dict(state['optimizer'])
    sampler.load_state_dict(state['sampler']);restore_rng(state['rng'])


def compare_states(a,b,atol=1e-6,rtol=1e-5):
    """Tolerance declared for CUDA floats; RNG and integer state exact."""
    if isinstance(a,torch.Tensor):
        if a.shape!=b.shape or a.dtype!=b.dtype:raise AssertionError('tensor contract mismatch')
        if a.is_floating_point():
            if not torch.allclose(a.cpu(),b.cpu(),atol=atol,rtol=rtol):raise AssertionError('float state mismatch')
        elif not torch.equal(a.cpu(),b.cpu()):raise AssertionError('integer/RNG state mismatch')
    elif isinstance(a,np.ndarray):
        if not np.array_equal(a,b):raise AssertionError('numpy RNG mismatch')
    elif isinstance(a,dict):
        if a.keys()!=b.keys():raise AssertionError('state keys mismatch')
        for k in a:compare_states(a[k],b[k],atol,rtol)
    elif isinstance(a,(tuple,list)):
        if len(a)!=len(b):raise AssertionError('state length mismatch')
        for x,y in zip(a,b):compare_states(x,y,atol,rtol)
    elif a!=b:raise AssertionError('scalar state mismatch')
