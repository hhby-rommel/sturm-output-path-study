# Copyright (C) 2026 STURM output-path study contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Prepared exposure-matched control; formal launch remains separately gated."""
from __future__ import annotations

import copy
import torch
from torch.utils.data import DataLoader, Sampler, WeightedRandomSampler

from src.dataset.sturm_flood import grouped_sensor_collate
from src.training.single_sensor_decoder_control import validate_scope


def exposure_count(total: int, sensor: str, epoch: int) -> int:
    if total < 2 or sensor not in {'S1','S2'} or epoch < 1:
        raise ValueError('positive epoch, at least two draws, S1/S2 required')
    # Odd N: give S1 the extra draw in odd epochs, S2 in even epochs.
    return total//2 + int(total%2 and ((sensor=='S1') == bool(epoch%2)))


class ExposureMatchedSampler(Sampler):
    """Uniform within one sensor, conditional original sampling distribution.

    Unlike the original mixed sampler this fixes sensor draw count, batches16,
    and updates per model. No claim of historical draw/gradient equivalence.
    """
    def __init__(self, size: int, total: int, sensor: str, seed: int):
        if size < 1:
            raise ValueError('empty sensor selection')
        self.size, self.total, self.sensor = size, total, sensor
        self.epoch = 1
        self.generator = torch.Generator().manual_seed(seed+1)
        exposure_count(total,sensor,self.epoch)

    def __len__(self):
        return exposure_count(self.total,self.sensor,self.epoch)

    def __iter__(self):
        return iter(WeightedRandomSampler(
            torch.full((self.size,),1.0/self.size,dtype=torch.double),
            len(self),replacement=True,generator=self.generator))

    def set_epoch(self, epoch: int):
        exposure_count(self.total,self.sensor,epoch)
        self.epoch=epoch  # Do not reseed at epoch boundaries.

    def state_dict(self):
        return dict(size=self.size,total=self.total,sensor=self.sensor,epoch=self.epoch,
                    generator=self.generator.get_state().clone())

    def load_state_dict(self, state):
        if (state['size'],state['total'],state['sensor']) != (self.size,self.total,self.sensor):
            raise ValueError('sampler scope mismatch')
        self.set_epoch(state['epoch'])
        self.generator.set_state(state['generator'].cpu())


def make_exposure_loader(dataset, *, split: str, sensor: str, seed: int,
                         mixed_draws_per_epoch: int):
    validate_scope(split,seed,sensor)  # Local health only; no formal training entry.
    base=getattr(dataset,'base',dataset)
    if getattr(base,'split',None)!=split or not dataset.rows or any(r['sensor']!=sensor for r in dataset.rows):
        raise ValueError('single-sensor train/val scope mismatch')
    sampler=ExposureMatchedSampler(len(dataset.rows),mixed_draws_per_epoch,sensor,seed) if split=='train' else None
    return DataLoader(dataset,batch_size=16,sampler=sampler,shuffle=False,num_workers=0,
                      pin_memory=torch.cuda.is_available(),collate_fn=grouped_sensor_collate)


def pooled_confusion(matrices):
    """Sum frozen models' counts first; never average sensor F1."""
    matrix=torch.stack([torch.as_tensor(m,dtype=torch.int64) for m in matrices]).sum(0)
    if matrix.shape!=(2,2) or torch.any(matrix<0):
        raise ValueError('nonnegative binary confusion matrices required')
    tn,fp,fn,tp=(int(v) for v in matrix.flatten())
    return dict(confusion_matrix=matrix.tolist(),flood_precision=tp/max(tp+fp,1),
                flood_recall=tp/max(tp+fn,1),flood_f1=2*tp/max(2*tp+fp+fn,1))


def create_final_evaluation_claim(path, *, frozen_best_hashes: dict, approved: bool,
                                  seed: int):
    """Persist claim before any future test dataset construction; fail closed."""
    import json
    if not approved or seed not in {2026,2027,2028}:
        raise ValueError('explicit formal evaluation authorization required')
    if set(frozen_best_hashes)!={'S1','S2'} or any(
        len(v)!=64 or any(c not in '0123456789abcdef' for c in v) for v in frozen_best_hashes.values()):
        raise ValueError('both frozen best SHA256 values required')
    with path.open('x',encoding='utf-8') as f:
        json.dump(dict(seed=seed,best_sha256=copy.deepcopy(frozen_best_hashes),
                       status='CLAIMED_BEFORE_TEST_DATASET_ACCESS_NO_AUTOMATIC_RETRY'),f)
