# Copyright (C) 2026 STURM output-path study contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Local-only matched initialization and separate-core architecture controls."""
import copy
import torch
from torch import nn
from .yolo26_semantic import SharedYolo26, load_matching_yolo26_weights
from .yolo26_unet_decoder import Yolo26UNetDecoder


def matched_pair(weights_path, seed=10110):
    torch.manual_seed(seed)
    native = SharedYolo26(common_channels=16)
    report = load_matching_yolo26_weights(native.core, weights_path)
    torch.manual_seed(seed)
    decoder = Yolo26UNetDecoder(common_channels=16)
    state = decoder.state_dict()
    shared = native.state_dict()
    copied = []
    for key, value in shared.items():
        if key not in state or state[key].shape != value.shape:
            raise RuntimeError(f'unmatched common initialization {key}')
        state[key] = value.clone()
        copied.append(key)
    decoder.load_state_dict(state, strict=True)
    assert all(torch.equal(native.state_dict()[k], decoder.state_dict()[k]) for k in copied)
    return native, decoder, dict(pretraining=report, matched_initial_keys=len(copied),
                                common_channels=16, initialization_seed=seed)


class IndependentYolo26Decoder(nn.Module):
    """Two separate parameter sets, routed through the same grouped engine.

    Uses one registered aggregate loss/global clip/update for architecture-only
    comparison. It is not two separately optimized training jobs.
    """
    def __init__(self, shared):
        super().__init__()
        self.models = nn.ModuleDict()
        for sensor in ('S1', 'S2'):
            expert = copy.deepcopy(shared)
            other = 'S2' if sensor == 'S1' else 'S1'
            del expert.input_adapters.projections[other]
            del expert.feature_adapters[f'P3_{other}']
            del expert.feature_adapters[f'P4_{other}']
            self.models[sensor] = expert
        ids = [{id(p) for p in m.parameters()} for m in self.models.values()]
        assert not (ids[0] & ids[1])

    def forward(self, batch):
        sensor = batch['sensor']
        if isinstance(sensor, list):
            if len(set(sensor)) != 1:
                raise ValueError('native batch must contain one sensor')
            sensor = sensor[0]
        return self.models[sensor](batch)
