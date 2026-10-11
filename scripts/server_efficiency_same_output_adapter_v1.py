# Copyright (C) 2026 STURM output-path study contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
# Source-release notice added 2026-10-11; numerical code unchanged.
"""Same-output supplement: native main logits + explicit timed interpolation."""
import importlib.util
from pathlib import Path

def construct(unit, config):
    path=Path(config['original_adapter_path'])
    spec=importlib.util.spec_from_file_location('original_reviewed_efficiency_adapter',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module.construct(unit,config)

def forward(model,image,sensor):
    import torch
    import torch.nn.functional as F
    logits=model.forward_native(image,sensor)
    if not isinstance(logits,torch.Tensor):raise TypeError('Explicit native tensor interface required')
    if type(model).__name__=='SharedYolo26MainOnly':
        if tuple(logits.shape[1:])!=(2,16,16):raise ValueError('Historical native shape differs')
        logits=F.interpolate(logits,size=(128,128),mode='bilinear',align_corners=False)
    elif type(model).__name__ not in ('Yolo26UNetDecoder','IndependentNativeSturmUNet'):
        raise TypeError('Only three core systems registered; no substitute model')
    if tuple(logits.shape[1:])!=(2,128,128):raise ValueError('Output must be 128x128 two-class logits')
    return logits

def select_main(output):return output
