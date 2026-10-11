# Copyright (C) 2026 STURM output-path study contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Official LALE default architecture, scratch STURM shared-core adaptation."""
import ast
import hashlib
import json
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F
from .optional_comparison import _GroupedSensorModel

ROOT = Path(__file__).resolve().parents[2]
COMMIT = '8afced654190225e3e08a0e1cc3d8e403e7957d5'
CLASSES = ('RMSNorm', 'StarReLU', 'ConvMixerBlock', 'TransformerBlock', 'Downsample', 'MLPDecoder', 'LALE')

def official_lale_class():
    path = ROOT/'external/lale-official/RS_train.py'
    lock = json.loads((ROOT/'configs/lale_official_source_lock.json').read_text(encoding='utf-8-sig'))
    if lock['commit'] != COMMIT or hashlib.sha256(path.read_bytes()).hexdigest() != lock['core_sha256']:
        raise RuntimeError('official LALE source identity changed')
    tree = ast.parse(path.read_text(encoding='utf-8'))
    nodes = [node for node in tree.body if isinstance(node, ast.ClassDef) and node.name in CLASSES]
    if tuple(node.name for node in nodes) != CLASSES:
        raise RuntimeError('unexpected official architecture class list')
    # Preserve original AST bodies, including normalization and attention calls.
    namespace = {'__name__': 'lale_official_architecture', 'torch': torch, 'nn': nn, 'F': F}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), 'exec'), namespace)
    return namespace['LALE']

class LALESTURM(_GroupedSensorModel):
    architecture_name = 'official_LALE_default_shared_STURM_scratch'
    def __init__(self):
        super().__init__(common_channels=3)
        self.core = official_lale_class()(in_chans=3, num_classes=2,
            dims=[32, 64, 128, 256], depths=[1, 1, 1, 1], num_heads=4)

    def forward_native(self, image, sensor):
        if image.shape[-2:] != (128, 128):
            raise ValueError('local LALE pretest requires 128x128 input')
        return self.core(self.input_adapters(image, self._sensor_name(sensor)))
