# Copyright (C) 2026 STURM output-path study contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Pinned official EoMT-S core with STURM sensor bridges; diagnostic only."""
from pathlib import Path
import hashlib
import importlib.util
import json
import sys
import types
import torch
from torch import nn
from torch.nn import functional as F
import timm
from timm.layers import resample_abs_pos_embed
from .optional_comparison import _GroupedSensorModel

ROOT = Path(__file__).resolve().parents[2]
SOURCE_COMMIT = '0f1e1ccbc0e105f1a100e885174e1d716ad02990'
WEIGHT_SHA = '7365b41a8a129afc74c12e5dd7af534146286d02822e4b89b7bbe8ea613ff74d'

def official_core_class():
    """Load unchanged core in a temporary namespace, restoring global aliases."""
    source = ROOT / 'external/eomt-official'
    lock = json.loads((ROOT / 'configs/eomt_official_source_lock.json').read_text())
    if lock['commit'] != SOURCE_COMMIT:
        raise RuntimeError('unexpected EoMT source revision')
    for name in ('models/eomt.py', 'models/scale_block.py', 'models/vit.py'):
        if hashlib.sha256((source / name).read_bytes()).hexdigest() != lock['source_files_sha256'][name]:
            raise RuntimeError(f'official source modified: {name}')
    names = ('models', 'models.scale_block', 'models.eomt')
    saved = {name: sys.modules.get(name) for name in names}
    try:
        package = types.ModuleType('models')
        package.__path__ = [str(source / 'models')]
        sys.modules['models'] = package
        for name in names[1:]:
            spec = importlib.util.spec_from_file_location(name, source / (name.replace('.', '/') + '.py'))
            module = importlib.util.module_from_spec(spec)
            sys.modules[name] = module
            spec.loader.exec_module(module)
        return sys.modules['models.eomt'].EoMT
    finally:
        for name, value in saved.items():
            if value is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = value

class _DINOv2Encoder(nn.Module):
    """Exact timm branch of official ViT, offline; no unused DINOv3 imports."""
    def __init__(self):
        super().__init__()
        self.backbone = timm.create_model('vit_small_patch14_reg4_dinov2', pretrained=False,
                                          img_size=(640, 640), patch_size=16, num_classes=0)
        self.register_buffer('pixel_mean', torch.tensor([.485, .456, .406]).reshape(1, 3, 1, 1))
        self.register_buffer('pixel_std', torch.tensor([.229, .224, .225]).reshape(1, 3, 1, 1))

class EoMTSSTURM(_GroupedSensorModel):
    architecture_name = 'official_eomt_small_sturm_semantic_adaptation'
    source_commit = SOURCE_COMMIT
    pretraining = 'official COCO panoptic EoMT-S 640 2x, not ADE20K-S'

    def __init__(self, weights_path=None):
        super().__init__(common_channels=3)
        weights_path = Path(weights_path or ROOT / 'models/pretrained/research_comparators/eomt_small_coco_panoptic.bin')
        if hashlib.sha256(weights_path.read_bytes()).hexdigest() != WEIGHT_SHA:
            raise RuntimeError('EoMT pretrained asset hash mismatch')
        checkpoint = torch.load(weights_path, map_location='cpu', weights_only=True)
        if set(key for key in checkpoint if not key.startswith('network.')) != {'criterion.empty_weight'}:
            raise RuntimeError('unexpected official checkpoint namespace')
        state = {key.removeprefix('network.'): value for key, value in checkpoint.items()
                 if key.startswith('network.')}
        self.core = official_core_class()(_DINOv2Encoder(), num_classes=133, num_q=200, num_blocks=3)
        self.core.load_state_dict(state, strict=True)
        self.pretrained_loaded_keys = len(state)
        self.core.attn_mask_probs.fill_(1.)  # pilot retraining policy, explicit in metadata
        self.core.class_head = nn.Linear(384, 3)  # two semantic classes plus no-object
        backbone = self.core.encoder.backbone
        # DINOv2 reg4 uses no_embed_class: class/register tokens have no pos embed.
        backbone.pos_embed = nn.Parameter(resample_abs_pos_embed(backbone.pos_embed,
                                           new_size=(8, 8), num_prefix_tokens=0 if backbone.no_embed_class else backbone.num_prefix_tokens))
        backbone.patch_embed.set_input_size(img_size=(128, 128))
        self.source_provenance = {'official_files_unchanged': True,
                                  'encoder': 'equivalent offline timm DINOv2 branch',
                                  'strict_original_classes': 133,
                                  'geometry': '640->128; patch16; position embedding bicubic antialias'}

    def train(self, mode=True):
        super().train(mode)
        self.core.masked_attn_enabled = mode
        return self

    def forward_queries(self, image, sensor):
        if image.shape[-2:] != (128, 128):
            raise ValueError('local EoMT diagnostic requires 128x128 inputs')
        return self.core(self.input_adapters(image, self._sensor_name(sensor)))

    def forward_native(self, image, sensor):
        masks, classes = self.forward_queries(image, sensor)
        masks = F.interpolate(masks[-1], image.shape[-2:], mode='bilinear', align_corners=False)
        scores = torch.einsum('bqc,bqhw->bchw', classes[-1].softmax(-1)[..., :2], masks.sigmoid())
        return scores.clamp_min(1e-8).log()  # derived log scores, not native dense classifier logits
