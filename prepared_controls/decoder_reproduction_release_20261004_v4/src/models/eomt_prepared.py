# Copyright (C) 2026 STURM output-path study contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
# Source-release notice added 2026-10-11; numerical code unchanged.
"""Structured-query adapter for the shared engine; legacy pilot unchanged."""
import torch
from torch.nn import functional as F
from .eomt_s_comparator import EoMTSSTURM

class EoMTPreparedSTURM(EoMTSSTURM):
    architecture_name = 'official_eomt_small_sturm_dense_valid_v2'

    def forward_native(self, image, sensor):
        masks, classes = self.forward_queries(image, sensor)
        last = F.interpolate(masks[-1], image.shape[-2:], mode='bilinear', align_corners=False)
        scores = torch.einsum('bqc,bqhw->bchw', classes[-1].softmax(-1)[..., :2], last.sigmoid())
        return {'logits': scores.clamp_min(1e-8).log(), 'query_masks': masks, 'query_classes': classes}
