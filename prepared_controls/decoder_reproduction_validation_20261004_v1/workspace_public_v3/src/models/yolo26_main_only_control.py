# Copyright (C) 2026 STURM output-path study contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Native common16 control with the decoder's main-output-only supervision.

Keep the original native topology, initialization and pretrained transfer.
Do not add the high-resolution decoder or reset/freeze any head parameters.
The registered auxiliary head is retained for checkpoint compatibility, but
its logits are not exposed to the loss and receive no supervised gradients.
This is a new control, not the historical native run.
"""
import torch

from .yolo26_semantic import SharedYolo26


class SharedYolo26MainOnly(SharedYolo26):
    """Return native main logits for training/evaluation and grouped batches."""

    def forward_native(self, image, sensor):
        outputs = super().forward_native(image, sensor)
        if isinstance(outputs, (tuple, list)):
            if not outputs:
                raise RuntimeError('Native head returned no main logits.')
            outputs = outputs[0]
        if not isinstance(outputs, torch.Tensor):
            raise TypeError('Native main-only control must return a Tensor.')
        return outputs
