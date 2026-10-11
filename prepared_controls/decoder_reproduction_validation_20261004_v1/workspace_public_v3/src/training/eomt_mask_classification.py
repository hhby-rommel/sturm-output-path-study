# Copyright (C) 2026 STURM output-path study contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Query matching diagnostic: native objective family, dense VALID pixels.

Unlike the official uncertain-point sampler, this bounded 128px diagnostic
uses all valid pixels. Not an exact reproduction of the official trainer.
"""
import torch
from torch import nn
from torch.nn import functional as F
from scipy.optimize import linear_sum_assignment

class ValidPixelQueryLoss(nn.Module):
    def forward(self, outputs, target, valid_mask):
        mask_layers, class_layers = outputs
        if len(mask_layers) != len(class_layers) or not mask_layers:
            raise ValueError('unaligned query prediction layers')
        total = mask_layers[0].sum() * 0 + class_layers[0].sum() * 0
        for masks, classes in zip(mask_layers, class_layers):
            masks = F.interpolate(masks, target.shape[-2:], mode='bilinear', align_corners=False)
            losses = []
            for b in range(target.shape[0]):
                valid = valid_mask[b].bool() & target[b].ne(255)
                if not valid.any():
                    continue  # excludes classification too: no observable target
                labels = torch.unique(target[b][valid])
                if not torch.all((labels == 0) | (labels == 1)):
                    raise ValueError('invalid operational class')
                truth = (target[b][valid][None] == labels[:, None]).float()
                pred = masks[b, :, valid]
                with torch.no_grad():
                    # BCE matrix without [Q,C,P] expansion; only valid locations.
                    cost_bce = (F.softplus(pred).mean(-1)[:, None]
                                - pred @ truth.T / truth.shape[-1])
                    prob = pred.sigmoid()
                    cost_dice = 1 - (2 * prob @ truth.T + 1) / (prob.sum(-1)[:, None] + truth.sum(-1)[None] + 1)
                    cost = -2 * classes[b].softmax(-1)[:, labels] + 5 * cost_bce + 5 * cost_dice
                    rows, cols = linear_sum_assignment(cost.cpu().numpy())
                rows = torch.as_tensor(rows, device=pred.device)
                cols = torch.as_tensor(cols, device=pred.device)
                cls_target = torch.full((classes.shape[1],), 2, dtype=torch.long, device=pred.device)
                cls_target[rows] = labels[cols]
                classification = F.cross_entropy(classes[b], cls_target,
                                                 weight=classes.new_tensor([1., 1., .1]))
                selected, expected = pred[rows], truth[cols]
                bce = F.binary_cross_entropy_with_logits(selected, expected)
                probability = selected.sigmoid()
                dice = (1 - (2 * (probability * expected).sum(-1) + 1) /
                        (probability.sum(-1) + expected.sum(-1) + 1)).mean()
                losses.append(2 * classification + 5 * bce + 5 * dice)
            if losses:
                total = total + torch.stack(losses).mean()
            else:
                total = total + masks.sum() * 0 + classes.sum() * 0
        return total
