# Copyright (C) 2026 STURM output-path study contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
# Source-release notice added 2026-10-11; numerical code unchanged.
"""EoMT query loss and optimization bridge for the existing STURM engine.

Dense valid-pixel Hungarian objective; deliberately not the official point
sampler. Mask losses normalize by batch matched-mask count, as upstream.
"""
import importlib.util
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F
from scipy.optimize import linear_sum_assignment

class EoMTProtocolLoss(nn.Module):
    ignore_index = 255

    def forward(self, outputs, target, valid_mask):
        masks_all, classes_all = outputs['query_masks'], outputs['query_classes']
        if len(masks_all) != len(classes_all) or not masks_all:
            raise ValueError('unaligned query layers')
        total = masks_all[0].sum() * 0 + classes_all[0].sum() * 0
        components = {'class': total, 'bce': total, 'dice': total}
        valid = valid_mask.bool() & target.ne(self.ignore_index)
        for masks, classes in zip(masks_all, classes_all):
            if not torch.isfinite(masks).all() or not torch.isfinite(classes).all():
                raise FloatingPointError('nonfinite query outputs')
            masks = F.interpolate(masks, target.shape[-2:], mode='bilinear', align_corners=False)
            class_preds, class_targets, bces, dices = [], [], [], []
            for b in range(target.shape[0]):
                if not valid[b].any():
                    continue
                labels = torch.unique(target[b][valid[b]])
                if not torch.all((labels == 0) | (labels == 1)):
                    raise ValueError('invalid operational target')
                truth = (target[b][valid[b]][None] == labels[:, None]).float()
                pred = masks[b, :, valid[b]]
                with torch.no_grad():
                    prob = pred.sigmoid()
                    cost_bce = F.softplus(pred).mean(-1)[:, None] - pred @ truth.T / truth.shape[-1]
                    cost_dice = 1 - (2 * prob @ truth.T + 1) / (prob.sum(-1)[:, None] + truth.sum(-1)[None] + 1)
                    rows, cols = linear_sum_assignment((-2 * classes[b].softmax(-1)[:, labels] + 5 * cost_bce + 5 * cost_dice).cpu().numpy())
                rows = torch.as_tensor(rows, device=pred.device)
                cols = torch.as_tensor(cols, device=pred.device)
                cls_target = torch.full((classes.shape[1],), 2, dtype=torch.long, device=pred.device)
                cls_target[rows] = labels[cols]
                class_preds.append(classes[b])
                class_targets.append(cls_target)
                selected, expected = pred[rows], truth[cols]
                bces.append(F.binary_cross_entropy_with_logits(selected, expected, reduction='none').mean(-1))
                probability = selected.sigmoid()
                dices.append(1 - (2 * (probability * expected).sum(-1) + 1) /
                             (probability.sum(-1) + expected.sum(-1) + 1))
            if class_preds:
                classification = F.cross_entropy(torch.cat(class_preds), torch.cat(class_targets),
                                                  weight=classes.new_tensor([1., 1., .1]))
                bce, dice = torch.cat(bces).mean(), torch.cat(dices).mean()
                total = total + 2 * classification + 5 * bce + 5 * dice
                for key, value in [('class', classification), ('bce', bce), ('dice', dice)]:
                    components[key] = components[key] + value
            else:
                total = total + masks.sum() * 0 + classes.sum() * 0
        details = {key + '_loss': value.detach() for key, value in components.items()}
        details['valid_pixels'] = valid.sum().detach()
        details['total_loss'] = total.detach()
        return total, details

def build_eomt_optimizer(model, *, lr=1e-4, weight_decay=.05, llrd=.8):
    backbone, others = [], []
    depths = len(model.core.encoder.backbone.blocks)
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if name.startswith('core.encoder.backbone.'):
            if '.blocks.' in name:
                depth = int(name.split('.blocks.')[1].split('.')[0])
                multiplier = llrd ** (depths - 1 - depth)
            elif name.startswith('core.encoder.backbone.norm.'):
                multiplier = 1.
            else:
                multiplier = llrd ** (depths - 1)
            backbone.append({'params': [p], 'lr': lr * multiplier, 'name': name})
        else:
            others.append({'params': [p], 'lr': lr, 'name': name})
    return torch.optim.AdamW(backbone + others, weight_decay=weight_decay), len(backbone)

class AttentionPolySchedule:
    """Unchanged upstream LR schedule plus separately declared mask annealing."""
    def __init__(self, optimizer, model, *, backbone_groups, total_steps, warmup_steps):
        from src.models.eomt_s_comparator import ROOT
        path = ROOT / 'external/eomt-official/training/two_stage_warmup_poly_schedule.py'
        import hashlib
        import json
        lock = json.loads((ROOT / 'configs/eomt_official_source_lock.json').read_text())
        if hashlib.sha256(path.read_bytes()).hexdigest() != lock['source_files_sha256']['training/two_stage_warmup_poly_schedule.py']:
            raise RuntimeError('official EoMT scheduler source modified')
        spec = importlib.util.spec_from_file_location('_eomt_official_schedule', path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        self.lr_schedule = module.TwoStageWarmupPolySchedule(optimizer, backbone_groups, tuple(warmup_steps), total_steps, .9)
        self.model = model
        self.total_steps = total_steps
        self.update_attention()

    def update_attention(self):
        progress = self.lr_schedule.last_epoch / self.total_steps
        # Protocol adaptation: progressive removal ends before final training.
        values = [max(0., min(1., (end - progress) / (end - start)))
                  for start, end in zip((.1, .3, .5), (.3, .5, .7))]
        self.model.core.attn_mask_probs.copy_(torch.tensor(values, device=self.model.core.attn_mask_probs.device))

    def step(self):
        self.lr_schedule.step()
        self.update_attention()

    def state_dict(self):
        return {'version': 'eomt_attn_poly_v1', 'total_steps': self.total_steps, 'lr_schedule': self.lr_schedule.state_dict()}

    def load_state_dict(self, state):
        if state['version'] != 'eomt_attn_poly_v1' or state['total_steps'] != self.total_steps:
            raise ValueError('EoMT scheduler contract mismatch')
        self.lr_schedule.load_state_dict(state['lr_schedule'])
        self.update_attention()

def restore_eomt_checkpoint(path, model, optimizer, scheduler, train_loader, *, seed, max_epochs, version):
    """CPU-stage optimizer state and restore RNG; never constructs a dataset."""
    state = torch.load(path, map_location='cpu', weights_only=False)
    if (state['seed'] != seed or state['max_epochs'] != max_epochs or
            state['optimizer_spec_version'] != version):
        raise ValueError('EoMT checkpoint identity/budget mismatch')
    expected_names = [g.get('name') for g in optimizer.param_groups]
    if [g.get('name') for g in state['optimizer_state_dict']['param_groups']] != expected_names:
        raise ValueError('EoMT optimizer group mismatch')
    model.load_state_dict(state['model_state_dict'], strict=True)
    optimizer.load_state_dict(state['optimizer_state_dict'])
    scheduler.load_state_dict(state['scheduler_state_dict'])
    from scripts.server_main import _restore_rng_state
    _restore_rng_state(state['rng_state'], train_loader)
    return state
