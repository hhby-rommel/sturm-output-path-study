# Copyright (C) 2026 STURM output-path study contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
# Source-release notice added 2026-10-11; numerical code unchanged.
"""Isolated, explicitly selected research comparators.

These classes are not registered in the formal runner. No constructor reaches
the network or silently substitutes a different architecture.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from .optional_comparison import _GroupedSensorModel
from .yolo26_semantic import SensorInputAdapter


def _conv_bn_relu(in_channels: int, out_channels: int, kernel_size: int,
                  *, dilation: int = 1) -> nn.Sequential:
    padding = ((kernel_size - 1) // 2) * dilation
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding,
                  dilation=dilation, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
    )


class _ASPP(nn.Module):
    def __init__(self, in_channels: int, out_channels: int = 256) -> None:
        super().__init__()
        self.branches = nn.ModuleList([
            _conv_bn_relu(in_channels, out_channels, 1),
            *(_conv_bn_relu(in_channels, out_channels, 3, dilation=rate)
              for rate in (6, 12, 18)),
        ])
        self.pool = nn.Sequential(nn.AdaptiveAvgPool2d(1),
                                  _conv_bn_relu(in_channels, out_channels, 1))
        self.project = nn.Sequential(
            _conv_bn_relu(5 * out_channels, out_channels, 1), nn.Dropout(0.1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        size = x.shape[-2:]
        features = [branch(x) for branch in self.branches]
        pooled = self.pool(x)
        features.append(F.interpolate(pooled, size=size, mode="bilinear",
                                      align_corners=False))
        return self.project(torch.cat(features, dim=1))


class DeepLabV3PlusResNet50(_GroupedSensorModel):
    """DeepLabV3+ with a dilated ResNet-50 encoder (output stride 16).

    The low-level 1/4-resolution feature is fused with ASPP output before
    prediction. A torchvision DeepLabV3 model lacks this decoder and must not
    be reported as DeepLabV3+.
    """

    architecture_name = "deeplabv3plus_resnet50_os16"

    def __init__(self, *, common_channels: int = 3, num_classes: int = 2,
                 backbone_weights_path: str | Path | None = None) -> None:
        super().__init__(common_channels=common_channels)
        if common_channels != 3:
            raise ValueError("ResNet-50 backbone requires common_channels=3")
        try:
            from torchvision.models import resnet50
        except Exception as exc:
            raise ImportError("torchvision ResNet-50 is required") from exc
        self.encoder = resnet50(weights=None,
                                replace_stride_with_dilation=[False, False, True])
        self.pretraining = "scratch"
        if backbone_weights_path is not None:
            path = Path(backbone_weights_path)
            if not path.is_file():
                raise FileNotFoundError(path)
            state = torch.load(path, map_location="cpu", weights_only=True)
            if not isinstance(state, dict):
                raise ValueError("backbone checkpoint must be a state_dict")
            self.encoder.load_state_dict(state, strict=True)
            self.pretraining = str(path.resolve())
        self.aspp = _ASPP(2048)
        self.low_project = _conv_bn_relu(256, 48, 1)
        self.decoder = nn.Sequential(
            _conv_bn_relu(304, 256, 3),
            _conv_bn_relu(256, 256, 3),
            nn.Conv2d(256, num_classes, 1),
        )

    def forward_native(self, image: torch.Tensor,
                       sensor: str | int | torch.Tensor) -> torch.Tensor:
        x = self.input_adapters(image, self._sensor_name(sensor))
        encoder = self.encoder
        x = encoder.maxpool(encoder.relu(encoder.bn1(encoder.conv1(x))))
        low = encoder.layer1(x)
        x = encoder.layer4(encoder.layer3(encoder.layer2(low)))
        high = self.aspp(x)
        high = F.interpolate(high, size=low.shape[-2:], mode="bilinear",
                             align_corners=False)
        logits = self.decoder(torch.cat([high, self.low_project(low)], dim=1))
        return F.interpolate(logits, size=image.shape[-2:], mode="bilinear",
                             align_corners=False)


class DeepLabV3PlusMobileNetV3Large(_GroupedSensorModel):
    """DeepLabV3+ / MobileNetV3-Large with output stride 16.

    ``torchvision`` supplies only the MobileNetV3 encoder. This class adds
    its own ASPP and 1/4-scale low-feature decoder, which distinguishes the
    plus architecture from torchvision's DeepLabV3-MobileNetV3 model.
    """

    architecture_name = "deeplabv3plus_mobilenet_v3_large_os16"

    def __init__(self, *, common_channels: int = 3, num_classes: int = 2,
                 backbone_weights_path: str | Path | None = None) -> None:
        super().__init__(common_channels=common_channels)
        if common_channels != 3:
            raise ValueError("MobileNetV3-Large backbone requires common_channels=3")
        try:
            from torchvision.models import mobilenet_v3_large
        except Exception as exc:
            raise ImportError("torchvision MobileNetV3-Large is required") from exc
        backbone = mobilenet_v3_large(weights=None, dilated=True)
        self.pretraining = "scratch"
        if backbone_weights_path is not None:
            path = Path(backbone_weights_path)
            if not path.is_file():
                raise FileNotFoundError(path)
            state = torch.load(path, map_location="cpu", weights_only=True)
            if not isinstance(state, dict):
                raise ValueError("backbone checkpoint must be a state_dict")
            backbone.load_state_dict(state, strict=True)
            self.pretraining = str(path.resolve())
        self.encoder = backbone.features
        if (self.encoder[3].out_channels, self.encoder[16].out_channels) != (24, 960):
            raise ValueError("unexpected MobileNetV3-Large encoder feature widths")
        self.aspp = _ASPP(960)
        self.low_project = _conv_bn_relu(24, 48, 1)
        self.decoder = nn.Sequential(
            _conv_bn_relu(304, 256, 3),
            _conv_bn_relu(256, 256, 3),
            nn.Conv2d(256, num_classes, 1),
        )

    def extract_features(self, common_image: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        low: torch.Tensor | None = None
        features = common_image
        for index, layer in enumerate(self.encoder):
            features = layer(features)
            if index == 3:
                low = features
        assert low is not None
        return low, features

    def forward_native(self, image: torch.Tensor,
                       sensor: str | int | torch.Tensor) -> torch.Tensor:
        common = self.input_adapters(image, self._sensor_name(sensor))
        low, high = self.extract_features(common)
        high = self.aspp(high)
        high = F.interpolate(high, size=low.shape[-2:], mode="bilinear",
                             align_corners=False)
        logits = self.decoder(torch.cat((high, self.low_project(low)), dim=1))
        return F.interpolate(logits, size=image.shape[-2:], mode="bilinear",
                             align_corners=False)


class SegFormerB0(_GroupedSensorModel):
    """Official Hugging Face SegFormer-B0, with MiT-B0 self-attention.

    ``pretrained_dir`` must be a local Transformers model directory. It may
    contain a different segmentation head, which is deliberately reset to two
    classes; all encoder weights must load without mismatch.
    """

    architecture_name = "segformer_b0_mit_b0"
    _depths = (2, 2, 2, 2)
    _widths = (32, 64, 160, 256)
    _heads = (1, 2, 5, 8)
    _sr_ratios = (8, 4, 2, 1)

    def __init__(self, *, common_channels: int = 3, num_classes: int = 2,
                 pretrained_dir: str | Path | None = None) -> None:
        super().__init__(common_channels=common_channels)
        if common_channels != 3:
            raise ValueError("MiT-B0 checkpoint interface requires common_channels=3")
        try:
            from transformers import SegformerConfig, SegformerForSemanticSegmentation
        except Exception as exc:
            raise ImportError("Install transformers to use the real SegFormer-B0; "
                              "SegFormerB0Like is not a substitute") from exc
        self.pretraining = "scratch"
        if pretrained_dir is None:
            config = SegformerConfig(num_labels=num_classes,
                                     depths=list(self._depths),
                                     hidden_sizes=list(self._widths),
                                     num_attention_heads=list(self._heads),
                                     sr_ratios=list(self._sr_ratios),
                                     decoder_hidden_size=256)
            self.core = SegformerForSemanticSegmentation(config)
        else:
            path = Path(pretrained_dir)
            if not path.is_dir():
                raise FileNotFoundError(path)
            self.core, load_info = SegformerForSemanticSegmentation.from_pretrained(
                str(path), local_files_only=True, num_labels=num_classes,
                ignore_mismatched_sizes=True, output_loading_info=True)
            acceptable = "decode_head.classifier"
            unexpected = [str(item) for field in ("missing_keys", "unexpected_keys",
                                                "mismatched_keys")
                          for item in load_info.get(field, [])
                          if acceptable not in str(item)]
            if unexpected:
                raise ValueError("pretrained SegFormer has non-head weight mismatches: "
                                 + "; ".join(unexpected[:8]))
            self.pretraining = str(path.resolve())
        self._verify_identity(num_classes)

    def _verify_identity(self, num_classes: int) -> None:
        config = self.core.config
        expected = (("depths", self._depths), ("hidden_sizes", self._widths),
                    ("num_attention_heads", self._heads),
                    ("sr_ratios", self._sr_ratios),
                    ("patch_sizes", (7, 3, 3, 3)),
                    ("strides", (4, 2, 2, 2)))
        bad = [name for name, values in expected
               if tuple(getattr(config, name)) != values]
        if bad or config.decoder_hidden_size != 256 or config.num_labels != num_classes:
            raise ValueError(f"checkpoint is not SegFormer-B0: {bad}")
        attention = [module for module in self.core.segformer.modules()
                     if type(module).__name__ == "SegformerEfficientSelfAttention"]
        if len(attention) != sum(self._depths):
            raise ValueError("MiT-B0 self-attention blocks are absent or incomplete")

    def forward_native(self, image: torch.Tensor,
                       sensor: str | int | torch.Tensor) -> torch.Tensor:
        common = self.input_adapters(image, self._sensor_name(sensor))
        logits = self.core(pixel_values=common).logits
        return F.interpolate(logits, size=image.shape[-2:], mode="bilinear",
                             align_corners=False)


class _CPFDoubleConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True),
        )

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return self.net(image)


class _CPFChannelAttention(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        hidden = max(channels // 16, 4)
        self.mlp = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Conv2d(channels, hidden, 1, bias=False),
            nn.ReLU(inplace=True), nn.Conv2d(hidden, channels, 1, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return features * self.mlp(features)


class _CPFSpatialAttention(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(2, 1, 7, padding=3, bias=False)
        self.act = nn.Sigmoid()

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        statistics = torch.cat((features.mean(dim=1, keepdim=True),
                                features.amax(dim=1, keepdim=True)), dim=1)
        return features * self.act(self.conv(statistics))


class _CPFCrossGating(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.gate = nn.Sequential(
            nn.Conv2d(2 * channels, channels, 1, bias=False),
            nn.BatchNorm2d(channels), nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 1, bias=False), nn.Sigmoid(),
        )

    def forward(self, source: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return target * self.gate(torch.cat((source, target), dim=1))


class _CPFBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.ca_vv = _CPFChannelAttention(channels)
        self.ca_vh = _CPFChannelAttention(channels)
        self.sa_vv = _CPFSpatialAttention()
        self.sa_vh = _CPFSpatialAttention()
        self.gate_vv_to_vh = _CPFCrossGating(channels)
        self.gate_vh_to_vv = _CPFCrossGating(channels)
        self.proj = nn.Sequential(nn.Conv2d(2 * channels, channels, 1, bias=False),
                                  nn.BatchNorm2d(channels), nn.ReLU(inplace=True))

    def forward(self, vv: torch.Tensor, vh: torch.Tensor) -> torch.Tensor:
        vv_attended = self.sa_vv(self.ca_vv(vv))
        vh_attended = self.sa_vh(self.ca_vh(vh))
        vh_gated = self.gate_vv_to_vh(vv_attended, vh_attended)
        vv_gated = self.gate_vh_to_vv(vh_attended, vv_attended)
        return self.proj(torch.cat((vv_gated, vh_gated), dim=1))


class CPFUNetS1(nn.Module):
    """Talreja et al. (2026) CPF fusion plus U-Net, adapted to two raw logits.

    This is unrelated to Sun and Qu's 2024 panoramic-image CPF-UNet.

    Structure follows the author's CPFBlock and CPFUNet at commit
    32bd74124c9d713dad64d5b2ff085d536e766d01. The original final
    sigmoid is omitted: [0, z] has softmax class-1 probability sigmoid(z).
    """

    architecture_name = "cpf_unet_s1_2026"
    source_commit = "32bd74124c9d713dad64d5b2ff085d536e766d01"

    def __init__(self, *, base_channels: int = 64) -> None:
        super().__init__()
        if base_channels < 1:
            raise ValueError("base_channels must be positive")
        c = base_channels
        self.stem_vv = _CPFDoubleConv(1, c)
        self.stem_vh = _CPFDoubleConv(1, c)
        self.cpf = _CPFBlock(c)
        self.pool = nn.MaxPool2d(2)
        self.enc2 = _CPFDoubleConv(c, 2 * c)
        self.enc3 = _CPFDoubleConv(2 * c, 4 * c)
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.dec3 = _CPFDoubleConv(6 * c, 2 * c)
        self.dec2 = _CPFDoubleConv(3 * c, c)
        self.out = nn.Conv2d(c, 1, 1)

    def forward_logit(self, image: torch.Tensor) -> torch.Tensor:
        if image.ndim != 4 or image.shape[1] != 2:
            raise ValueError("CPF-UNet requires S1 image [B,2,H,W] in [VV,VH] order")
        vv = self.stem_vv(image[:, 0:1])
        vh = self.stem_vh(image[:, 1:2])
        first = self.cpf(vv, vh)
        second = self.enc2(self.pool(first))
        third = self.enc3(self.pool(second))
        decoded = self.dec3(torch.cat((self.up(third), second), dim=1))
        decoded = self.dec2(torch.cat((self.up(decoded), first), dim=1))
        return self.out(decoded)

    def forward_native(self, image: torch.Tensor,
                       sensor: str | int | torch.Tensor) -> torch.Tensor:
        if SensorInputAdapter._sensor_name(sensor) != "S1":
            raise ValueError("CPF-UNet is an S1-only VV/VH comparator; S2 is unsupported")
        positive_logit = self.forward_logit(image)
        return torch.cat((torch.zeros_like(positive_logit), positive_logit), dim=1)

    def forward_grouped(self, groups: Mapping[str, Mapping[str, object]]) -> dict[str, torch.Tensor]:
        if "S2" in groups:
            raise ValueError("CPF-UNet cannot receive an S2 group")
        if "S1" not in groups:
            raise ValueError("CPF-UNet requires an S1 group")
        return {"S1": self.forward_native(groups["S1"]["image"], "S1")}

    def forward(self, inputs: torch.Tensor | Mapping[str, object],
                sensor: str | int | torch.Tensor | None = None) -> object:
        if isinstance(inputs, Mapping):
            if "image" not in inputs:
                return self.forward_grouped(inputs)  # type: ignore[arg-type]
            if sensor is None:
                sensor = inputs.get("sensor")  # type: ignore[assignment]
                if isinstance(sensor, list):
                    if not sensor or len(set(sensor)) != 1:
                        raise ValueError("one S1 sensor per native batch is required")
                    sensor = sensor[0]
            if sensor is None:
                raise ValueError("sensor is required")
            return self.forward_native(inputs["image"], sensor)  # type: ignore[arg-type]
        if sensor is None:
            raise ValueError("sensor is required")
        return self.forward_native(inputs, sensor)
