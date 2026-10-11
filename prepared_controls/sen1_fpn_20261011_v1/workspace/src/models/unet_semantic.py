# Copyright (C) 2026 STURM output-path study contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
# Source-release notice added 2026-10-11; numerical code unchanged.
"""Compact shared U-Net baselines for a non-YOLO mechanism control."""

from __future__ import annotations

from collections.abc import Mapping

import torch
import torch.nn as nn

from .yolo26_semantic import SensorInputAdapter


class DoubleConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class SharedUNet(nn.Module):
    """Shared S1/S2 U-Net with a common 3-channel bridge."""

    def __init__(self, *, base_channels: int = 16, num_classes: int = 2, common_channels: int = 3) -> None:
        super().__init__()
        if base_channels < 4 or num_classes != 2:
            raise ValueError("base_channels must be >=4 and the project uses exactly two classes")
        self.input_adapters = SensorInputAdapter(common_channels=common_channels)
        b = base_channels
        self.enc1 = DoubleConv(common_channels, b)
        self.enc2 = DoubleConv(b, 2 * b)
        self.enc3 = DoubleConv(2 * b, 4 * b)
        self.bottleneck = DoubleConv(4 * b, 8 * b)
        self.pool = nn.MaxPool2d(2)
        self.up3 = nn.ConvTranspose2d(8 * b, 4 * b, 2, stride=2)
        self.dec3 = DoubleConv(8 * b, 4 * b)
        self.up2 = nn.ConvTranspose2d(4 * b, 2 * b, 2, stride=2)
        self.dec2 = DoubleConv(4 * b, 2 * b)
        self.up1 = nn.ConvTranspose2d(2 * b, b, 2, stride=2)
        self.dec1 = DoubleConv(2 * b, b)
        self.head = nn.Conv2d(b, num_classes, 1)

    @staticmethod
    def _sensor_name(sensor: str | int | torch.Tensor) -> str:
        if isinstance(sensor, torch.Tensor):
            values = torch.unique(sensor.detach().cpu())
            if values.numel() != 1:
                raise ValueError("a native tensor forward requires one sensor per batch")
            sensor = int(values.item())
        return SensorInputAdapter._sensor_name(sensor)

    def forward_native(self, image: torch.Tensor, sensor: str | int | torch.Tensor) -> torch.Tensor:
        name = self._sensor_name(sensor)
        x1 = self.enc1(self.input_adapters(image, name))
        x2 = self.enc2(self.pool(x1))
        x3 = self.enc3(self.pool(x2))
        xb = self.bottleneck(self.pool(x3))
        x = self.up3(xb)
        x = self.dec3(torch.cat([x, x3], dim=1))
        x = self.up2(x)
        x = self.dec2(torch.cat([x, x2], dim=1))
        x = self.up1(x)
        x = self.dec1(torch.cat([x, x1], dim=1))
        return self.head(x)

    def forward_grouped(self, groups: Mapping[str, Mapping[str, object]]) -> dict[str, torch.Tensor]:
        outputs: dict[str, torch.Tensor] = {}
        for sensor in ("S1", "S2"):
            if sensor in groups:
                batch = groups[sensor]
                outputs[sensor] = self.forward_native(batch["image"], sensor)  # type: ignore[arg-type]
        if not outputs:
            raise ValueError("grouped batch contains neither S1 nor S2")
        return outputs

    def forward(
        self,
        inputs: torch.Tensor | Mapping[str, object],
        sensor: str | int | torch.Tensor | None = None,
    ) -> object:
        if isinstance(inputs, Mapping):
            if "image" in inputs:
                if sensor is None:
                    sensor = inputs.get("sensor")  # type: ignore[assignment]
                    if isinstance(sensor, list):
                        if not sensor or len(set(sensor)) != 1:
                            raise ValueError("a native mapping forward requires one sensor per batch")
                        sensor = sensor[0]
                if sensor is None:
                    raise ValueError("sensor is required for a native mapping forward")
                return self.forward_native(inputs["image"], sensor)  # type: ignore[arg-type]
            return self.forward_grouped(inputs)  # type: ignore[arg-type]
        if sensor is None:
            raise ValueError("sensor is required when forwarding a native tensor")
        return self.forward_native(inputs, sensor)


class _PaperConvBlock(nn.Module):
    """PyTorch translation of the two-convolution STURM-Flood U-Net block."""

    def __init__(self, in_channels: int, out_channels: int, *, norm: bool, dropout: float) -> None:
        super().__init__()
        # Keras BatchNormalization defaults are momentum=0.99 and epsilon=1e-3.
        # PyTorch defines momentum as the *new-batch* contribution, therefore
        # the equivalent update is momentum=0.01 rather than 0.99.
        self.norm = (
            nn.BatchNorm2d(in_channels, eps=1e-3, momentum=0.01)
            if norm
            else nn.Identity()
        )
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        for convolution in (self.conv1, self.conv2):
            # Match the registered paper-replica policy for ReLU convolutions.
            # Biases are explicit because Keras' bias initializer is zero.
            nn.init.kaiming_normal_(convolution.weight, mode="fan_in", nonlinearity="relu")
            nn.init.zeros_(convolution.bias)
        self.activation = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.norm(x)
        x = self.activation(self.conv1(x))
        x = self.activation(self.conv2(x))
        return self.dropout(x)


class SturmPaperUNet(nn.Module):
    """Paper-inspired five-level U-Net with the project's grouped sensor API.

    The upstream STURM-Flood implementation is TensorFlow/Keras and uses a
    five-block U-Net with two convolutions per block, transposed-convolution
    upsampling, ReLU activations and contracting-path dropout.  This implementation preserves
    that topology while retaining the project's grouped API and
    valid-pixel-aware PyTorch training contract.  ``native_input=True`` skips
    the project-specific sensor bridge so an independent S1 or S2 instance
    matches the paper's direct native-band input more closely.  It is
    intentionally exposed as a separate class so the compact mechanism-control
    U-Net remains unchanged.  The formal replica defaults to dropout=0.30,
    Kaiming-normal ReLU convolutions, and Xavier-uniform upsampling/classifier
    layers, all with explicit zero biases.
    """

    def __init__(
        self,
        *,
        base_channels: int = 32,
        num_classes: int = 2,
        common_channels: int = 3,
        dropout: float = 0.30,
        native_input: bool = False,
        use_batchnorm: bool = True,
    ) -> None:
        super().__init__()
        if base_channels < 4 or num_classes != 2 or common_channels < 1:
            raise ValueError("base_channels must be >=4 and the project uses exactly two classes")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        self.native_input = bool(native_input)
        # Native paper replicas must not carry an unused project sensor
        # bridge.  Besides adding dead parameters, retaining it made the S1
        # and S2 parameter counts differ for an implementation detail that is
        # never executed.  The shared mechanism-control U-Net still receives
        # the adapter when ``native_input`` is false.
        self.input_adapters = (
            None if self.native_input else SensorInputAdapter(common_channels=common_channels)
        )
        self.pool = nn.MaxPool2d(2)
        widths = [base_channels * (2**i) for i in range(5)]
        self.encoders = nn.ModuleList(
            [
                _PaperConvBlock(common_channels, widths[0], norm=False, dropout=dropout),
                _PaperConvBlock(widths[0], widths[1], norm=use_batchnorm, dropout=dropout),
                _PaperConvBlock(widths[1], widths[2], norm=use_batchnorm, dropout=dropout),
                _PaperConvBlock(widths[2], widths[3], norm=use_batchnorm, dropout=dropout),
            ]
        )
        self.bottleneck = _PaperConvBlock(widths[3], widths[4], norm=False, dropout=dropout)
        self.upconvs = nn.ModuleList(
            [
                nn.ConvTranspose2d(widths[4], widths[3], 3, stride=2, padding=1, output_padding=1),
                nn.ConvTranspose2d(widths[3], widths[2], 3, stride=2, padding=1, output_padding=1),
                nn.ConvTranspose2d(widths[2], widths[1], 3, stride=2, padding=1, output_padding=1),
                nn.ConvTranspose2d(widths[1], widths[0], 3, stride=2, padding=1, output_padding=1),
            ]
        )
        for upconv in self.upconvs:
            nn.init.xavier_uniform_(upconv.weight)
            nn.init.zeros_(upconv.bias)
        self.decoders = nn.ModuleList(
            [
                # The upstream STURM-Flood upsampling_block has no Dropout;
                # dropout is confined to conv_block on the contracting path
                # and bottleneck.
                _PaperConvBlock(widths[4], widths[3], norm=use_batchnorm, dropout=0.0),
                _PaperConvBlock(widths[3], widths[2], norm=use_batchnorm, dropout=0.0),
                _PaperConvBlock(widths[2], widths[1], norm=use_batchnorm, dropout=0.0),
                _PaperConvBlock(widths[1], widths[0], norm=False, dropout=0.0),
            ]
        )
        self.head = nn.Conv2d(widths[0], num_classes, 1)
        nn.init.xavier_uniform_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    @staticmethod
    def _sensor_name(sensor: str | int | torch.Tensor) -> str:
        if isinstance(sensor, torch.Tensor):
            values = torch.unique(sensor.detach().cpu())
            if values.numel() != 1:
                raise ValueError("a native tensor forward requires one sensor per batch")
            sensor = int(values.item())
        return SensorInputAdapter._sensor_name(sensor)

    def forward_native(self, image: torch.Tensor, sensor: str | int | torch.Tensor) -> torch.Tensor:
        name = self._sensor_name(sensor)
        if self.native_input:
            x = image
        else:
            if self.input_adapters is None:
                raise RuntimeError("non-native U-Net forward requires an input adapter")
            x = self.input_adapters(image, name)
        skips = []
        for encoder in self.encoders:
            x = encoder(x)
            skips.append(x)
            x = self.pool(x)
        x = self.bottleneck(x)
        for upconv, decoder, skip in zip(self.upconvs, self.decoders, reversed(skips)):
            x = upconv(x)
            if x.shape[-2:] != skip.shape[-2:]:
                x = torch.nn.functional.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            x = decoder(torch.cat([x, skip], dim=1))
        return self.head(x)

    def forward_grouped(self, groups: Mapping[str, Mapping[str, object]]) -> dict[str, torch.Tensor]:
        outputs: dict[str, torch.Tensor] = {}
        for sensor in ("S1", "S2"):
            if sensor in groups:
                batch = groups[sensor]
                outputs[sensor] = self.forward_native(batch["image"], sensor)  # type: ignore[arg-type]
        if not outputs:
            raise ValueError("grouped batch contains neither S1 nor S2")
        return outputs

    def forward(
        self,
        inputs: torch.Tensor | Mapping[str, object],
        sensor: str | int | torch.Tensor | None = None,
    ) -> object:
        if isinstance(inputs, Mapping):
            if "image" in inputs:
                if sensor is None:
                    sensor = inputs.get("sensor")  # type: ignore[assignment]
                    if isinstance(sensor, list):
                        if not sensor or len(set(sensor)) != 1:
                            raise ValueError("a native mapping forward requires one sensor per batch")
                        sensor = sensor[0]
                if sensor is None:
                    raise ValueError("sensor is required for a native mapping forward")
                return self.forward_native(inputs["image"], sensor)  # type: ignore[arg-type]
            return self.forward_grouped(inputs)  # type: ignore[arg-type]
        if sensor is None:
            raise ValueError("sensor is required when forwarding a native tensor")
        return self.forward_native(inputs, sensor)


class IndependentNativeSturmUNet(nn.Module):
    """Two independent paper-topology U-Nets for native S1 and S2 inputs."""

    def __init__(
        self,
        *,
        base_channels: int = 64,
        num_classes: int = 2,
        dropout: float = 0.30,
        use_batchnorm: bool = True,
    ) -> None:
        super().__init__()
        self.models = nn.ModuleDict(
            {
                "S1": SturmPaperUNet(
                    base_channels=base_channels,
                    num_classes=num_classes,
                    common_channels=2,
                    native_input=True,
                    dropout=dropout,
                    use_batchnorm=use_batchnorm,
                ),
                "S2": SturmPaperUNet(
                    base_channels=base_channels,
                    num_classes=num_classes,
                    common_channels=9,
                    native_input=True,
                    dropout=dropout,
                    use_batchnorm=use_batchnorm,
                ),
            }
        )

    @staticmethod
    def _sensor_name(sensor: str | int | torch.Tensor) -> str:
        return SturmPaperUNet._sensor_name(sensor)

    def forward_native(self, image: torch.Tensor, sensor: str | int | torch.Tensor) -> torch.Tensor:
        name = self._sensor_name(sensor)
        return self.models[name].forward_native(image, name)

    def forward(self, inputs: torch.Tensor | Mapping[str, object], sensor=None) -> object:
        if isinstance(inputs, Mapping):
            if "image" not in inputs:
                return {
                    name: self.forward(batch) for name, batch in inputs.items()
                }
            sensor = sensor if sensor is not None else inputs.get("sensor")
            if isinstance(sensor, list):
                if not sensor or len(set(sensor)) != 1:
                    raise ValueError("native U-Net batch must contain one sensor")
                sensor = sensor[0]
            if sensor is None:
                raise ValueError("sensor is required for native U-Net input")
            return self.forward_native(inputs["image"], sensor)  # type: ignore[arg-type]
        if sensor is None:
            raise ValueError("sensor is required for native U-Net tensor")
        return self.forward_native(inputs, sensor)
