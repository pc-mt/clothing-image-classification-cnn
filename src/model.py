"""Compact convolutional neural network used by the project."""

from __future__ import annotations

import torch
from torch import nn


class ConvBNAct(nn.Sequential):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        groups: int = 1,
    ) -> None:
        super().__init__(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size,
                stride,
                kernel_size // 2,
                groups=groups,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(inplace=True),
        )


class InvertedResidualBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        expansion: int = 3,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        hidden_channels = in_channels * expansion
        self.use_residual = stride == 1 and in_channels == out_channels
        self.block = nn.Sequential(
            ConvBNAct(in_channels, hidden_channels, kernel_size=1),
            ConvBNAct(
                hidden_channels,
                hidden_channels,
                kernel_size=3,
                stride=stride,
                groups=hidden_channels,
            ),
            nn.Conv2d(hidden_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
        )
        self.activation = nn.SiLU(inplace=True)
        self.dropout = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        outputs = self.block(inputs)
        if self.use_residual:
            outputs = outputs + inputs
        return self.dropout(self.activation(outputs))


class ClothingCNN(nn.Module):
    """Five-class CNN trained from scratch with fewer than 800k parameters."""

    def __init__(self, num_classes: int = 5) -> None:
        super().__init__()
        self.stem = ConvBNAct(3, 32, kernel_size=3, stride=2)
        self.features = nn.Sequential(
            InvertedResidualBlock(32, 48, dropout=0.02),
            InvertedResidualBlock(48, 64, stride=2, dropout=0.03),
            InvertedResidualBlock(64, 64, dropout=0.03),
            InvertedResidualBlock(64, 96, stride=2, dropout=0.04),
            InvertedResidualBlock(96, 96, dropout=0.04),
            InvertedResidualBlock(96, 128, stride=2, dropout=0.05),
            InvertedResidualBlock(128, 128, dropout=0.05),
            InvertedResidualBlock(128, 144, dropout=0.06),
            InvertedResidualBlock(144, 176, stride=2, dropout=0.07),
            InvertedResidualBlock(176, 192, dropout=0.07),
        )
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.35),
            nn.Linear(192, 96),
            nn.SiLU(inplace=True),
            nn.Dropout(0.25),
            nn.Linear(96, num_classes),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.pool(self.features(self.stem(inputs))))


def count_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
