"""Pretrained ResNet backbone with a heatmap detection head.

Uses a ResNet encoder (ImageNet-pretrained, optionally frozen) followed
by upsampling layers that produce a single-channel heatmap, reusing
the same peak-to-box post-processing as heatmap_cnn.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class ResNetDetector(nn.Module):
    """ResNet backbone + lightweight decoder for heatmap regression."""

    def __init__(
        self,
        backbone: str = "resnet18",
        pretrained: bool = True,
        freeze_backbone: bool = False,
        in_channels: int = 3,
        **kwargs,
    ):
        super().__init__()

        weights = "IMAGENET1K_V1" if pretrained else None

        if backbone == "resnet18":
            base = models.resnet18(weights=weights)
            encoder_out = 512
        elif backbone == "resnet34":
            base = models.resnet34(weights=weights)
            encoder_out = 512
        elif backbone == "resnet50":
            base = models.resnet50(weights=weights)
            encoder_out = 2048
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")

        # If input has != 3 channels, replace first conv
        if in_channels != 3:
            base.conv1 = nn.Conv2d(
                in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False
            )

        self.encoder = nn.Sequential(
            base.conv1, base.bn1, base.relu, base.maxpool,
            base.layer1, base.layer2, base.layer3, base.layer4,
        )

        if freeze_backbone:
            for param in self.encoder.parameters():
                param.requires_grad = False

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(encoder_out, 256, 4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, 16, 4, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 1, 1),
        )

    def forward(self, x):
        input_h, input_w = x.shape[2], x.shape[3]
        features = self.encoder(x)
        heatmap = self.decoder(features)
        # Resize to exact input dimensions
        heatmap = F.interpolate(heatmap, size=(input_h, input_w), mode="bilinear", align_corners=False)
        return torch.sigmoid(heatmap)
