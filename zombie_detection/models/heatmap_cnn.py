"""Encoder-decoder CNN that outputs a zombie center heatmap.

The heatmap is a single-channel image where each zombie's center
produces a Gaussian peak. At inference, peaks are extracted and
converted to fixed-size bounding boxes.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from skimage.feature import peak_local_max


class _EncoderBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        feat = self.conv(x)
        return self.pool(feat), feat


class _DecoderBlock(nn.Module):
    def __init__(self, in_ch, skip_ch, out_ch):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, out_ch, 2, stride=2)
        self.conv = nn.Sequential(
            nn.Conv2d(out_ch + skip_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x, skip):
        x = self.up(x)
        # Handle size mismatches from odd dimensions
        dh = skip.size(2) - x.size(2)
        dw = skip.size(3) - x.size(3)
        x = F.pad(x, [0, dw, 0, dh])
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


class HeatmapCNN(nn.Module):
    """U-Net style encoder-decoder for heatmap regression."""

    def __init__(self, in_channels: int = 3, base_filters: int = 32, **kwargs):
        super().__init__()
        f = base_filters
        self.enc1 = _EncoderBlock(in_channels, f)
        self.enc2 = _EncoderBlock(f, f * 2)
        self.enc3 = _EncoderBlock(f * 2, f * 4)
        self.enc4 = _EncoderBlock(f * 4, f * 8)

        self.bottleneck = nn.Sequential(
            nn.Conv2d(f * 8, f * 16, 3, padding=1),
            nn.BatchNorm2d(f * 16),
            nn.ReLU(inplace=True),
            nn.Conv2d(f * 16, f * 16, 3, padding=1),
            nn.BatchNorm2d(f * 16),
            nn.ReLU(inplace=True),
        )

        self.dec4 = _DecoderBlock(f * 16, f * 8, f * 8)
        self.dec3 = _DecoderBlock(f * 8, f * 4, f * 4)
        self.dec2 = _DecoderBlock(f * 4, f * 2, f * 2)
        self.dec1 = _DecoderBlock(f * 2, f, f)

        self.head = nn.Conv2d(f, 1, 1)

    def forward(self, x):
        x, s1 = self.enc1(x)
        x, s2 = self.enc2(x)
        x, s3 = self.enc3(x)
        x, s4 = self.enc4(x)
        x = self.bottleneck(x)
        x = self.dec4(x, s4)
        x = self.dec3(x, s3)
        x = self.dec2(x, s2)
        x = self.dec1(x, s1)
        return torch.sigmoid(self.head(x))


def heatmap_to_boxes(
    heatmap: np.ndarray,
    bbox_w: int,
    bbox_h: int,
    screen_w: int,
    screen_h: int,
    threshold: float = 0.3,
    min_distance: int = 5,
) -> np.ndarray:
    """Convert a heatmap (H_hm, W_hm) to bounding boxes in screen coords.

    Returns (N, 4) array of [x, y, w, h] in original screen space.
    """
    hm_h, hm_w = heatmap.shape

    coords = peak_local_max(
        heatmap,
        min_distance=min_distance,
        threshold_abs=threshold,
    )

    if len(coords) == 0:
        return np.zeros((0, 4), dtype=np.float32)

    scale_x = screen_w / hm_w
    scale_y = screen_h / hm_h

    boxes = np.zeros((len(coords), 4), dtype=np.float32)
    for i, (r, c) in enumerate(coords):
        cx = c * scale_x
        cy = r * scale_y
        boxes[i] = [cx - bbox_w / 2, cy - bbox_h / 2, bbox_w, bbox_h]

    # Sort by confidence (heatmap value at peak), descending
    confidences = np.array([heatmap[r, c] for r, c in coords])
    order = np.argsort(-confidences)
    return boxes[order]
