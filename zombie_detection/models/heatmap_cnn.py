"""Encoder-decoder CNN that outputs a zombie center heatmap.

The heatmap is a single-channel image where each zombie's center
produces a Gaussian peak. At inference, peaks are extracted and
converted to fixed-size bounding boxes.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def _peak_local_max_torch(
    heatmap: np.ndarray,
    *,
    min_distance: int,
    threshold_abs: float,
) -> np.ndarray:
    """Return (N,2) int coords (row, col) of local maxima.

    This is a small replacement for `skimage.feature.peak_local_max` using only
    PyTorch. It performs max-pooling based local-max detection, applies an
    absolute threshold, excludes a border of `min_distance`, and then applies a
    greedy non-maximum suppression so returned peaks are at least `min_distance`
    apart (Chebyshev distance), similar to skimage defaults.
    """
    if heatmap.ndim != 2:
        raise ValueError(f"Expected 2D heatmap, got shape {heatmap.shape}")
    if min_distance < 0:
        raise ValueError("min_distance must be >= 0")

    hm = torch.as_tensor(heatmap, dtype=torch.float32)
    if hm.numel() == 0:
        return np.zeros((0, 2), dtype=np.int64)

    # Local-max via (2d+1)x(2d+1) max pool.
    d = int(min_distance)
    k = 2 * d + 1
    hm4 = hm[None, None, :, :]  # (1,1,H,W)
    pooled = F.max_pool2d(hm4, kernel_size=k, stride=1, padding=d)
    is_peak = (hm4 == pooled) & (hm4 >= float(threshold_abs))

    # Match skimage default exclude_border=True (effectively excludes a `d` border
    # when min_distance is used).
    if d > 0:
        is_peak[:, :, :d, :] = False
        is_peak[:, :, -d:, :] = False
        is_peak[:, :, :, :d] = False
        is_peak[:, :, :, -d:] = False

    coords = torch.nonzero(is_peak[0, 0], as_tuple=False)  # (M,2) [r,c]
    if coords.numel() == 0:
        return np.zeros((0, 2), dtype=np.int64)

    # Greedy NMS to ensure spacing. Sort by confidence descending.
    scores = hm[coords[:, 0], coords[:, 1]]
    order = torch.argsort(scores, descending=True)
    coords = coords[order]

    keep: list[torch.Tensor] = []
    for rc in coords:
        if not keep:
            keep.append(rc)
            continue
        kept = torch.stack(keep, dim=0)  # (K,2)
        # Chebyshev distance (max(|dr|,|dc|)) like a square footprint.
        dist = torch.max(torch.abs(kept - rc), dim=1).values
        if torch.all(dist > d):
            keep.append(rc)

    kept_coords = torch.stack(keep, dim=0).to(dtype=torch.int64)
    return kept_coords.cpu().numpy()


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

    coords = _peak_local_max_torch(
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
