"""PyTorch Dataset for zombie detection.

Supports multiple target formats for different model families:
- 'bbox': list of dicts with 'boxes' and 'labels' tensors (for RCNN/DETR)
- 'heatmap': 2D Gaussian heatmap at zombie centers (for heatmap CNN / ResNet head)
- 'yolo': normalized [cx, cy, w, h] per box (for YOLO export, not used at runtime)
"""

from pathlib import Path
from typing import Optional, Callable, Literal

import numpy as np
import torch
from torch.utils.data import Dataset


TargetMode = Literal["bbox", "heatmap", "yolo"]


class ZombieDetectionDataset(Dataset):
    """Load *_obs.npy / *_zombies.npy pairs from a split directory."""

    def __init__(
        self,
        data_dir: str | Path,
        target_mode: TargetMode = "bbox",
        transform: Optional[Callable] = None,
        heatmap_size: tuple[int, int] = (360, 640),
        heatmap_sigma: float = 3.0,
        screen_size: tuple[int, int] = (720, 1280),
    ):
        self.data_dir = Path(data_dir)
        self.target_mode = target_mode
        self.transform = transform
        self.heatmap_size = heatmap_size
        self.heatmap_sigma = heatmap_sigma
        self.screen_h, self.screen_w = screen_size

        obs_files = sorted(self.data_dir.glob("*_obs.npy"))
        self.samples = []
        for obs_path in obs_files:
            stem = obs_path.name.replace("_obs.npy", "")
            box_path = self.data_dir / f"{stem}_zombies.npy"
            if box_path.exists():
                self.samples.append((obs_path, box_path))

        if len(self.samples) == 0:
            raise FileNotFoundError(f"No obs/zombies pairs found in {self.data_dir}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        obs_path, box_path = self.samples[idx]
        image = np.load(str(obs_path))           # (H, W, 3) uint8
        boxes = np.load(str(box_path))            # (N, 4) float32  [x, y, w, h]

        if self.transform is not None:
            image, boxes = self.transform(image, boxes)

        img_tensor = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0

        if self.target_mode == "bbox":
            target = self._make_bbox_target(boxes, image.shape[:2])
        elif self.target_mode == "heatmap":
            target = self._make_heatmap_target(boxes, image.shape[:2])
        elif self.target_mode == "yolo":
            target = self._make_yolo_target(boxes, image.shape[:2])
        else:
            raise ValueError(f"Unknown target_mode: {self.target_mode}")

        return img_tensor, target

    def _make_bbox_target(self, boxes: np.ndarray, img_hw: tuple) -> dict:
        """Return dict with 'boxes' as [x1,y1,x2,y2] and 'labels'."""
        if len(boxes) == 0:
            return {
                "boxes": torch.zeros((0, 4), dtype=torch.float32),
                "labels": torch.zeros((0,), dtype=torch.int64),
            }
        img_h, img_w = img_hw
        scale_x = img_w / self.screen_w
        scale_y = img_h / self.screen_h

        xyxy = np.zeros((len(boxes), 4), dtype=np.float32)
        xyxy[:, 0] = boxes[:, 0] * scale_x
        xyxy[:, 1] = boxes[:, 1] * scale_y
        xyxy[:, 2] = (boxes[:, 0] + boxes[:, 2]) * scale_x
        xyxy[:, 3] = (boxes[:, 1] + boxes[:, 3]) * scale_y

        return {
            "boxes": torch.from_numpy(xyxy),
            "labels": torch.ones(len(boxes), dtype=torch.int64),
        }

    def _make_heatmap_target(self, boxes: np.ndarray, img_hw: tuple) -> torch.Tensor:
        """Gaussian heatmap at zombie centers."""
        hm_h, hm_w = self.heatmap_size
        heatmap = np.zeros((1, hm_h, hm_w), dtype=np.float32)

        if len(boxes) == 0:
            return torch.from_numpy(heatmap)

        scale_x = hm_w / self.screen_w
        scale_y = hm_h / self.screen_h
        sigma = self.heatmap_sigma

        yy, xx = np.mgrid[0:hm_h, 0:hm_w].astype(np.float32)

        for box in boxes:
            cx = (box[0] + box[2] / 2.0) * scale_x
            cy = (box[1] + box[3] / 2.0) * scale_y
            gaussian = np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * sigma ** 2))
            heatmap[0] = np.maximum(heatmap[0], gaussian)

        return torch.from_numpy(heatmap)

    def _make_yolo_target(self, boxes: np.ndarray, img_hw: tuple) -> torch.Tensor:
        """Normalized [class, cx, cy, w, h] for YOLO format."""
        if len(boxes) == 0:
            return torch.zeros((0, 5), dtype=torch.float32)

        yolo = np.zeros((len(boxes), 5), dtype=np.float32)
        yolo[:, 0] = 0  # class 0 = zombie
        yolo[:, 1] = (boxes[:, 0] + boxes[:, 2] / 2.0) / self.screen_w
        yolo[:, 2] = (boxes[:, 1] + boxes[:, 3] / 2.0) / self.screen_h
        yolo[:, 3] = boxes[:, 2] / self.screen_w
        yolo[:, 4] = boxes[:, 3] / self.screen_h

        return torch.from_numpy(yolo)


def bbox_collate_fn(batch):
    """Collate for variable-length bbox targets."""
    images = torch.stack([item[0] for item in batch])
    targets = [item[1] for item in batch]
    return images, targets


def heatmap_collate_fn(batch):
    """Standard collate for heatmap targets."""
    images = torch.stack([item[0] for item in batch])
    targets = torch.stack([item[1] for item in batch])
    return images, targets
