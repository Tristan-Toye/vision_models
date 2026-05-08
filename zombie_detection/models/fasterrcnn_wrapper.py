"""Faster R-CNN wrapper using torchvision's pretrained model.

Fine-tunes a Faster R-CNN with ResNet-50 FPN backbone on the zombie
detection dataset. Supports freezing the backbone for transfer learning.
"""

import numpy as np
import torch
import torch.nn as nn
import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor


def _set_backbone_input_channels(model: nn.Module, in_channels: int, pretrained: bool) -> None:
    """Replace ResNet-FPN backbone conv1 when input channel count != 3."""
    if in_channels == 3:
        return
    old = model.backbone.body.conv1
    new_conv = nn.Conv2d(
        in_channels,
        old.out_channels,
        kernel_size=old.kernel_size,
        stride=old.stride,
        padding=old.padding,
        bias=False,
    )
    with torch.no_grad():
        if pretrained:
            n_copy = min(old.in_channels, in_channels)
            new_conv.weight[:, :n_copy] = old.weight[:, :n_copy]
            if in_channels > n_copy:
                mean_ch = old.weight.mean(dim=1, keepdim=True)
                new_conv.weight[:, n_copy:in_channels] = mean_ch.expand(
                    -1, in_channels - n_copy, -1, -1
                )
        else:
            nn.init.kaiming_normal_(new_conv.weight, mode="fan_out", nonlinearity="relu")
    model.backbone.body.conv1 = new_conv


def _align_rcnn_transform_norm(model: nn.Module, in_channels: int) -> None:
    """Extend Faster R-CNN ImageNet mean/std lists when backbone takes != 3 channels."""
    t = model.transform
    n = len(t.image_mean)
    if in_channels <= n:
        return
    extra = in_channels - n
    base_m = sum(t.image_mean) / n
    base_s = sum(t.image_std) / n
    t.image_mean = list(t.image_mean) + [base_m] * extra
    t.image_std = list(t.image_std) + [base_s] * extra


class FasterRCNNDetector(torch.nn.Module):
    """Faster R-CNN fine-tuned for zombie detection (2 classes: bg + zombie)."""

    def __init__(
        self,
        pretrained: bool = True,
        freeze_backbone: bool = False,
        num_classes: int = 2,
        in_channels: int = 3,
        **kwargs,
    ):
        super().__init__()

        weights = FasterRCNN_ResNet50_FPN_Weights.COCO_V1 if pretrained else None
        self.model = fasterrcnn_resnet50_fpn(weights=weights)

        in_features = self.model.roi_heads.box_predictor.cls_score.in_features
        self.model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

        _set_backbone_input_channels(self.model, in_channels, pretrained)
        _align_rcnn_transform_norm(self.model, in_channels)

        if freeze_backbone:
            for param in self.model.backbone.parameters():
                param.requires_grad = False

    def forward(self, images, targets=None):
        return self.model(images, targets)

    def predict(
        self,
        image: np.ndarray,
        conf: float = 0.5,
        device: str = "cpu",
    ) -> np.ndarray:
        """Run inference on a single image (H, W, C) uint8 with C == model input channels.

        Returns (N, 4) array of [x, y, w, h] in pixel coords.
        """
        self.eval()
        img_tensor = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
        img_tensor = img_tensor.to(device)

        with torch.no_grad():
            outputs = self.model([img_tensor])

        pred = outputs[0]
        keep = pred["scores"] >= conf
        boxes_xyxy = pred["boxes"][keep].cpu().numpy()
        scores = pred["scores"][keep].cpu().numpy()

        if len(boxes_xyxy) == 0:
            return np.zeros((0, 4), dtype=np.float32)

        boxes = np.zeros((len(boxes_xyxy), 4), dtype=np.float32)
        boxes[:, 0] = boxes_xyxy[:, 0]
        boxes[:, 1] = boxes_xyxy[:, 1]
        boxes[:, 2] = boxes_xyxy[:, 2] - boxes_xyxy[:, 0]
        boxes[:, 3] = boxes_xyxy[:, 3] - boxes_xyxy[:, 1]

        order = np.argsort(-scores)
        return boxes[order]
