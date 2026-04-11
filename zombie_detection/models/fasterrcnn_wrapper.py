"""Faster R-CNN wrapper using torchvision's pretrained model.

Fine-tunes a Faster R-CNN with ResNet-50 FPN backbone on the zombie
detection dataset. Supports freezing the backbone for transfer learning.
"""

import numpy as np
import torch
import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor


class FasterRCNNDetector(torch.nn.Module):
    """Faster R-CNN fine-tuned for zombie detection (2 classes: bg + zombie)."""

    def __init__(
        self,
        pretrained: bool = True,
        freeze_backbone: bool = False,
        num_classes: int = 2,
        **kwargs,
    ):
        super().__init__()

        weights = FasterRCNN_ResNet50_FPN_Weights.COCO_V1 if pretrained else None
        self.model = fasterrcnn_resnet50_fpn(weights=weights)

        in_features = self.model.roi_heads.box_predictor.cls_score.in_features
        self.model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

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
        """Run inference on a single image (H, W, 3) uint8.

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
