"""Classical detection baselines: HOG+SVM and template matching.

These do not use deep learning and serve as comparison baselines.
"""

import os
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
import joblib


class HOGSVMDetector:
    """Sliding-window detector using HOG features + linear SVM.

    Training: extract HOG features from zombie and background patches,
    train a linear SVM.
    Inference: slide window over image, classify each patch.
    """

    def __init__(
        self,
        window_size: tuple[int, int] = (31, 29),  # (h, w) matching zombie bbox
        step_size: int = 8,
        hog_cell_size: tuple[int, int] = (4, 4),
        hog_block_size: tuple[int, int] = (2, 2),
        hog_nbins: int = 9,
        checkpoint: Optional[str] = None,
        **kwargs,
    ):
        self.window_h, self.window_w = window_size
        self.step_size = step_size
        # OpenCV 4.8+ Python bindings reject keyword args for HOGDescriptor; use positional ctor.
        _block_size = (
            hog_block_size[0] * hog_cell_size[0],
            hog_block_size[1] * hog_cell_size[1],
        )
        _block_stride = (hog_cell_size[0], hog_cell_size[1])
        _cell_size = hog_cell_size
        # OpenCV requires (winSize - blockSize) % blockStride == 0. Our bbox-sized
        # window (29x31) does not satisfy that for the default 8x8 blocks and 4x4 stride,
        # so we round the *HOG feature window* up to a compatible size.
        def _round_up_win(win: int, block: int, stride: int) -> int:
            if win < block:
                return block
            rem = (win - block) % stride
            return win if rem == 0 else win + (stride - rem)

        self.hog_win_w = _round_up_win(self.window_w, _block_size[0], _block_stride[0])
        self.hog_win_h = _round_up_win(self.window_h, _block_size[1], _block_stride[1])
        _win_size = (self.hog_win_w, self.hog_win_h)
        self.hog = cv2.HOGDescriptor(_win_size, _block_size, _block_stride, _cell_size, hog_nbins)
        self.scaler = StandardScaler()
        self.svm = LinearSVC(C=1.0, max_iter=5000)
        self.is_trained = False

        if checkpoint and Path(checkpoint).exists():
            self._load(checkpoint)

    def _extract_hog(self, patch: np.ndarray) -> np.ndarray:
        """Extract HOG features from a single patch."""
        if len(patch.shape) == 3:
            patch = cv2.cvtColor(patch, cv2.COLOR_RGB2GRAY)
        patch = cv2.resize(patch, (self.hog_win_w, self.hog_win_h))
        return self.hog.compute(patch).flatten()

    def train_from_dataset(
        self,
        data_dir: Path,
        bbox_w: int = 29,
        bbox_h: int = 31,
        neg_per_image: int = 10,
        max_images: int = 2000,
    ):
        """Build training set from npy dataset and train SVM."""
        features = []
        labels = []

        obs_files = sorted(Path(data_dir).glob("*_obs.npy"))[:max_images]

        for obs_path in obs_files:
            stem = obs_path.name.replace("_obs.npy", "")
            box_path = obs_path.parent / f"{stem}_zombies.npy"
            if not box_path.exists():
                continue

            image = np.load(str(obs_path))
            boxes = np.load(str(box_path))
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            h, w = gray.shape

            # Positive patches
            for box in boxes:
                x, y = int(box[0]), int(box[1])
                bw, bh = int(box[2]), int(box[3])
                x = max(0, min(x, w - bw))
                y = max(0, min(y, h - bh))
                patch = gray[y:y + bh, x:x + bw]
                if patch.shape[0] < 4 or patch.shape[1] < 4:
                    continue
                feat = self._extract_hog(patch)
                features.append(feat)
                labels.append(1)

            # Negative patches (random, non-overlapping with zombies)
            for _ in range(neg_per_image):
                rx = np.random.randint(0, max(1, w - bbox_w))
                ry = np.random.randint(0, max(1, h - bbox_h))
                patch = gray[ry:ry + bbox_h, rx:rx + bbox_w]
                if patch.shape[0] < 4 or patch.shape[1] < 4:
                    continue
                feat = self._extract_hog(patch)
                features.append(feat)
                labels.append(0)

        X = np.array(features)
        y = np.array(labels)

        X = self.scaler.fit_transform(X)
        self.svm.fit(X, y)
        self.is_trained = True

    def predict(
        self,
        image: np.ndarray,
        conf: float = 0.0,
        nms_threshold: float = 0.3,
    ) -> np.ndarray:
        """Slide window over image and return detected boxes [x,y,w,h]."""
        if not self.is_trained:
            return np.zeros((0, 4), dtype=np.float32)

        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if len(image.shape) == 3 else image
        h, w = gray.shape

        detections = []
        scores = []

        for y in range(0, h - self.window_h, self.step_size):
            for x in range(0, w - self.window_w, self.step_size):
                patch = gray[y:y + self.window_h, x:x + self.window_w]
                feat = self._extract_hog(patch)
                feat = self.scaler.transform(feat.reshape(1, -1))
                score = self.svm.decision_function(feat)[0]
                if score > conf:
                    detections.append([x, y, self.window_w, self.window_h])
                    scores.append(score)

        if len(detections) == 0:
            return np.zeros((0, 4), dtype=np.float32)

        boxes = np.array(detections, dtype=np.float32)
        scores = np.array(scores, dtype=np.float32)

        # NMS
        keep = self._nms(boxes, scores, nms_threshold)
        return boxes[keep]

    def _nms(self, boxes, scores, threshold):
        """Non-maximum suppression for xywh boxes."""
        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = x1 + boxes[:, 2]
        y2 = y1 + boxes[:, 3]
        areas = boxes[:, 2] * boxes[:, 3]

        order = scores.argsort()[::-1]
        keep = []

        while len(order) > 0:
            i = order[0]
            keep.append(i)

            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])

            inter = np.maximum(0, xx2 - xx1) * np.maximum(0, yy2 - yy1)
            iou = inter / (areas[i] + areas[order[1:]] - inter)

            inds = np.where(iou <= threshold)[0]
            order = order[inds + 1]

        return keep

    def save(self, path: str):
        joblib.dump({"svm": self.svm, "scaler": self.scaler}, path)

    def _load(self, path: str):
        data = joblib.load(path)
        self.svm = data["svm"]
        self.scaler = data["scaler"]
        self.is_trained = True


class TemplateMatchDetector:
    """OpenCV template matching using the zombie sprite as template.

    This is a non-learning baseline that works best at distortion level 0.
    """

    def __init__(
        self,
        template_path: Optional[str] = None,
        method: int = cv2.TM_CCOEFF_NORMED,
        threshold: float = 0.6,
        **kwargs,
    ):
        self.method = method
        self.threshold = threshold
        self.template = None

        if template_path and Path(template_path).exists():
            self.template = cv2.imread(str(template_path), cv2.IMREAD_GRAYSCALE)

    def set_template_from_patch(self, patch: np.ndarray):
        """Set template from a zombie patch (H, W, 3) or (H, W)."""
        if len(patch.shape) == 3:
            patch = cv2.cvtColor(patch, cv2.COLOR_RGB2GRAY)
        self.template = patch

    def extract_template_from_dataset(self, data_dir: Path, bbox_w: int = 29, bbox_h: int = 31):
        """Extract average zombie template from the training set."""
        obs_files = sorted(Path(data_dir).glob("*_obs.npy"))
        patches = []

        for obs_path in obs_files[:500]:
            stem = obs_path.name.replace("_obs.npy", "")
            box_path = obs_path.parent / f"{stem}_zombies.npy"
            if not box_path.exists():
                continue

            image = np.load(str(obs_path))
            boxes = np.load(str(box_path))
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            h, w = gray.shape

            for box in boxes:
                x, y = int(box[0]), int(box[1])
                x = max(0, min(x, w - bbox_w))
                y = max(0, min(y, h - bbox_h))
                patch = gray[y:y + bbox_h, x:x + bbox_w]
                if patch.shape == (bbox_h, bbox_w):
                    patches.append(patch.astype(np.float32))

            if len(patches) >= 200:
                break

        if patches:
            self.template = np.mean(patches, axis=0).astype(np.uint8)

    def predict(self, image: np.ndarray, conf: float = 0.0) -> np.ndarray:
        """Template match and return detected boxes [x,y,w,h]."""
        if self.template is None:
            return np.zeros((0, 4), dtype=np.float32)

        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if len(image.shape) == 3 else image
        th, tw = self.template.shape[:2]

        result = cv2.matchTemplate(gray, self.template, self.method)

        locations = np.where(result >= self.threshold)
        if len(locations[0]) == 0:
            return np.zeros((0, 4), dtype=np.float32)

        boxes = []
        scores = []
        for y, x in zip(locations[0], locations[1]):
            boxes.append([x, y, tw, th])
            scores.append(result[y, x])

        boxes = np.array(boxes, dtype=np.float32)
        scores = np.array(scores, dtype=np.float32)

        # NMS
        keep = self._nms(boxes, scores, 0.3)
        return boxes[keep]

    def _nms(self, boxes, scores, threshold):
        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = x1 + boxes[:, 2]
        y2 = y1 + boxes[:, 3]
        areas = boxes[:, 2] * boxes[:, 3]

        order = scores.argsort()[::-1]
        keep = []

        while len(order) > 0:
            i = order[0]
            keep.append(i)
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])
            inter = np.maximum(0, xx2 - xx1) * np.maximum(0, yy2 - yy1)
            iou = inter / (areas[i] + areas[order[1:]] - inter)
            inds = np.where(iou <= threshold)[0]
            order = order[inds + 1]

        return keep
