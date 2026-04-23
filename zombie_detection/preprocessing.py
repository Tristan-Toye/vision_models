"""Configurable image preprocessing pipeline for zombie detection.

Each transform takes (image_hwc_uint8, boxes_n4) and returns the same.
Transforms can be composed and toggled via config.
"""

import cv2
import numpy as np


class Resize:
    """Resize image to (target_h, target_w). Boxes are NOT rescaled here
    because the dataset handles coordinate scaling at target-creation time
    using the original screen dimensions."""

    def __init__(self, target_h: int, target_w: int):
        self.target_h = target_h
        self.target_w = target_w

    def __call__(self, image: np.ndarray, boxes: np.ndarray):
        image = cv2.resize(image, (self.target_w, self.target_h), interpolation=cv2.INTER_LINEAR)
        return image, boxes


class ToGrayscale:
    """Convert to single-channel grayscale, then replicate to 3 channels
    so the tensor shape stays consistent."""

    def __call__(self, image: np.ndarray, boxes: np.ndarray):
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        image = np.stack([gray, gray, gray], axis=-1)
        return image, boxes


class AddEdgeChannel:
    """Append a Canny edge-detection channel, making the image 4-channel."""

    def __init__(self, low_thresh: int = 50, high_thresh: int = 150):
        self.low = low_thresh
        self.high = high_thresh

    def __call__(self, image: np.ndarray, boxes: np.ndarray):
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray, self.low, self.high)
        image = np.concatenate([image, edges[:, :, None]], axis=-1)
        return image, boxes


MANUAL_FEATURE_NAMES = ["edge_magnitude", "gradient_orientation", "hsv_saturation", "lbp"]
NUM_MANUAL_FEATURES = len(MANUAL_FEATURE_NAMES)


def _compute_lbp(gray: np.ndarray) -> np.ndarray:
    """Simplified 8-neighbor Local Binary Pattern."""
    h, w = gray.shape
    lbp = np.zeros((h, w), dtype=np.uint8)
    padded = cv2.copyMakeBorder(gray, 1, 1, 1, 1, cv2.BORDER_REFLECT)
    offsets = [(-1, -1), (-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1)]
    for bit, (dr, dc) in enumerate(offsets):
        neighbor = padded[1 + dr: 1 + dr + h, 1 + dc: 1 + dc + w]
        lbp |= ((neighbor >= gray).astype(np.uint8) << bit)
    return lbp


def compute_manual_features(image: np.ndarray) -> np.ndarray:
    """Compute 4 handcrafted feature maps from an RGB(+) image.

    Returns (H, W, 4) uint8 array with channels:
      0: edge magnitude (Sobel)
      1: gradient orientation
      2: HSV saturation
      3: Local Binary Pattern
    """
    rgb = image[:, :, :3]
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)

    sx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    sy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)

    edge_mag = np.sqrt(sx ** 2 + sy ** 2)
    mag_max = edge_mag.max()
    if mag_max > 0:
        edge_mag = (edge_mag / mag_max * 255).clip(0, 255).astype(np.uint8)
    else:
        edge_mag = edge_mag.astype(np.uint8)

    grad_orient = ((np.arctan2(sy, sx) / np.pi + 1) * 127.5).astype(np.uint8)

    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    saturation = hsv[:, :, 1]

    lbp = _compute_lbp(gray)

    return np.stack([edge_mag, grad_orient, saturation, lbp], axis=-1)


class AddManualFeatures:
    """Append 4 handcrafted feature channels to the image.

    Optional channel_mask allows zeroing out specific feature channels
    for ablation experiments (indices 0-3 map to MANUAL_FEATURE_NAMES).
    """

    def __init__(self, channel_mask: list[bool] | None = None):
        self.channel_mask = channel_mask

    def __call__(self, image: np.ndarray, boxes: np.ndarray):
        features = compute_manual_features(image)
        if self.channel_mask is not None:
            for i, keep in enumerate(self.channel_mask):
                if not keep:
                    features[:, :, i] = 0
        return np.concatenate([image, features], axis=-1), boxes


class RandomHorizontalFlip:
    """Flip image and boxes horizontally with probability p."""

    def __init__(self, p: float = 0.5):
        self.p = p

    def __call__(self, image: np.ndarray, boxes: np.ndarray):
        if np.random.random() < self.p:
            h, w = image.shape[:2]
            image = image[:, ::-1].copy()
            if len(boxes) > 0:
                boxes = boxes.copy()
                boxes[:, 0] = w - boxes[:, 0] - boxes[:, 2]
        return image, boxes


class RandomBrightnessContrast:
    """Random brightness and contrast jitter."""

    def __init__(self, brightness_range=(0.85, 1.15), contrast_range=(0.85, 1.15)):
        self.b_range = brightness_range
        self.c_range = contrast_range

    def __call__(self, image: np.ndarray, boxes: np.ndarray):
        b = np.random.uniform(*self.b_range)
        c = np.random.uniform(*self.c_range)
        img = image.astype(np.float32)
        img = (img - 128.0) * c + 128.0
        img = img * b
        image = np.clip(img, 0, 255).astype(np.uint8)
        return image, boxes


class GaussianBlur:
    """Mild Gaussian blur."""

    def __init__(self, kernel_size: int = 3, sigma: float = 0.5):
        self.ksize = kernel_size
        self.sigma = sigma

    def __call__(self, image: np.ndarray, boxes: np.ndarray):
        image = cv2.GaussianBlur(image, (self.ksize, self.ksize), self.sigma)
        return image, boxes


class Compose:
    """Chain multiple (image, boxes) transforms."""

    def __init__(self, transforms: list):
        self.transforms = transforms

    def __call__(self, image: np.ndarray, boxes: np.ndarray):
        for t in self.transforms:
            image, boxes = t(image, boxes)
        return image, boxes


def build_preprocessing_pipeline(
    cfg: dict,
    variant: str = "rgb",
    augment: bool = False,
    resize: "list[int] | None" = "from_config",
    manual_features: bool = False,
    manual_features_mask: list[bool] | None = None,
) -> Compose:
    """Build a preprocessing pipeline from config.

    Args:
        cfg: full config dict
        variant: one of 'rgb', 'grayscale', 'rgb_edge'
        augment: whether to include data augmentation transforms
        resize: [h, w] to resize, None for no resize, or "from_config" to
                read from cfg["preprocessing"]["resize"]
        manual_features: whether to append handcrafted feature channels
        manual_features_mask: optional per-channel mask for ablation
                              (length 4, True=keep, False=zero)
    """
    transforms = []

    if resize == "from_config":
        resize = cfg["preprocessing"].get("resize")

    if resize is not None:
        target_h, target_w = resize
        transforms.append(Resize(target_h, target_w))

    if variant == "grayscale":
        transforms.append(ToGrayscale())
    elif variant == "rgb_edge":
        transforms.append(AddEdgeChannel())

    if augment:
        transforms.append(RandomHorizontalFlip(p=0.5))
        transforms.append(RandomBrightnessContrast())
        transforms.append(GaussianBlur(kernel_size=3, sigma=0.5))

    if manual_features:
        transforms.append(AddManualFeatures(channel_mask=manual_features_mask))

    return Compose(transforms)
