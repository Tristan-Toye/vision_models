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
) -> Compose:
    """Build a preprocessing pipeline from config.

    Args:
        cfg: full config dict
        variant: one of 'rgb', 'grayscale', 'rgb_edge'
        augment: whether to include data augmentation transforms
        resize: [h, w] to resize, None for no resize, or "from_config" to
                read from cfg["preprocessing"]["resize"]
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

    return Compose(transforms)
