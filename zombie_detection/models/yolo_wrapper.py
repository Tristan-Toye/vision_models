"""YOLO wrapper for zombie detection using Ultralytics YOLOv8n / YOLOv11n.

Handles dataset export to YOLO format, training via the Ultralytics API,
and inference with bounding-box output.
"""

import shutil
import json
from pathlib import Path
from typing import Optional

import numpy as np
import yaml
from PIL import Image
from tqdm import tqdm


class YOLODetector:
    """Wraps Ultralytics YOLO for training and inference."""

    def __init__(
        self,
        model_name: str = "yolov8n",
        checkpoint: Optional[str] = None,
        **kwargs,
    ):
        self.model_name = model_name
        self.checkpoint = checkpoint
        self.model = None

    def _load_model(self):
        from ultralytics import YOLO
        if self.checkpoint and Path(self.checkpoint).exists():
            self.model = YOLO(self.checkpoint)
        else:
            self.model = YOLO(f"{self.model_name}.pt")

    def export_dataset_to_yolo_format(
        self,
        data_dir: Path,
        output_dir: Path,
        screen_w: int = 1280,
        screen_h: int = 720,
        image_format: str = "jpg",
        jpeg_quality: int = 85,
        resize_hw: tuple[int, int] | None = None,
    ):
        """Convert npy dataset to YOLO txt + images format.

        Creates:
            output_dir/
                images/train/  val/  test/
                labels/train/  val/  test/
                dataset.yaml
        """
        output_dir = Path(output_dir)
        image_format = image_format.lower().lstrip(".")
        if image_format not in {"jpg", "jpeg", "png"}:
            raise ValueError(f"Unsupported image_format: {image_format}")

        # If it already looks exported with the same settings, reuse it.
        yaml_path = output_dir / "dataset.yaml"
        meta_path = output_dir / "export_meta.json"
        meta = {
            "image_format": "jpg" if image_format == "jpeg" else image_format,
            "jpeg_quality": int(jpeg_quality),
            "resize_hw": list(resize_hw) if resize_hw is not None else None,
            "screen_w": int(screen_w),
            "screen_h": int(screen_h),
        }
        if (
            yaml_path.exists()
            and meta_path.exists()
            and (output_dir / "images").exists()
            and (output_dir / "labels").exists()
        ):
            try:
                prev = json.loads(meta_path.read_text())
                if prev == meta:
                    return str(yaml_path)
            except Exception:
                pass

        # Fresh export (or settings changed): clear old images/labels to avoid mixing formats
        shutil.rmtree(output_dir / "images", ignore_errors=True)
        shutil.rmtree(output_dir / "labels", ignore_errors=True)
        output_dir.mkdir(parents=True, exist_ok=True)

        for split in ["train", "val", "test"]:
            split_dir = Path(data_dir) / split
            if not split_dir.exists():
                continue

            img_out = output_dir / "images" / split
            lbl_out = output_dir / "labels" / split
            img_out.mkdir(parents=True, exist_ok=True)
            lbl_out.mkdir(parents=True, exist_ok=True)

            obs_files = sorted(split_dir.glob("*_obs.npy"))
            for obs_path in tqdm(obs_files, desc=f"  Export YOLO [{split}]", unit="img", leave=False):
                stem = obs_path.name.replace("_obs.npy", "")
                box_path = split_dir / f"{stem}_zombies.npy"
                if not box_path.exists():
                    continue

                img_path = img_out / f"{stem}.{image_format}"
                lbl_path = lbl_out / f"{stem}.txt"

                # Skip work if already exported.
                if img_path.exists() and lbl_path.exists():
                    continue

                image = np.load(str(obs_path))  # (H, W, 3) uint8 (screen coords)
                boxes = np.load(str(box_path))  # (N, 4) [x, y, w, h] in screen coords

                img_pil = Image.fromarray(image)
                out_w, out_h = image.shape[1], image.shape[0]
                if resize_hw is not None:
                    out_h, out_w = int(resize_hw[0]), int(resize_hw[1])
                    img_pil = img_pil.resize((out_w, out_h), resample=Image.BILINEAR)

                    # Scale boxes from screen coords -> resized image pixel coords
                    if len(boxes) > 0:
                        boxes = boxes.copy()
                        sx = out_w / float(screen_w)
                        sy = out_h / float(screen_h)
                        boxes[:, 0] *= sx
                        boxes[:, 1] *= sy
                        boxes[:, 2] *= sx
                        boxes[:, 3] *= sy
                else:
                    # If not resizing, boxes are still in screen coords. Convert to image coords in case
                    # the stored arrays are not at screen resolution.
                    if (image.shape[1], image.shape[0]) != (screen_w, screen_h) and len(boxes) > 0:
                        boxes = boxes.copy()
                        sx = image.shape[1] / float(screen_w)
                        sy = image.shape[0] / float(screen_h)
                        boxes[:, 0] *= sx
                        boxes[:, 1] *= sy
                        boxes[:, 2] *= sx
                        boxes[:, 3] *= sy

                if image_format in {"jpg", "jpeg"}:
                    if img_pil.mode != "RGB":
                        img_pil = img_pil.convert("RGB")
                    img_pil.save(str(img_path), quality=jpeg_quality, optimize=True)
                else:
                    img_pil.save(str(img_path))

                with open(str(lbl_path), "w") as f:
                    for box in boxes:
                        # Clip boxes to image bounds to keep YOLO labels normalized in [0, 1]
                        x, y, w, h = float(box[0]), float(box[1]), float(box[2]), float(box[3])
                        x1 = max(0.0, x)
                        y1 = max(0.0, y)
                        x2 = min(float(out_w), x + w)
                        y2 = min(float(out_h), y + h)
                        cw = x2 - x1
                        ch = y2 - y1
                        if cw <= 1.0 or ch <= 1.0:
                            continue

                        cx = (x1 + x2) / 2.0 / float(out_w)
                        cy = (y1 + y2) / 2.0 / float(out_h)
                        nw = cw / float(out_w)
                        nh = ch / float(out_h)
                        # Extra safety: clamp to [0, 1]
                        cx = min(1.0, max(0.0, cx))
                        cy = min(1.0, max(0.0, cy))
                        nw = min(1.0, max(0.0, nw))
                        nh = min(1.0, max(0.0, nh))
                        f.write(f"0 {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}\n")

        ds_yaml = {
            "path": str(output_dir.resolve()),
            "train": "images/train",
            "val": "images/val",
            "test": "images/test",
            "names": {0: "zombie"},
        }
        with open(yaml_path, "w") as f:
            yaml.dump(ds_yaml, f, default_flow_style=False)

        meta_path.write_text(json.dumps(meta, indent=2))
        return str(yaml_path)

    def train(
        self,
        dataset_yaml: str,
        epochs: int = 50,
        batch_size: int = 16,
        imgsz: int = 640,
        project: str = "zombie_detection/runs",
        name: str = "yolo_train",
        **train_kwargs,
    ) -> dict:
        """Train YOLO model using Ultralytics API. Returns results dict."""
        if self.model is None:
            self._load_model()

        results = self.model.train(
            data=dataset_yaml,
            epochs=epochs,
            batch=batch_size,
            imgsz=imgsz,
            project=project,
            name=name,
            verbose=True,
            **train_kwargs,
        )
        return results

    def predict(self, image: np.ndarray, conf: float = 0.25, **predict_kwargs) -> np.ndarray:
        """Run inference on a single image (H, W, 3) uint8.

        Returns (N, 4) array of [x, y, w, h] in pixel coords.
        """
        if self.model is None:
            self._load_model()

        results = self.model.predict(image, conf=conf, verbose=False, **predict_kwargs)

        if len(results) == 0 or len(results[0].boxes) == 0:
            return np.zeros((0, 4), dtype=np.float32)

        xyxy = results[0].boxes.xyxy.cpu().numpy()
        confs = results[0].boxes.conf.cpu().numpy()

        boxes = np.zeros((len(xyxy), 4), dtype=np.float32)
        boxes[:, 0] = xyxy[:, 0]
        boxes[:, 1] = xyxy[:, 1]
        boxes[:, 2] = xyxy[:, 2] - xyxy[:, 0]
        boxes[:, 3] = xyxy[:, 3] - xyxy[:, 1]

        order = np.argsort(-confs)
        return boxes[order]

    def evaluate(self, dataset_yaml: str, split: str = "test") -> dict:
        """Evaluate on a dataset split."""
        if self.model is None:
            self._load_model()
        return self.model.val(data=dataset_yaml, split=split)
