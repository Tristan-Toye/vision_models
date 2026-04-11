"""YOLO wrapper for zombie detection using Ultralytics YOLOv8n / YOLOv11n.

Handles dataset export to YOLO format, training via the Ultralytics API,
and inference with bounding-box output.
"""

import shutil
from pathlib import Path
from typing import Optional

import numpy as np
import yaml
from PIL import Image


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
    ):
        """Convert npy dataset to YOLO txt + images format.

        Creates:
            output_dir/
                images/train/  val/  test/
                labels/train/  val/  test/
                dataset.yaml
        """
        output_dir = Path(output_dir)

        for split in ["train", "val", "test"]:
            split_dir = Path(data_dir) / split
            if not split_dir.exists():
                continue

            img_out = output_dir / "images" / split
            lbl_out = output_dir / "labels" / split
            img_out.mkdir(parents=True, exist_ok=True)
            lbl_out.mkdir(parents=True, exist_ok=True)

            obs_files = sorted(split_dir.glob("*_obs.npy"))
            for obs_path in obs_files:
                stem = obs_path.name.replace("_obs.npy", "")
                box_path = split_dir / f"{stem}_zombies.npy"
                if not box_path.exists():
                    continue

                image = np.load(str(obs_path))
                boxes = np.load(str(box_path))

                img_pil = Image.fromarray(image)
                img_pil.save(str(img_out / f"{stem}.png"))

                with open(str(lbl_out / f"{stem}.txt"), "w") as f:
                    for box in boxes:
                        cx = (box[0] + box[2] / 2.0) / screen_w
                        cy = (box[1] + box[3] / 2.0) / screen_h
                        w = box[2] / screen_w
                        h = box[3] / screen_h
                        f.write(f"0 {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")

        ds_yaml = {
            "path": str(output_dir.resolve()),
            "train": "images/train",
            "val": "images/val",
            "test": "images/test",
            "names": {0: "zombie"},
        }
        yaml_path = output_dir / "dataset.yaml"
        with open(yaml_path, "w") as f:
            yaml.dump(ds_yaml, f, default_flow_style=False)

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

    def predict(self, image: np.ndarray, conf: float = 0.25) -> np.ndarray:
        """Run inference on a single image (H, W, 3) uint8.

        Returns (N, 4) array of [x, y, w, h] in pixel coords.
        """
        if self.model is None:
            self._load_model()

        results = self.model.predict(image, conf=conf, verbose=False)

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
