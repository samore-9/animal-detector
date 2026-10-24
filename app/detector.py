"""Model wrapper: finds animals in an image.

Model selection (automatic):
  1. models/animal_detector.pt   -> your own fine-tuned YOLO detector (if present)
     otherwise a pretrained YOLOv8 COCO detector (downloaded on first run),
     filtered to the animal classes it knows.
  2. models/animal_classifier.pt -> optional YOLO classifier trained on a Kaggle
     animal dataset. Used as a fallback ("best guess") when the detector finds nothing.
"""
from __future__ import annotations

import time
from collections import OrderedDict
from pathlib import Path

from PIL import Image
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT / "models"
CUSTOM_DETECTOR = MODELS_DIR / "animal_detector.pt"
CUSTOM_CLASSIFIER = MODELS_DIR / "animal_classifier.pt"
DEFAULT_DETECTOR = "yolov8s.pt"  # auto-downloaded by ultralytics on first use

# The 10 animal classes inside the 80-class COCO dataset.
COCO_ANIMALS = {
    "bird", "cat", "dog", "horse", "sheep",
    "cow", "elephant", "bear", "zebra", "giraffe",
}


class AnimalDetector:
    def __init__(self) -> None:
        if CUSTOM_DETECTOR.exists():
            self.detector = YOLO(str(CUSTOM_DETECTOR))
            self.detector_name = f"custom ({CUSTOM_DETECTOR.name})"
            self.allowed = None  # trust every class of the custom model
        else:
            self.detector = YOLO(DEFAULT_DETECTOR)
            self.detector_name = f"{DEFAULT_DETECTOR} (COCO pretrained)"
            self.allowed = COCO_ANIMALS

        self.classifier = None
        if CUSTOM_CLASSIFIER.exists():
            self.classifier = YOLO(str(CUSTOM_CLASSIFIER))

    # ------------------------------------------------------------------
    def analyze(self, image: Image.Image, conf: float = 0.35) -> dict:
        start = time.perf_counter()
        width, height = image.size

        detections = self._detect(image, conf, width, height)
        mode = "detector"

        if not detections and self.classifier is not None:
            detections = self._classify(image)
            mode = "classifier" if detections else "none"
        elif not detections:
            mode = "none"

        # Summarise: one row per species with a count.
        summary: "OrderedDict[str, dict]" = OrderedDict()
        for d in sorted(detections, key=lambda x: -x["confidence"]):
            row = summary.setdefault(
                d["label"], {"label": d["label"], "count": 0, "best_confidence": d["confidence"]}
            )
            row["count"] += 1

        return {
            "mode": mode,
            "animals": list(summary.values()),
            "detections": detections,
            "image": {"width": width, "height": height},
            "model": self.detector_name,
            "ms": round((time.perf_counter() - start) * 1000),
        }

    # ------------------------------------------------------------------
    def _detect(self, image, conf, width, height):
        result = self.detector.predict(image, conf=conf, verbose=False)[0]
        names = result.names
        out = []
        for box in result.boxes:
            label = names[int(box.cls[0])]
            if self.allowed is not None and label not in self.allowed:
                continue
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            out.append({
                "label": label.replace("_", " "),
                "confidence": round(float(box.conf[0]), 4),
                # normalised 0-1 so the browser can draw at any size
                "box": [x1 / width, y1 / height, x2 / width, y2 / height],
            })
        return out

    def _classify(self, image, min_conf: float = 0.30, top_k: int = 3):
        result = self.classifier.predict(image, verbose=False)[0]
        names = result.names
        out = []
        for idx, p in zip(result.probs.top5, result.probs.top5conf.tolist()):
            if p >= min_conf and len(out) < top_k:
                out.append({
                    "label": names[int(idx)].replace("_", " "),
                    "confidence": round(float(p), 4),
                    "box": None,  # classifier has no location
                })
        return out
