"""Fine-tune a YOLO *detector* on a Kaggle dataset in YOLO format.

  python training/train_detector.py --data training/data.yaml --epochs 50

Result is copied to models/animal_detector.pt. When that file exists the web
app uses it INSTEAD of the pretrained COCO model, so it will only know the
classes you trained on (e.g. buffalo, elephant, rhino, zebra).
"""
import argparse
import shutil
from pathlib import Path

from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent.parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="path to data.yaml")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--base", default="yolov8s.pt", help="starting weights (transfer learning)")
    ap.add_argument("--device", default=None, help="0 for first GPU, 'cpu' to force CPU")
    args = ap.parse_args()

    model = YOLO(args.base)
    model.train(data=args.data, epochs=args.epochs, imgsz=args.imgsz, batch=args.batch,
                device=args.device, project=str(ROOT / "runs"), name="detector")

    best = Path(model.trainer.best)
    target = ROOT / "models" / "animal_detector.pt"
    target.parent.mkdir(exist_ok=True)
    shutil.copy2(best, target)
    print(f"\nSaved {target}. Restart the web app to use it.")


if __name__ == "__main__":
    main()
