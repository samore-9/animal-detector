"""Train an animal *classifier* (species of the whole image).

  python training/train_classifier.py --data datasets/animals_cls --epochs 15

Result is copied to models/animal_classifier.pt, which the web app then uses
automatically as a fallback when the detector finds nothing.
"""
import argparse
import shutil
from pathlib import Path

from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent.parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="folder with train/ and val/ sub-folders")
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--imgsz", type=int, default=224)
    ap.add_argument("--base", default="yolov8n-cls.pt", help="try yolov8s-cls.pt for higher accuracy")
    ap.add_argument("--device", default=None, help="0 for first GPU, 'cpu' to force CPU")
    args = ap.parse_args()

    model = YOLO(args.base)
    model.train(data=str(Path(args.data).resolve()), epochs=args.epochs, imgsz=args.imgsz,
                batch=args.batch, device=args.device, project=str(ROOT / "runs"), name="classifier")

    best = Path(model.trainer.best)
    target = ROOT / "models" / "animal_classifier.pt"
    target.parent.mkdir(exist_ok=True)
    shutil.copy2(best, target)
    print(f"\nSaved {target}. Restart the web app to use it.")


if __name__ == "__main__":
    main()
