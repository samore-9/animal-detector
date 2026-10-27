"""Turn a Kaggle "one folder per animal" dataset into train/val folders.

Works with both:
  * Animals-10            (raw-img/ has Italian folder names -> translated to English)
  * Animal Image Dataset  (90 animals: animals/animals/<name>/*.jpg)

Usage:
  python training/prepare_classifier_data.py --src path/to/raw-img --out datasets/animals_cls
"""
import argparse
import random
import shutil
from pathlib import Path

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
ITALIAN_TO_ENGLISH = {
    "cane": "dog", "cavallo": "horse", "elefante": "elephant", "farfalla": "butterfly",
    "gallina": "chicken", "gatto": "cat", "mucca": "cow", "pecora": "sheep",
    "scoiattolo": "squirrel", "ragno": "spider",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="folder that contains one sub-folder per animal")
    ap.add_argument("--out", default="datasets/animals_cls")
    ap.add_argument("--val", type=float, default=0.2, help="fraction used for validation")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    src, out = Path(args.src), Path(args.out)
    if not src.is_dir():
        raise SystemExit(f"Source folder not found: {src}")
    random.seed(args.seed)

    classes = sorted(p for p in src.iterdir() if p.is_dir())
    if not classes:
        raise SystemExit("No sub-folders found. Point --src at the folder that contains the animal folders.")

    total = 0
    for folder in classes:
        name = ITALIAN_TO_ENGLISH.get(folder.name.lower(), folder.name.lower()).replace(" ", "_")
        files = [f for f in folder.iterdir() if f.suffix.lower() in IMG_EXT]
        random.shuffle(files)
        n_val = max(1, int(len(files) * args.val))
        for i, f in enumerate(files):
            split = "val" if i < n_val else "train"
            dest = out / split / name
            dest.mkdir(parents=True, exist_ok=True)
            shutil.copy2(f, dest / f.name)
        total += len(files)
        print(f"{name:<16} {len(files):>5} images")
    print(f"\nDone: {total} images in {len(classes)} classes -> {out}")


if __name__ == "__main__":
    main()
