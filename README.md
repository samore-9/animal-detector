# Animal Detector

Drop a photo into the web page and get back the names of the animals in it, with boxes drawn around each one.

- **Backend:** Python, FastAPI, Ultralytics YOLOv8 (PyTorch)
- **Frontend:** one plain HTML/CSS/JS page (no Node.js or build step needed)
- **Works out of the box** with a pretrained model, and **can be re-trained** on Kaggle datasets

## How it works

```
Browser (drag & drop)  --POST image-->  FastAPI /api/detect  -->  YOLOv8 detector
                       <--JSON boxes--                        -->  (optional) your trained classifier as fallback
```

| Model | File | What it does |
|---|---|---|
| Detector (default) | `yolov8s.pt`, auto-downloaded | Finds and names **bird, cat, dog, horse, sheep, cow, elephant, bear, zebra, giraffe** (the animal classes in COCO). Handles several animals in one photo. |
| Your detector (optional) | `models/animal_detector.pt` | Fine-tuned on a Kaggle YOLO dataset. If present, it replaces the default. |
| Your classifier (optional) | `models/animal_classifier.pt` | Trained on Animals-10 or the 90-animal dataset. Used as a "best guess" when the detector finds nothing, so you get names for animals COCO doesn't know (butterfly, spider, squirrel, etc.). |

## 1. Requirements

- Python **3.10, 3.11 or 3.12** (check with `python --version`)
- About 3 GB free disk space (PyTorch is large)
- Internet on the first run only, to download the model weights (~22 MB)
- A GPU is not needed for running the app

## 2. Setup (VS Code)

1. Unzip the project and open the folder: **File > Open Folder > animal-detector**.
2. Open a terminal: **Terminal > New Terminal**.
3. Create and activate a virtual environment.

   Windows (PowerShell):
   ```powershell
   python -m venv .venv
   .venv\Scripts\Activate.ps1
   ```
   If PowerShell blocks the script, run `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` once, then retry.

   macOS / Linux:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```
4. Install the packages:
   ```bash
   pip install -r requirements.txt
   ```
5. In VS Code press **Ctrl+Shift+P > Python: Select Interpreter** and choose the `.venv` one.

## 3. Run the app

```bash
python run.py
```

Wait for `Model ready: yolov8s.pt (COCO pretrained)`, then open **http://127.0.0.1:8000** (a browser tab opens by itself after a few seconds).

Drag an image onto the box, click **Browse files**, or paste an image with Ctrl+V. Use the confidence slider to make detection stricter or looser. Stop the server with **Ctrl+C**.

Different port: `PORT=9000 python run.py` (macOS/Linux) or `$env:PORT=9000; python run.py` (PowerShell).

## 4. Run in Jupyter Lab instead

```bash
pip install -r requirements.txt jupyterlab ipykernel
python -m ipykernel install --user --name animal-detector
jupyter lab
```

Open `notebooks/animal_detection_demo.ipynb`, choose the **animal-detector** kernel, put a photo in `sample_images/`, set `IMAGE_PATH`, and run the cells. To start the web page from Jupyter, open a terminal inside Jupyter Lab and run `python run.py`.

## 5. Test it

1. **In the browser:** upload a photo of a dog, a cat, a horse, or several animals. You should see boxes and names.
2. **From the command line** (server must be running in another terminal):
   ```bash
   python scripts/test_image.py sample_images/your_photo.jpg
   ```
3. **Health check:** open http://127.0.0.1:8000/api/health, which should show `"status": "ok"`.
4. **Raw API:**
   ```bash
   curl -F "file=@photo.jpg" -F "conf=0.35" http://127.0.0.1:8000/api/detect
   ```
5. **Error handling:** upload a `.txt` file renamed to `.jpg`. The page should show a clear error message instead of crashing.

## 6. Kaggle datasets

I confirmed the first two exist on Kaggle. For the rest I found the dataset's description but not an exact Kaggle page, so I've given Kaggle search links. Check the license on each dataset page before using it.

| Dataset | Best for | Size and classes | Link |
|---|---|---|---|
| **Animals-10** | Classifier, species names | About 28K images, 10 classes (dog, cat, horse, spider, butterfly, chicken, sheep, cow, squirrel, elephant). Folder names are in Italian; my prepare script translates them. | https://www.kaggle.com/datasets/alessiocorrado99/animals10 |
| **Animal Image Dataset (90 different animals)** | Classifier with many species | 90 classes, about 60 images each (small, so expect weaker accuracy than Animals-10) | https://www.kaggle.com/datasets/iamsouravbanerjee/animal-image-dataset-90-different-animals |
| **Object Detection: Wildlife (YOLO format)** | Detector fine-tuning | 1,500 images, 4 classes (buffalo, elephant, rhino, zebra), already in YOLO format with an 8:1:1 split | Search: https://www.kaggle.com/search?q=wildlife+yolo+buffalo+elephant+rhino+zebra |
| **Animal Image Classification (15 classes)** | Quick small classifier | 15 classes, at least 120 images each, 224x224 | Search: https://www.kaggle.com/search?q=animal+image+classification+15+classes |
| **Cats, Dogs and Horses** | Small test set | 1,763 images, 3 classes | Search: https://www.kaggle.com/search?q=animal+image+dataset+cats+dogs+horse |
| **Animal-10N** | Noisy-label experiments | 10 classes, online images with human-labelled noise | Search: https://www.kaggle.com/search?q=animal-10n |
| Roboflow "Animal-10" (not Kaggle) | Detection with boxes | Object-detection dataset, CC BY 4.0 | https://universe.roboflow.com/project-kykhc/animal-10 |
| COCO (not Kaggle) | What the default model already learned | 80 classes including 10 animals | https://cocodataset.org |
| Open Images (not Kaggle) | Hundreds of animal classes with boxes | Large; download only the animal classes | https://storage.googleapis.com/openimages/web/index.html |

**Which should you pick?**
- Want it working now: do nothing, the pretrained model already runs.
- Want more species names (butterfly, spider, squirrel...): train a **classifier** on Animals-10 (section 7A).
- Want boxes for your own set of animals: fine-tune a **detector** on a YOLO-format dataset (section 7B).

### Downloading from Kaggle

Browser: open the link, press **Download**, unzip into a `datasets/` folder in this project.

Command line (optional): create an API token at kaggle.com > Settings > API, save `kaggle.json` as instructed there, then:
```bash
pip install kaggle
kaggle datasets download -d alessiocorrado99/animals10 -p datasets --unzip
```

## 7. Training on Kaggle datasets

### 7A. Classifier (Animals-10 or the 90-animal dataset)

```bash
# Animals-10: the photos are in datasets/raw-img
python training/prepare_classifier_data.py --src datasets/raw-img --out datasets/animals_cls

# 90-animal dataset: the photos are in datasets/animals/animals
# python training/prepare_classifier_data.py --src datasets/animals/animals --out datasets/animals90_cls

python training/train_classifier.py --data datasets/animals_cls --epochs 15
```
The best weights are saved to `models/animal_classifier.pt`. Restart `python run.py`, and `/api/health` will show `"classifier": true`.

Check the actual folder names after unzipping, since Kaggle zips sometimes add an extra folder level.

### 7B. Detector (YOLO-format dataset with boxes)

1. Unzip the dataset into `datasets/wildlife` (or similar).
2. Copy `training/data.yaml.example` to `training/data.yaml` and fix the paths and class names to match the dataset's own `data.yaml`.
3. Train:
   ```bash
   python training/train_detector.py --data training/data.yaml --epochs 50
   ```
Weights go to `models/animal_detector.pt`. **Note:** once this file exists, the app uses it *instead of* the COCO model, so it will only recognise the classes it was trained on. Delete the file to go back to the default.

**Training time:** on CPU, a classifier with 15 epochs takes a long time on 20K+ images; use `--epochs 5` to try it first. With an NVIDIA GPU add `--device 0`. Free GPUs are available in Kaggle Notebooks or Google Colab if you'd rather train there and copy the `.pt` file into `models/`.

## 8. Troubleshooting

| Problem | Fix |
|---|---|
| `pip install` fails on torch | Use Python 3.10-3.12. On Windows, update pip: `python -m pip install -U pip`. |
| Model download fails on first run | You need internet once. Or download `yolov8s.pt` manually from the Ultralytics releases page and put it in the project root. |
| Page says "Could not reach the server" | `python run.py` isn't running, or it's on another port. |
| `Address already in use` | Another program uses port 8000; run with `PORT=9000`. |
| An animal isn't detected | Lower the confidence slider. The default model only knows 10 animals; train a classifier (7A) for more. |
| iPhone `.heic` photos fail | `pip install pillow-heif`, restart the server. (Some browsers can't preview HEIC; converting to JPG is easiest.) |
| Slow first request | The model warms up on the first image; later ones are faster. |

## Project layout

```
animal-detector/
├── run.py                       start the web app
├── requirements.txt
├── app/
│   ├── main.py                  FastAPI server and /api/detect
│   ├── detector.py              model loading and inference
│   └── static/index.html        the drag-and-drop web page
├── training/
│   ├── prepare_classifier_data.py
│   ├── train_classifier.py
│   ├── train_detector.py
│   └── data.yaml.example
├── scripts/test_image.py        command-line test
├── notebooks/animal_detection_demo.ipynb
├── models/                      your trained weights go here
└── sample_images/               put test photos here
```

## Limits to know about

- The default model finds the 10 COCO animals only. Anything else (fish, insects, reptiles) needs your own trained model.
- Confidence is a model score, not a guarantee. Small, blurry or partly hidden animals can be missed or mislabelled.
- Fine-tuning on a small dataset can make the model worse at everything outside that dataset.
