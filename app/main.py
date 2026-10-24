"""FastAPI server: serves the web page and the /api/detect endpoint."""
from __future__ import annotations

from contextlib import asynccontextmanager
from io import BytesIO
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.staticfiles import StaticFiles
from PIL import Image, ImageOps, UnidentifiedImageError

try:  # optional HEIC/HEIF support (pip install pillow-heif)
    from pillow_heif import register_heif_opener
    register_heif_opener()
except ImportError:
    pass

from app.detector import AnimalDetector

MAX_BYTES = 15 * 1024 * 1024  # 15 MB upload limit
STATIC_DIR = Path(__file__).parent / "static"

state: dict = {}


@asynccontextmanager
async def lifespan(_: FastAPI):
    print("Loading model... (first run downloads weights, needs internet)")
    state["detector"] = AnimalDetector()
    print("Model ready:", state["detector"].detector_name)
    yield
    state.clear()


app = FastAPI(title="Animal Detector", lifespan=lifespan)


@app.get("/api/health")
def health():
    d = state.get("detector")
    return {"status": "ok", "model": d.detector_name if d else None,
            "classifier": bool(d and d.classifier)}


@app.post("/api/detect")
async def detect(file: UploadFile = File(...), conf: float = Form(0.35)):
    data = await file.read()
    if not data:
        raise HTTPException(400, "The file is empty.")
    if len(data) > MAX_BYTES:
        raise HTTPException(413, "Image is larger than 15 MB. Try a smaller one.")
    try:
        image = Image.open(BytesIO(data))
        image = ImageOps.exif_transpose(image).convert("RGB")
    except (UnidentifiedImageError, OSError):
        raise HTTPException(400, "That file is not a readable image. Use JPG, PNG, WEBP, BMP, GIF or TIFF.")

    conf = min(max(conf, 0.05), 0.95)
    return await run_in_threadpool(state["detector"].analyze, image, conf)


# Must come last so it doesn't shadow /api routes.
app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="static")
