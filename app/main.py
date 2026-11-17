"""VisionInspect AI — FastAPI service."""
from __future__ import annotations

import json
import random
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from app import inference, llm

ROOT = Path(__file__).resolve().parent.parent
MAX_BYTES = 8 * 1024 * 1024

app = FastAPI(title="VisionInspect AI", version="1.0.0",
              description="Surface-defect inspection, object detection and GenAI QA reports.")
app.mount("/static", StaticFiles(directory=ROOT / "app" / "static"), name="static")


async def _read(file: UploadFile) -> bytes:
    data = await file.read()
    if not data:
        raise HTTPException(400, "Empty upload.")
    if len(data) > MAX_BYTES:
        raise HTTPException(413, "Image larger than 8 MB.")
    return data


def _decode(data: bytes):
    try:
        return inference.decode_image(data)
    except ValueError as e:
        raise HTTPException(415, str(e)) from e


@app.get("/", include_in_schema=False)
def index():
    return FileResponse(ROOT / "app" / "static" / "index.html")


@app.get("/api/health")
def health():
    return {"status": "ok", "model_loaded": (inference.ART / "defectnet.pt").exists(),
            "llm_model": llm.MODEL}


@app.get("/api/metrics")
def metrics():
    p = inference.ART / "metrics.json"
    if not p.exists():
        raise HTTPException(404, "Run `python -m ml.train` first.")
    return json.loads(p.read_text())


@app.get("/api/samples")
def samples(n: int = 8):
    """Random held-out test images (never seen in training)."""
    files = sorted((ROOT / "data" / "test").glob("*/*.png"))
    pick = random.sample(files, min(n, len(files)))
    return [{"label": f.parent.name, "url": f"/api/samples/{f.parent.name}/{f.name}"} for f in pick]


@app.get("/api/samples/{label}/{name}", include_in_schema=False)
def sample_file(label: str, name: str):
    p = (ROOT / "data" / "test" / label / name).resolve()
    if not p.is_relative_to(ROOT / "data" / "test") or not p.exists():
        raise HTTPException(404)
    return FileResponse(p)


@app.post("/api/inspect")
async def inspect(file: UploadFile = File(...)):
    return inference.inspect_surface(_decode(await _read(file)))


@app.post("/api/detect")
async def detect(file: UploadFile = File(...), threshold: float = Form(0.4)):
    return inference.detect_objects(_decode(await _read(file)), threshold)


class ReportRequest(BaseModel):
    inspection: dict
    notes: str = ""


@app.post("/api/report")
def report(req: ReportRequest):
    img = req.inspection.get("input_preview")
    if not img:
        raise HTTPException(400, "inspection.input_preview missing — pass the /api/inspect response.")
    try:
        return llm.generate_report(img, req.inspection, req.notes)
    except llm.LLMUnavailable as e:
        raise HTTPException(503, str(e)) from e
