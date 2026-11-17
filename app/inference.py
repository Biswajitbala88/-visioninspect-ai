"""Model serving: defect classification + Grad-CAM localization, and COCO object detection."""
from __future__ import annotations

import base64
import time
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
import torch

from ml.dataset import CLASSES, IMG_SIZE
from ml.model import DefectNet, grad_cam, preprocess

ART = Path(__file__).resolve().parent.parent / "artifacts"


def decode_image(data: bytes) -> np.ndarray:
    arr = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if arr is None:
        raise ValueError("Could not decode image (expected PNG/JPEG/BMP/WebP).")
    return arr


def to_png_b64(img: np.ndarray) -> str:
    ok, buf = cv2.imencode(".png", img)
    return "data:image/png;base64," + base64.b64encode(buf.tobytes()).decode()


# ------------------------------------------------------------ defect inspection
@lru_cache(maxsize=1)
def defect_model() -> DefectNet:
    m = DefectNet(len(CLASSES))
    m.load_state_dict(torch.load(ART / "defectnet.pt", map_location="cpu", weights_only=True))
    m.eval()
    return m


@lru_cache(maxsize=1)
def onnx_session():
    import onnxruntime as ort
    path = ART / "defectnet.int8.onnx"
    return ort.InferenceSession(str(path), providers=["CPUExecutionProvider"]) if path.exists() else None


def inspect_surface(bgr: np.ndarray) -> dict:
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
    x = preprocess(gray)

    t0 = time.perf_counter()
    probs, cam = grad_cam(defect_model(), x.clone().requires_grad_(False))
    torch_ms = (time.perf_counter() - t0) * 1000

    onnx_ms, onnx_label = None, None
    sess = onnx_session()
    if sess is not None:
        t0 = time.perf_counter()
        logits = sess.run(None, {"image": x.numpy()})[0]
        onnx_ms = (time.perf_counter() - t0) * 1000
        onnx_label = CLASSES[int(logits.argmax())]

    label = CLASSES[int(probs.argmax())]
    regions = []
    if label != "ok":
        mask = (cam >= 0.5).astype(np.uint8)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in sorted(contours, key=cv2.contourArea, reverse=True)[:3]:
            x0, y0, w, h = cv2.boundingRect(c)
            if w * h >= 16:
                regions.append({"x": x0, "y": y0, "w": w, "h": h,
                                "area_pct": round(100 * w * h / IMG_SIZE**2, 1)})

    # visualization at 384px: original | heatmap overlay with boxes
    S, k = 384, 384 / IMG_SIZE
    base = cv2.cvtColor(cv2.resize(gray, (S, S), interpolation=cv2.INTER_CUBIC), cv2.COLOR_GRAY2BGR)
    heat = cv2.applyColorMap((cv2.resize(cam, (S, S)) * 255).astype(np.uint8), cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(base, 0.55, heat, 0.45, 0) if label != "ok" else base.copy()
    for r in regions:
        cv2.rectangle(overlay, (int(r["x"] * k), int(r["y"] * k)),
                      (int((r["x"] + r["w"]) * k), int((r["y"] + r["h"]) * k)), (255, 255, 255), 2)

    return {
        "label": label,
        "confidence": round(float(probs.max()), 4),
        "probabilities": {c: round(float(p), 4) for c, p in zip(CLASSES, probs)},
        "regions": regions,
        "latency_ms": {"pytorch_gradcam": round(torch_ms, 2),
                       "onnx_int8": round(onnx_ms, 2) if onnx_ms is not None else None},
        "onnx_int8_agrees": onnx_label == label if onnx_label else None,
        "input_preview": to_png_b64(base),
        "heatmap": to_png_b64(overlay),
    }


# ------------------------------------------------------------ object detection
@lru_cache(maxsize=1)
def detector():
    from torchvision.models.detection import SSDLite320_MobileNet_V3_Large_Weights, ssdlite320_mobilenet_v3_large
    w = SSDLite320_MobileNet_V3_Large_Weights.COCO_V1
    m = ssdlite320_mobilenet_v3_large(weights=w)
    m.eval()
    return m, w.meta["categories"]


PALETTE = [(66, 135, 245), (245, 130, 48), (60, 180, 75), (230, 25, 75), (145, 30, 180), (70, 240, 240)]


def detect_objects(bgr: np.ndarray, threshold: float = 0.4) -> dict:
    model, cats = detector()
    h, w = bgr.shape[:2]
    scale = 640 / max(h, w)
    if scale < 1:
        bgr = cv2.resize(bgr, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    x = torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0
    t0 = time.perf_counter()
    with torch.no_grad():
        out = model([x])[0]
    ms = (time.perf_counter() - t0) * 1000

    vis, dets = bgr.copy(), []
    for box, lab, sc in zip(out["boxes"], out["labels"], out["scores"]):
        if sc < threshold:
            continue
        x0, y0, x1, y1 = [int(v) for v in box.tolist()]
        name = cats[int(lab)]
        col = PALETTE[int(lab) % len(PALETTE)]
        dets.append({"label": name, "score": round(float(sc), 3), "box": [x0, y0, x1, y1]})
        cv2.rectangle(vis, (x0, y0), (x1, y1), col, 2)
        txt = f"{name} {sc:.2f}"
        (tw, th), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(vis, (x0, max(0, y0 - th - 6)), (x0 + tw + 6, y0), col, -1)
        cv2.putText(vis, txt, (x0 + 3, y0 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return {"model": "SSDLite320 MobileNetV3 (COCO, torchvision)", "detections": dets,
            "latency_ms": round(ms, 1), "image": to_png_b64(vis)}
