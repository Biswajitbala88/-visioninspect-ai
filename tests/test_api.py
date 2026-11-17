import json
from pathlib import Path

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.main import app
from ml.dataset import CLASSES, make_sample

client = TestClient(app)
ART = Path(__file__).resolve().parent.parent / "artifacts"
needs_model = pytest.mark.skipif(not (ART / "defectnet.pt").exists(), reason="run `python -m ml.train` first")


def png(img: np.ndarray) -> bytes:
    return cv2.imencode(".png", img)[1].tobytes()


def test_health():
    r = client.get("/api/health")
    assert r.status_code == 200 and r.json()["status"] == "ok"


def test_rejects_non_image():
    r = client.post("/api/inspect", files={"file": ("x.txt", b"not an image", "text/plain")})
    assert r.status_code == 415


@needs_model
@pytest.mark.parametrize("label", CLASSES)
def test_inspect_fresh_samples(label):
    rng = np.random.default_rng(1234)  # different seed than training data
    hits = 0
    for _ in range(10):
        r = client.post("/api/inspect", files={"file": ("s.png", png(make_sample(label, rng)), "image/png")})
        assert r.status_code == 200
        body = r.json()
        assert set(body["probabilities"]) == set(CLASSES)
        assert abs(sum(body["probabilities"].values()) - 1) < 1e-3
        hits += body["label"] == label
    assert hits >= 8, f"{label}: only {hits}/10 correct"


@needs_model
def test_defect_is_localized():
    rng = np.random.default_rng(7)
    body = client.post("/api/inspect", files={"file": ("s.png", png(make_sample("spot", rng)), "image/png")}).json()
    if body["label"] != "ok":
        assert body["regions"], "a defect prediction should come with at least one localized region"


@needs_model
def test_metrics():
    m = client.get("/api/metrics").json()
    assert m["classes"] == CLASSES
    assert len(m["confusion_matrix"]) == len(CLASSES)


def test_report_requires_inspection():
    r = client.post("/api/report", data=json.dumps({"inspection": {}}), headers={"Content-Type": "application/json"})
    assert r.status_code == 400
