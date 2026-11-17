"""Train + evaluate + optimize.

1. Classical baseline: HOG features (OpenCV) + SVM (scikit-learn)
2. Deep model: DefectNet (PyTorch) with augmentation, AdamW, cosine LR, early stopping
3. Optimization: ONNX export, INT8 dynamic quantization, latency benchmark
Writes artifacts/{defectnet.pt, defectnet.onnx, defectnet.int8.onnx, metrics.json}
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from torch.utils.data import DataLoader, Dataset

from ml.dataset import CLASSES, IMG_SIZE, load_split
from ml.model import DefectNet

ART = Path("artifacts")


# ---------------------------------------------------------------- baseline
def _hog(im: np.ndarray, cell: int = 16, bins: int = 9) -> np.ndarray:
    """Histogram of Oriented Gradients (Dalal & Triggs) with 2x2-cell L2-Hys blocks."""
    im = cv2.equalizeHist(im).astype(np.float32)
    gx, gy = cv2.Sobel(im, cv2.CV_32F, 1, 0, ksize=1), cv2.Sobel(im, cv2.CV_32F, 0, 1, ksize=1)
    mag, ang = cv2.cartToPolar(gx, gy, angleInDegrees=True)
    b = ((ang % 180) / (180 / bins)).astype(int) % bins
    n = im.shape[0] // cell
    onehot = np.zeros((n * cell, n * cell, bins), np.float32)
    np.put_along_axis(onehot, b[..., None], mag[..., None], axis=2)
    hist = onehot.reshape(n, cell, n, cell, bins).sum((1, 3))
    blocks = np.concatenate([hist[:-1, :-1], hist[1:, :-1], hist[:-1, 1:], hist[1:, 1:]], -1)
    blocks /= np.linalg.norm(blocks, axis=-1, keepdims=True) + 1e-6
    blocks = np.minimum(blocks, 0.2)
    blocks /= np.linalg.norm(blocks, axis=-1, keepdims=True) + 1e-6
    return blocks.ravel()


def hog_features(imgs: np.ndarray) -> np.ndarray:
    return np.stack([_hog(im) for im in imgs])


def train_baseline(xtr, ytr, xte, yte) -> dict:
    t0 = time.time()
    clf = make_pipeline(StandardScaler(), SVC(C=10, kernel="rbf", gamma="scale"))
    clf.fit(hog_features(xtr), ytr)
    pred = clf.predict(hog_features(xte))
    return {
        "name": "HOG + SVM (scikit-learn)",
        "accuracy": float((pred == yte).mean()),
        "macro_f1": float(f1_score(yte, pred, average="macro")),
        "train_seconds": round(time.time() - t0, 1),
    }


# ---------------------------------------------------------------- CNN
class ImgDS(Dataset):
    def __init__(self, x, y, augment: bool):
        self.x, self.y, self.augment = x, y, augment

    def __len__(self):
        return len(self.x)

    def __getitem__(self, i):
        im = self.x[i].astype(np.float32) / 255.0
        if self.augment:
            if np.random.rand() < 0.5:
                im = im[:, ::-1]
            if np.random.rand() < 0.5:
                im = im[::-1, :]
            im = np.rot90(im, np.random.randint(4))
            im = im * np.random.uniform(0.8, 1.2) + np.random.uniform(-0.08, 0.08)
            im = im + np.random.normal(0, 0.015, im.shape)
        im = np.ascontiguousarray(im, dtype=np.float32)
        im = (im - im.mean()) / (im.std() + 1e-6)
        return torch.from_numpy(im)[None], int(self.y[i])


def evaluate(model, loader):
    model.eval()
    preds, ys = [], []
    with torch.no_grad():
        for xb, yb in loader:
            preds.append(model(xb).argmax(1).numpy())
            ys.append(yb.numpy())
    return np.concatenate(preds), np.concatenate(ys)


def train_cnn(xtr, ytr, xva, yva, epochs: int, patience: int = 4) -> tuple[DefectNet, list[dict]]:
    torch.manual_seed(0)
    np.random.seed(0)
    model = DefectNet(len(CLASSES))
    tl = DataLoader(ImgDS(xtr, ytr, True), batch_size=64, shuffle=True, num_workers=2)
    vl = DataLoader(ImgDS(xva, yva, False), batch_size=128)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=3e-3, total_steps=epochs * len(tl))
    loss_fn = nn.CrossEntropyLoss(label_smoothing=0.05)
    best, best_state, bad, history = -1.0, None, 0, []
    for ep in range(1, epochs + 1):
        model.train()
        t0, tot = time.time(), 0.0
        for xb, yb in tl:
            opt.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            opt.step()
            sched.step()
            tot += loss.item() * len(xb)
        p, y = evaluate(model, vl)
        f1 = f1_score(y, p, average="macro")
        history.append({"epoch": ep, "train_loss": round(tot / len(tl.dataset), 4), "val_macro_f1": round(float(f1), 4)})
        print(f"epoch {ep:2d} loss {tot / len(tl.dataset):.4f} val_f1 {f1:.4f} ({time.time() - t0:.1f}s)")
        if f1 > best:
            best, best_state, bad = f1, {k: v.clone() for k, v in model.state_dict().items()}, 0
        else:
            bad += 1
            if bad >= patience:
                print("early stopping")
                break
    model.load_state_dict(best_state)
    return model, history


# ---------------------------------------------------------------- optimization
def export_and_benchmark(model: DefectNet, xte, yte, xcal) -> list[dict]:
    """Export to ONNX, then INT8 *static* (QDQ) quantization calibrated on training images.

    Dynamic quantization was tried first: it shrinks the file but runs ~7x slower on CPU because
    ConvInteger kernels are unoptimized, so static per-channel QDQ is used instead.
    """
    import onnxruntime as ort
    from onnxruntime.quantization import (CalibrationDataReader, CalibrationMethod, QuantFormat, QuantType,
                                          quantize_static)
    from onnxruntime.quantization.shape_inference import quant_pre_process

    model.eval()
    fp32, prep, int8 = ART / "defectnet.onnx", ART / "defectnet.prep.onnx", ART / "defectnet.int8.onnx"
    torch.onnx.export(model, torch.randn(1, 1, IMG_SIZE, IMG_SIZE), str(fp32), input_names=["image"],
                      output_names=["logits"], dynamic_axes={"image": {0: "batch"}, "logits": {0: "batch"}},
                      opset_version=17, dynamo=False)

    class Calib(CalibrationDataReader):
        def __init__(self):
            ds = ImgDS(xcal, np.zeros(len(xcal)), False)
            self.it = iter([{"image": ds[i][0][None].numpy()} for i in range(len(ds))])

        def get_next(self):
            return next(self.it, None)

    quant_pre_process(str(fp32), str(prep))
    quantize_static(str(prep), str(int8), Calib(), quant_format=QuantFormat.QDQ, per_channel=True,
                    activation_type=QuantType.QUInt8, weight_type=QuantType.QInt8,
                    calibrate_method=CalibrationMethod.Percentile,
                    extra_options={"CalibPercentile": 99.99})
    prep.unlink()

    batch = np.stack([ImgDS(xte, yte, False)[i][0].numpy() for i in range(len(xte))])
    one = batch[:1]
    rows = []

    def bench(name, fn, size_mb):
        for _ in range(10):
            fn(one)
        t0 = time.perf_counter()
        for _ in range(100):
            fn(one)
        lat = (time.perf_counter() - t0) / 100 * 1000
        pred = np.concatenate([fn(batch[i:i + 128]).argmax(1) for i in range(0, len(batch), 128)])
        rows.append({"runtime": name, "latency_ms": round(lat, 2), "size_mb": round(size_mb, 2),
                     "accuracy": round(float((pred == yte).mean()), 4)})

    torch.set_num_threads(1)
    with torch.no_grad():
        bench("PyTorch FP32", lambda b: model(torch.from_numpy(b)).numpy(), (ART / "defectnet.pt").stat().st_size / 1e6)
    so = ort.SessionOptions()
    so.intra_op_num_threads = 1
    for name, path in [("ONNX Runtime FP32", fp32), ("ONNX Runtime INT8", int8)]:
        s = ort.InferenceSession(str(path), so, providers=["CPUExecutionProvider"])
        bench(name, lambda b, s=s: s.run(None, {"image": b})[0], path.stat().st_size / 1e6)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=Path("data"))
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--export-only", action="store_true",
                    help="reuse artifacts/defectnet.pt; redo ONNX export, quantization and benchmarks")
    a = ap.parse_args()
    ART.mkdir(exist_ok=True)

    xtr, ytr = load_split(a.data, "train")
    xva, yva = load_split(a.data, "val")
    xte, yte = load_split(a.data, "test")
    print(f"train {len(xtr)} / val {len(xva)} / test {len(xte)}")
    xcal = xtr[np.random.default_rng(0).choice(len(xtr), 200, replace=False)]

    if a.export_only:
        model = DefectNet(len(CLASSES))
        model.load_state_dict(torch.load(ART / "defectnet.pt", weights_only=True))
        metrics = json.loads((ART / "metrics.json").read_text())
        metrics["runtimes"] = export_and_benchmark(model, xte, yte, xcal)
        for r in metrics["runtimes"]:
            print(r)
        (ART / "metrics.json").write_text(json.dumps(metrics, indent=2))
        return

    print("== Baseline: HOG + SVM")
    base = train_baseline(xtr, ytr, xte, yte)
    print(base)

    print("== DefectNet (PyTorch)")
    t0 = time.time()
    model, history = train_cnn(xtr, ytr, xva, yva, a.epochs)
    train_s = time.time() - t0
    torch.save(model.state_dict(), ART / "defectnet.pt")
    p, y = evaluate(model, DataLoader(ImgDS(xte, yte, False), batch_size=128))
    print(classification_report(y, p, target_names=CLASSES, digits=4))
    cnn = {"name": "DefectNet CNN (PyTorch)", "accuracy": float((p == y).mean()),
           "macro_f1": float(f1_score(y, p, average="macro")), "train_seconds": round(train_s, 1)}

    print("== Optimization")
    runtimes = export_and_benchmark(model, xte, yte, xcal)
    for r in runtimes:
        print(r)

    metrics = {
        "classes": CLASSES,
        "test_size": int(len(yte)),
        "models": [base, cnn],
        "per_class": classification_report(y, p, target_names=CLASSES, output_dict=True),
        "confusion_matrix": confusion_matrix(y, p).tolist(),
        "history": history,
        "runtimes": runtimes,
        "params": sum(p.numel() for p in model.parameters()),
    }
    (ART / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print("saved artifacts/")


if __name__ == "__main__":
    main()
