"""Synthetic surface-defect dataset (OpenCV).

Generates grayscale metal/fabric-like surface textures and injects one of
four defect types. Real inspection datasets (e.g. NEU-DET, MVTec AD) follow
the same structure, so the training code works unchanged on real data
placed in `data/<split>/<class>/*.png`.

Classes: ok, scratch, spot, crack
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

CLASSES = ["ok", "scratch", "spot", "crack"]
IMG_SIZE = 128


def _texture(rng: np.random.Generator, size: int = IMG_SIZE) -> np.ndarray:
    """Brushed-metal style background: low-freq noise + directional grain."""
    base = rng.uniform(110, 170)
    low = cv2.resize(rng.normal(0, 1, (8, 8)).astype(np.float32), (size, size),
                     interpolation=cv2.INTER_CUBIC) * rng.uniform(4, 12)
    grain = rng.normal(0, 1, (size, size)).astype(np.float32)
    k = int(rng.integers(5, 15))
    kernel = np.zeros((k, k), np.float32)
    kernel[k // 2, :] = 1.0 / k
    angle = rng.uniform(0, 180)
    rot = cv2.getRotationMatrix2D((k / 2, k / 2), angle, 1.0)
    kernel = cv2.warpAffine(kernel, rot, (k, k))
    kernel /= max(kernel.sum(), 1e-6)
    grain = cv2.filter2D(grain, -1, kernel) * rng.uniform(6, 14)
    img = base + low + grain + rng.normal(0, 2.5, (size, size))
    return img.astype(np.float32)


def _scratch(img, rng):
    n = int(rng.integers(1, 3))
    for _ in range(n):
        p1 = rng.integers(10, IMG_SIZE - 10, 2)
        ang = rng.uniform(0, np.pi)
        length = rng.uniform(40, 100)
        p2 = (p1 + length * np.array([np.cos(ang), np.sin(ang)])).astype(int)
        shade = rng.choice([-1, 1]) * rng.uniform(35, 70)
        mask = np.zeros_like(img)
        cv2.line(mask, tuple(int(v) for v in p1), tuple(int(v) for v in p2), 1.0,
                 int(rng.integers(1, 3)), cv2.LINE_AA)
        img += shade * cv2.GaussianBlur(mask, (3, 3), 0)
    return img


def _spot(img, rng):
    n = int(rng.integers(1, 4))
    yy, xx = np.mgrid[:IMG_SIZE, :IMG_SIZE]
    for _ in range(n):
        cx, cy = rng.integers(15, IMG_SIZE - 15, 2)
        r = rng.uniform(4, 12)
        blob = np.exp(-(((xx - cx) ** 2) / (2 * r**2) + ((yy - cy) ** 2) / (2 * (r * rng.uniform(0.6, 1.4)) ** 2)))
        img -= blob * rng.uniform(40, 90)
    return img


def _crack(img, rng):
    pts = [rng.integers(30, IMG_SIZE - 30, 2).astype(float)]
    ang = rng.uniform(0, 2 * np.pi)
    for _ in range(int(rng.integers(12, 25))):
        ang += rng.normal(0, 0.6)
        step = rng.uniform(3, 7)
        nxt = np.clip(pts[-1] + step * np.array([np.cos(ang), np.sin(ang)]), 0, IMG_SIZE - 1)
        pts.append(nxt)
    mask = np.zeros_like(img)
    cv2.polylines(mask, [np.array(pts, np.int32)], False, 1.0, 1, cv2.LINE_AA)
    # small branch
    if rng.random() < 0.6:
        b0 = pts[len(pts) // 2]
        ang2 = ang + rng.choice([-1, 1]) * rng.uniform(0.6, 1.4)
        b1 = b0 + rng.uniform(10, 25) * np.array([np.cos(ang2), np.sin(ang2)])
        cv2.line(mask, tuple(b0.astype(int)), tuple(b1.astype(int)), 1.0, 1, cv2.LINE_AA)
    img -= mask * rng.uniform(50, 90)
    return img


DEFECTS = {"ok": lambda img, rng: img, "scratch": _scratch, "spot": _spot, "crack": _crack}


def make_sample(label: str, rng: np.random.Generator) -> np.ndarray:
    img = DEFECTS[label](_texture(rng), rng)
    # global lighting variation, as from an uneven factory light source
    gx = np.linspace(-1, 1, IMG_SIZE)[None, :] * rng.uniform(-15, 15)
    gy = np.linspace(-1, 1, IMG_SIZE)[:, None] * rng.uniform(-15, 15)
    img = img + gx + gy
    return np.clip(img, 0, 255).astype(np.uint8)


def generate(out: Path, per_class: int, seed: int = 0) -> None:
    rng = np.random.default_rng(seed)
    splits = {"train": 0.7, "val": 0.15, "test": 0.15}
    for label in CLASSES:
        idx = rng.permutation(per_class)
        start = 0
        for split, frac in splits.items():
            n = int(round(per_class * frac))
            d = out / split / label
            d.mkdir(parents=True, exist_ok=True)
            for i in idx[start:start + n]:
                cv2.imwrite(str(d / f"{label}_{i:05d}.png"), make_sample(label, rng))
            start += n
    print(f"Wrote {per_class * len(CLASSES)} images to {out}")


def load_split(root: Path, split: str) -> tuple[np.ndarray, np.ndarray]:
    xs, ys = [], []
    for ci, label in enumerate(CLASSES):
        for p in sorted((root / split / label).glob("*.png")):
            xs.append(cv2.imread(str(p), cv2.IMREAD_GRAYSCALE))
            ys.append(ci)
    return np.stack(xs), np.array(ys)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("data"))
    ap.add_argument("--per-class", type=int, default=800)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    generate(a.out, a.per_class, a.seed)
