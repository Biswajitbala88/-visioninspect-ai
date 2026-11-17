"""DefectNet: compact CNN for 128x128 grayscale surface inspection + Grad-CAM."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def _block(cin: int, cout: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(cin, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
        nn.Conv2d(cout, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
    )


class DefectNet(nn.Module):
    def __init__(self, num_classes: int = 4):
        super().__init__()
        self.b1, self.b2, self.b3, self.b4 = _block(1, 16), _block(16, 32), _block(32, 64), _block(64, 128)
        self.pool = nn.MaxPool2d(2)
        self.head = nn.Sequential(nn.Dropout(0.3), nn.Linear(128, num_classes))

    def features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(self.b1(x))   # 64
        x = self.pool(self.b2(x))   # 32
        x = self.pool(self.b3(x))   # 16
        return self.b4(x)           # 16x16 feature map used for Grad-CAM

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        f = self.features(x)
        return self.head(F.adaptive_avg_pool2d(f, 1).flatten(1))


def preprocess(gray: np.ndarray) -> torch.Tensor:
    """uint8 HxW (already 128x128) -> normalized 1x1xHxW tensor (per-image standardization)."""
    x = gray.astype(np.float32) / 255.0
    x = (x - x.mean()) / (x.std() + 1e-6)
    return torch.from_numpy(x)[None, None]


def grad_cam(model: DefectNet, x: torch.Tensor, class_idx: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Returns (probabilities, cam[0..1] at input resolution)."""
    model.eval()
    feats = model.features(x)
    feats.retain_grad()
    logits = model.head(F.adaptive_avg_pool2d(feats, 1).flatten(1))
    probs = logits.softmax(1)[0].detach().numpy()
    idx = int(probs.argmax()) if class_idx is None else class_idx
    model.zero_grad()
    logits[0, idx].backward()
    weights = feats.grad.mean(dim=(2, 3), keepdim=True)
    cam = F.relu((weights * feats).sum(1, keepdim=True))
    cam = F.interpolate(cam, size=x.shape[-2:], mode="bilinear", align_corners=False)[0, 0].detach().numpy()
    cam = cam / (cam.max() + 1e-8)
    return probs, cam
