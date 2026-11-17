# VisionInspect AI — Technical Documentation

> Version 1.0 · Python 3.12 · PyTorch 2.15 (CPU) · OpenCV 5.0 · scikit-learn 1.9 · ONNX Runtime 1.32 · FastAPI

## Table of contents

1. [Overview](#1-overview)
2. [System architecture](#2-system-architecture)
3. [Installation and setup](#3-installation-and-setup)
4. [Data pipeline](#4-data-pipeline)
5. [Classical baseline: HOG + SVM](#5-classical-baseline-hog--svm)
6. [Deep model: DefectNet](#6-deep-model-defectnet)
7. [Evaluation and results](#7-evaluation-and-results)
8. [Model optimization: ONNX and INT8](#8-model-optimization-onnx-and-int8)
9. [Inference pipeline](#9-inference-pipeline)
10. [Object detection module](#10-object-detection-module)
11. [GenAI inspection reports](#11-genai-inspection-reports)
12. [REST API reference](#12-rest-api-reference)
13. [Web interface](#13-web-interface)
14. [Testing](#14-testing)
15. [Deployment](#15-deployment)
16. [Configuration](#16-configuration)
17. [Using real datasets](#17-using-real-datasets)
18. [Limitations and future work](#18-limitations-and-future-work)
19. [Troubleshooting](#19-troubleshooting)
20. [Demo walkthrough](#20-demo-walkthrough)

---

## 1. Overview

VisionInspect AI automates visual quality inspection of manufactured surfaces (metal sheets, coated
panels and similar). For each surface image it:

1. **Classifies** it as `ok`, `scratch`, `spot` or `crack`.
2. **Localizes** the defect with a Grad-CAM heatmap and bounding boxes.
3. **Reports**: a large language model turns the result into a structured QA report with disposition,
   severity, root causes and corrective actions.

A second module runs general-purpose **object detection** (80 COCO classes) on any photo.

### Headline results

| Metric | Value |
|---|---|
| CNN test accuracy (480 held-out images) | **97.3%** (macro-F1 0.973) |
| Classical baseline (HOG + SVM) | 45.8% |
| INT8 inference latency (CPU, 1 thread) | **3.8 ms** (vs 10.5 ms in PyTorch) |
| INT8 model size | **0.32 MB** (vs 1.20 MB) |
| Accuracy cost of INT8 | −0.4 points |

---

## 2. System architecture

```
┌─────────────────────────── OFFLINE: ml/ ───────────────────────────┐
│                                                                    │
│  dataset.py ──► data/{train,val,test}/{ok,scratch,spot,crack}/*.png │
│                        │                                           │
│  train.py   ┌──────────┴───────────┐                               │
│             ▼                      ▼                               │
│      HOG + SVM baseline     DefectNet CNN (PyTorch)                │
│             │                      │                               │
│             │              ONNX export (opset 17)                  │
│             │                      │                               │
│             │          INT8 static quantization (QDQ)              │
│             └──────────┬───────────┘                               │
│                        ▼                                           │
│   artifacts/ defectnet.pt · defectnet.onnx · defectnet.int8.onnx   │
│              metrics.json                                          │
└────────────────────────────────────────────────────────────────────┘
                         │
┌─────────────────────────── ONLINE: app/ ───────────────────────────┐
│                                                                    │
│  main.py (FastAPI)                                                 │
│   ├─ POST /api/inspect ─► inference.inspect_surface()              │
│   │      PyTorch CNN + Grad-CAM ─► OpenCV contours ─► boxes        │
│   │      ONNX INT8 run in parallel for latency/agreement check     │
│   ├─ POST /api/detect  ─► inference.detect_objects()               │
│   │      torchvision SSDLite320-MobileNetV3 (COCO)                 │
│   ├─ POST /api/report  ─► llm.generate_report()                    │
│   │      Claude: image + CNN JSON ─► JSON-schema-constrained report │
│   ├─ GET  /api/metrics, /api/samples, /api/health                  │
│   └─ GET  /            ─► static/index.html (single-page UI)       │
└────────────────────────────────────────────────────────────────────┘
```

### File layout

| Path | Purpose |
|---|---|
| `ml/dataset.py` | Synthetic defect generator and dataset loader |
| `ml/model.py` | `DefectNet` architecture, preprocessing, Grad-CAM |
| `ml/train.py` | Baseline, CNN training, evaluation, ONNX export, quantization, benchmarks |
| `app/main.py` | FastAPI app, routes, validation, error mapping |
| `app/inference.py` | Model loading (cached), defect inspection, object detection, image encoding |
| `app/llm.py` | LLM report generation, output schema, error handling |
| `app/static/index.html` | Web UI (vanilla HTML/CSS/JS, no build step) |
| `artifacts/` | Trained weights and `metrics.json` |
| `tests/test_api.py` | API and model tests |
| `Dockerfile` | Production container image |

---

## 3. Installation and setup

### Requirements
- Python 3.10+ (developed on 3.12)
- ~1.5 GB disk for dependencies (CPU-only PyTorch)
- No GPU needed

### Steps

```bash
cd vision-inspect-ai
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements-dev.txt      # runtime + pytest/httpx

python -m ml.dataset --per-class 800     # generate data (~5 s)
python -m ml.train                       # train everything (~13 min on 4 CPU cores)

cp .env.example .env                     # optional: add ANTHROPIC_API_KEY
uvicorn app.main:app --reload            # http://localhost:8000
```

Trained artifacts are included in `artifacts/`, so you can skip `ml.train`. You still need
`ml.dataset` because the UI's sample gallery serves images from `data/test/`.

### Command reference

| Command | What it does |
|---|---|
| `python -m ml.dataset [--out data] [--per-class 800] [--seed 0]` | Generate the dataset |
| `python -m ml.train [--data data] [--epochs 20]` | Full pipeline: baseline, CNN, export, quantize, benchmark |
| `python -m ml.train --export-only` | Reuse `defectnet.pt`; redo only ONNX export, quantization and benchmarks |
| `uvicorn app.main:app --reload` | Development server |
| `pytest -q` | Run tests |

---

## 4. Data pipeline

**File:** `ml/dataset.py`

### Why synthetic data
It keeps the project self-contained, reproducible (seeded) and free of licensing issues. The folder layout
matches real datasets, so swapping in real data needs no code changes (see [§17](#17-using-real-datasets)).

### Image generation

Each 128×128 grayscale image is built in layers:

1. **Base texture** (brushed metal look)
   - Random base intensity (110–170)
   - Low-frequency variation: 8×8 Gaussian noise upscaled with bicubic interpolation
   - Directional grain: Gaussian noise convolved with a randomly rotated line kernel (length 5–15 px)
   - Per-pixel sensor noise (σ = 2.5)
2. **Defect injection** (by class)

| Class | How it is drawn |
|---|---|
| `ok` | No defect |
| `scratch` | 1–2 straight anti-aliased lines, 40–100 px long, 1–2 px wide; randomly brighter or darker |
| `spot` | 1–3 dark elliptical Gaussian blobs, radius 4–12 px |
| `crack` | A random walk of 12–25 steps (heading jitter σ = 0.6 rad) drawn as a dark polyline, with a 60% chance of a side branch; starts 30+ px from the edges |

3. **Lighting gradient**: a random linear gradient on both axes (±15 intensity) to simulate uneven
   factory lighting.

### Splits

800 images per class (3,200 total), split per class with a seeded permutation:

| Split | Share | Images |
|---|---|---|
| train | 70% | 2,240 |
| val | 15% | 480 |
| test | 15% | 480 |

The validation set is used only for early stopping and model selection. The test set is used only for
final reporting.

---

## 5. Classical baseline: HOG + SVM

**File:** `ml/train.py` → `_hog()`, `train_baseline()`

A classical baseline shows whether the deep model is worth its cost.

### Features: Histogram of Oriented Gradients
OpenCV 5 removed `cv2.HOGDescriptor` from the main package, so HOG is implemented directly
(Dalal & Triggs):

1. Histogram equalization
2. Gradients with 1-pixel Sobel kernels → magnitude and orientation
3. 9 unsigned orientation bins (0–180°), 16×16 px cells → 8×8 cell grid
4. 2×2-cell blocks with stride 1 → 7×7 blocks × 36 values = **1,764-dim feature vector**
5. L2-Hys normalization (L2 norm, clip at 0.2, renormalize)

The binning is vectorized with `np.put_along_axis`, with no Python loop over pixels.

### Classifier
`StandardScaler` → `SVC(kernel="rbf", C=10, gamma="scale")`

### Result: 45.8% accuracy
The brushed-metal grain produces strong directional gradients in *every* image, so faint scratches and
cracks don't stand out in HOG space. That gap is the justification for a learned feature extractor.

---

## 6. Deep model: DefectNet

**File:** `ml/model.py`

### Architecture

```
Input 1×128×128
 ├─ Block(1→16)    Conv3×3-BN-ReLU ×2 → MaxPool  → 16×64×64
 ├─ Block(16→32)   Conv3×3-BN-ReLU ×2 → MaxPool  → 32×32×32
 ├─ Block(32→64)   Conv3×3-BN-ReLU ×2 → MaxPool  → 64×16×16
 ├─ Block(64→128)  Conv3×3-BN-ReLU ×2            → 128×16×16  ◄── Grad-CAM layer
 ├─ Global average pooling                       → 128
 ├─ Dropout(0.3)
 └─ Linear(128→4)                                → logits
Parameters: 294,228
```

Design choices:
- **Small and VGG-style**: 128×128 grayscale images don't need ImageNet-scale backbones. A small model
  trains on CPU in minutes and runs in milliseconds.
- **No pooling after the last block** keeps a 16×16 feature map, giving Grad-CAM usable spatial resolution.
- **Global average pooling** instead of large fully connected layers cuts parameters and overfitting.
- **Per-image standardization** (zero mean, unit variance per image) removes global brightness
  differences between lighting conditions.

### Training configuration

| Setting | Value |
|---|---|
| Optimizer | AdamW (weight decay 1e-4) |
| LR schedule | OneCycle, max LR 3e-3 |
| Loss | Cross-entropy, label smoothing 0.05 |
| Batch size | 64 |
| Max epochs | 20 |
| Early stopping | Patience 4 on validation macro-F1; best weights restored |
| Seeds | `torch.manual_seed(0)`, `np.random.seed(0)` |

### Augmentation (training only)
- Random horizontal and vertical flips
- Random 90° rotations
- Contrast ×0.8–1.2 and brightness ±0.08
- Gaussian noise (σ = 0.015)

All four defect types are orientation-invariant, so the geometric augmentations preserve labels.

### Training run
Training stopped early at epoch 16; the best checkpoint is from epoch 12 (val macro-F1 0.996). The
validation curve is noisy in early epochs because of the high OneCycle learning rate and BatchNorm
statistics, then settles.

---

## 7. Evaluation and results

All numbers come from `artifacts/metrics.json`, produced by `ml/train.py`.

### Model comparison (test set, 480 images)

| Model | Accuracy | Macro-F1 | Train time |
|---|---|---|---|
| HOG + SVM | 45.8% | 0.457 | 9.5 s |
| DefectNet CNN | **97.3%** | **0.973** | 752 s |

### Per-class metrics (DefectNet)

| Class | Precision | Recall | F1 |
|---|---|---|---|
| ok | 0.935 | 0.958 | 0.947 |
| scratch | 0.967 | 0.967 | 0.967 |
| spot | 0.991 | 0.967 | 0.979 |
| crack | 1.000 | 1.000 | 1.000 |

### Confusion matrix (rows = true, columns = predicted)

| | ok | scratch | spot | crack |
|---|---|---|---|---|
| **ok** | 115 | 4 | 1 | 0 |
| **scratch** | 4 | 116 | 0 | 0 |
| **spot** | 4 | 0 | 116 | 0 |
| **crack** | 0 | 0 | 0 | 120 |

**Reading it:** nearly all errors are between `ok` and very faint scratches or spots. Cracks are never
confused. In production, the costly error is a defect passed as `ok` (8 of 360 defects here). That's why
the UI sends any prediction under 85% confidence to manual review instead of trusting it.

---

## 8. Model optimization: ONNX and INT8

**File:** `ml/train.py` → `export_and_benchmark()`

### Pipeline
1. **ONNX export**: `torch.onnx.export`, opset 17, dynamic batch axis.
2. **Pre-processing**: `quant_pre_process` (shape inference and graph optimization).
3. **Static INT8 quantization**: QDQ format, per-channel INT8 weights, UINT8 activations.
4. **Calibration**: 200 random training images, **99.99th-percentile** range estimation.

### Benchmark method
- Batch size 1, single thread (`torch.set_num_threads(1)`, `intra_op_num_threads = 1`)
- 10 warm-up runs, then the mean of 100 timed runs
- Accuracy re-measured on the full test set for every runtime

### Results

| Runtime | Latency | Size | Accuracy |
|---|---|---|---|
| PyTorch FP32 | 10.46 ms | 1.20 MB | 97.29% |
| ONNX Runtime FP32 | 4.72 ms | 1.18 MB | 97.29% |
| ONNX Runtime INT8 | **3.77 ms** | **0.32 MB** | 96.88% |

### What was tried, and why it changed

| Attempt | Latency | Accuracy | Outcome |
|---|---|---|---|
| Dynamic INT8 quantization | 33.8 ms | 97.1% | **7× slower than FP32 ONNX.** ONNX Runtime's `ConvInteger` kernels are unoptimized on CPU. Rejected. |
| Static QDQ, MinMax calibration | 3.7 ms | 91.3% | Fast, but lost 6 points. Per-image standardization creates outlier activations that stretch the MinMax range and waste INT8 resolution. |
| **Static QDQ, percentile 99.99** | **3.8 ms** | **96.9%** | Clipping the outliers restored accuracy. **Adopted.** |

Lesson: dynamic quantization suits transformer/linear-heavy models; conv nets need static quantization
with careful calibration.

---

## 9. Inference pipeline

**File:** `app/inference.py` → `inspect_surface()`

```
upload bytes
  → cv2.imdecode (any PNG/JPEG/BMP/WebP, colour or gray)
  → grayscale → resize to 128×128 (INTER_AREA)
  → per-image standardization
  ├─► PyTorch DefectNet + Grad-CAM ──► probabilities, 128×128 attention map
  └─► ONNX INT8 session            ──► label (for latency + agreement check)
  → if label ≠ ok:
        threshold CAM at 0.5 → cv2.findContours
        keep the 3 largest contours with bounding-box area ≥ 16 px
  → render 384×384 preview + JET heatmap overlay (45%) + white boxes
  → JSON response
```

### Grad-CAM
1. Forward pass; keep the gradient of the last 128×16×16 feature map.
2. Back-propagate the score of the predicted class.
3. Channel weights = gradients averaged over space.
4. CAM = ReLU(Σ weight × feature map), upsampled bilinearly to 128×128, scaled to 0–1.

### Model loading
Models load lazily on first request and are cached with `functools.lru_cache`, so each process loads them
once. The detector weights (~14 MB) download on first use to `TORCH_HOME`.

---

## 10. Object detection module

**File:** `app/inference.py` → `detect_objects()`

| Item | Detail |
|---|---|
| Model | `ssdlite320_mobilenet_v3_large`, torchvision COCO_V1 weights |
| Classes | 80 COCO categories (person, car, bus, bottle, …) |
| Pre-processing | Downscale so the longest side ≤ 640 px; RGB; scale to [0, 1] |
| Post-processing | Filter by score threshold (default 0.4); draw labelled boxes with OpenCV |
| Latency | ~90 ms per image on CPU |

Example (a street photo): `bus 0.97`, `person 0.99`, `person 0.79`, `person 0.65`.

This module shows pretrained-model usage on real-world photos. It's the starting point for fine-tuning a
detector on custom classes.

---

## 11. GenAI inspection reports

**File:** `app/llm.py`

### Purpose
The CNN outputs a label. A QA team needs a decision and an explanation. The LLM:
- looks at the **image itself** together with the CNN's JSON output and optional operator notes,
- produces a report in a **fixed JSON schema** (no free-text parsing downstream),
- **cross-checks** the CNN and picks `manual_review` when they disagree or confidence is low.

### Request

| Setting | Value |
|---|---|
| Model | `claude-opus-5-5` (override with `CLAUDE_MODEL`) |
| Input | Base64 PNG image + CNN result JSON + operator notes |
| Output control | `output_config.format` = JSON schema (structured outputs) |
| Effort | `low`: a short, well-defined task |
| Reliability | Server-side refusal fallback (`fallbacks: "default"`) |

### Output schema

| Field | Type | Values |
|---|---|---|
| `disposition` | enum | `accept`, `rework`, `scrap`, `manual_review` |
| `severity` | enum | `none`, `minor`, `major`, `critical` |
| `summary` | string | One-paragraph summary |
| `observations` | string[] | What is visible |
| `likely_root_causes` | string[] | Process hypotheses |
| `recommended_actions` | string[] | Corrective steps |
| `model_agreement` | string | Whether the LLM agrees with the CNN, and why |

The service also adds `_meta`: `{model, input_tokens, output_tokens}`.

### Error handling

| Situation | HTTP | Message |
|---|---|---|
| No API key configured | 503 | `Set ANTHROPIC_API_KEY to enable GenAI reports.` |
| Key rejected | 503 | `Anthropic credentials were rejected.` |
| Network failure | 503 | `Could not reach the Anthropic API.` |
| Model declined | 503 | `The model declined to produce a report for this input.` |

Everything else in the app works without an API key.

---

## 12. REST API reference

Base URL: `http://localhost:8000`. Interactive OpenAPI docs: `/docs` (Swagger) and `/redoc`.

### `GET /api/health`
```json
{"status": "ok", "model_loaded": true, "llm_model": "claude-opus-5-5"}
```

### `POST /api/inspect`
Classify and localize a surface defect.

| Field | In | Type | Required |
|---|---|---|---|
| `file` | multipart | image (PNG/JPEG/BMP/WebP, ≤ 8 MB) | yes |

```bash
curl -F file=@data/test/scratch/scratch_00012.png http://localhost:8000/api/inspect
```

```json
{
  "label": "scratch",
  "confidence": 0.9939,
  "probabilities": {"ok": 0.0, "scratch": 0.9939, "spot": 0.0023, "crack": 0.0038},
  "regions": [
    {"x": 22, "y": 0, "w": 80, "h": 45, "area_pct": 22.0},
    {"x": 58, "y": 85, "w": 30, "h": 28, "area_pct": 5.1}
  ],
  "latency_ms": {"pytorch_gradcam": 52.85, "onnx_int8": 2.23},
  "onnx_int8_agrees": true,
  "input_preview": "data:image/png;base64,...",
  "heatmap": "data:image/png;base64,..."
}
```

`regions` coordinates are in the 128×128 model input space. `regions` is empty when `label` is `ok`.

### `POST /api/detect`

| Field | In | Type | Default |
|---|---|---|---|
| `file` | multipart | image | required |
| `threshold` | form | float 0–1 | 0.4 |

```bash
curl -F file=@street.jpg -F threshold=0.5 http://localhost:8000/api/detect
```

```json
{
  "model": "SSDLite320 MobileNetV3 (COCO, torchvision)",
  "detections": [
    {"label": "person", "score": 0.988, "box": [28, 233, 136, 537]},
    {"label": "bus", "score": 0.973, "box": [11, 138, 473, 460]}
  ],
  "latency_ms": 90.6,
  "image": "data:image/png;base64,..."
}
```

`box` is `[x0, y0, x1, y1]` in the (possibly downscaled) image's pixel coordinates.

### `POST /api/report`
Body: JSON with the full `/api/inspect` response as `inspection`, plus optional `notes`.

```bash
curl -s -F file=@sample.png http://localhost:8000/api/inspect > insp.json
jq -n --slurpfile i insp.json '{inspection: $i[0], notes: "Line 3, after die change"}' \
  | curl -s -H 'Content-Type: application/json' -d @- http://localhost:8000/api/report
```

```json
{
  "disposition": "rework",
  "severity": "minor",
  "summary": "...",
  "observations": ["..."],
  "likely_root_causes": ["..."],
  "recommended_actions": ["..."],
  "model_agreement": "...",
  "_meta": {"model": "claude-opus-5-5", "input_tokens": 0, "output_tokens": 0}
}
```
(Field values above are placeholders showing the shape.)

### `GET /api/metrics`
Returns `artifacts/metrics.json`: `classes`, `test_size`, `models`, `per_class`, `confusion_matrix`,
`history`, `runtimes`, `params`.

### `GET /api/samples?n=8`
Returns `n` random held-out test images: `[{"label": "crack", "url": "/api/samples/crack/crack_00012.png"}]`.
File paths are resolved and checked to stay inside `data/test/` (blocks path traversal).

### Error codes

| Code | When |
|---|---|
| 400 | Empty upload, or `/api/report` without `inspection.input_preview` |
| 404 | `/api/metrics` before training; unknown sample file |
| 413 | Upload larger than 8 MB |
| 415 | File can't be decoded as an image |
| 422 | Request validation failed (FastAPI) |
| 503 | GenAI report unavailable (see §11) |

---

## 13. Web interface

Served at `/`. A single static HTML file with no framework or build step. It supports light and dark
mode and works on phone-width screens.

| Tab | URL | What it shows |
|---|---|---|
| Defect inspection | `/#inspect` | Upload or pick a held-out sample → input, Grad-CAM overlay with boxes, class probabilities, PyTorch vs INT8 latency, INT8 agreement. Predictions below 85% confidence get a "manual review" flag. Includes the GenAI report panel with an operator-notes box. |
| Object detection | `/#detect` | Upload any photo, adjust the threshold slider, see annotated boxes and a detections table |
| Model card | `/#card` | Baseline vs CNN, runtime benchmarks, per-class metrics, confusion matrix, training curve |

On load, the inspection tab automatically runs the first defective sample so the demo starts with a result.

---

## 14. Testing

```bash
pytest -q          # 9 tests, ~12 s
```

| Test | Checks |
|---|---|
| `test_health` | Service is up |
| `test_rejects_non_image` | Non-image upload → 415 |
| `test_inspect_fresh_samples[ok/scratch/spot/crack]` | 10 **newly generated** images per class (seed 1234, never seen in training): ≥ 8/10 correct; probabilities sum to 1 |
| `test_defect_is_localized` | A defect prediction comes with at least one region |
| `test_metrics` | Model card is well-formed |
| `test_report_requires_inspection` | Invalid report request → 400 |

Model-dependent tests skip automatically if `artifacts/defectnet.pt` is missing.

---

## 15. Deployment

### Docker
```bash
docker build -t visioninspect .
docker run -p 8000:8000 -e ANTHROPIC_API_KEY=sk-... visioninspect
```

The image:
- is based on `python:3.12-slim` with CPU-only PyTorch,
- bakes in a small sample gallery and the detector weights, so it runs without internet access (except
  for GenAI reports),
- runs as a non-root user,
- has a `HEALTHCHECK` against `/api/health`.

> The Dockerfile has not been build-tested yet (Docker wasn't available on the development machine).

### Cloud

The service is stateless and CPU-only, so it scales horizontally behind a load balancer.

| Cloud | Registry | Runtime | Secret store |
|---|---|---|---|
| AWS | ECR | ECS Fargate + ALB, or App Runner | Secrets Manager |
| Azure | ACR | Azure Container Apps | Key Vault |
| GCP | Artifact Registry | Cloud Run | Secret Manager |

GCP example:
```bash
gcloud builds submit --tag REGION-docker.pkg.dev/PROJECT/repo/visioninspect
gcloud run deploy visioninspect --image REGION-docker.pkg.dev/PROJECT/repo/visioninspect \
  --port 8000 --memory 2Gi --set-secrets ANTHROPIC_API_KEY=anthropic-key:latest
```

### Production hardening checklist
- Run multiple workers: `uvicorn app.main:app --workers 2` (each loads its own models, ~300 MB RAM each)
- Use the INT8 ONNX session as the main path and compute Grad-CAM only on request
- Add authentication (API key or OAuth) and rate limiting in front of `/api/report`
- Log predictions and confidence to monitor data drift

---

## 16. Configuration

| Variable | Default | Purpose |
|---|---|---|
| `ANTHROPIC_API_KEY` | — | Enables `/api/report` |
| `CLAUDE_MODEL` | `claude-opus-5-5` | LLM used for reports |
| `TORCH_HOME` | `~/.cache/torch` | Where detector weights are cached |

Constants in code:

| Constant | File | Value |
|---|---|---|
| `IMG_SIZE` | `ml/dataset.py` | 128 |
| `CLASSES` | `ml/dataset.py` | `ok, scratch, spot, crack` |
| `MAX_BYTES` | `app/main.py` | 8 MB |
| CAM threshold | `app/inference.py` | 0.5 |
| Manual-review threshold | `app/static/index.html` | 0.85 |

---

## 17. Using real datasets

The loader reads `data/<split>/<class>/*.png`. To train on real data:

1. Arrange images as `data/train/<class>/`, `data/val/<class>/`, `data/test/<class>/`.
2. Update `CLASSES` in `ml/dataset.py` to match the folder names.
3. Convert to PNG if needed. Any size works for inference; for training, resize to 128×128 first (or
   change `IMG_SIZE`).
4. Run `python -m ml.train`.

Good public datasets:

| Dataset | Content |
|---|---|
| **NEU-DET** | 1,800 steel-surface images, 6 defect classes, with boxes |
| **MVTec AD** | 15 object/texture categories for anomaly detection |
| **Severstal Steel Defect** (Kaggle) | Steel sheets with segmentation masks |
| **DAGM 2007** | Textured surfaces with weakly labelled defects |

For few real samples, swap DefectNet for an ImageNet-pretrained backbone (e.g. `torchvision`
ResNet-18 / MobileNetV3) and fine-tune.

---

## 18. Limitations and future work

| Limitation | Next step |
|---|---|
| Synthetic data only | Train and report on NEU-DET or MVTec AD |
| Grad-CAM boxes are coarse (no box supervision, 16×16 attention map) | Fine-tune a detector (Faster R-CNN / YOLO) or a segmentation model (U-Net) on labelled data |
| One defect type per image | Multi-label head (sigmoid outputs) |
| Fixed 0.85 manual-review threshold | Calibrate with temperature scaling and pick the threshold from a cost-weighted ROC |
| No unseen-defect detection | Add anomaly detection (PatchCore / autoencoder) for defect types not in training |
| No experiment tracking | MLflow or Weights & Biases for runs and the model registry |
| CPU only | CUDA / TensorRT export for line-speed throughput |

---

## 19. Troubleshooting

| Problem | Cause / fix |
|---|---|
| `AttributeError: module 'cv2' has no attribute 'HOGDescriptor'` | OpenCV 5 moved it; this project uses its own HOG, so pull the latest code |
| `/api/metrics` returns 404 | Run `python -m ml.train` |
| Sample gallery is empty | Run `python -m ml.dataset` |
| `/api/report` returns 503 | Set `ANTHROPIC_API_KEY` in `.env` or the environment and restart |
| First `/api/detect` call is slow | Detector weights download once (~14 MB) |
| `address already in use` | Another process has the port: `uvicorn app.main:app --port 8001` |
| INT8 model slower than FP32 | You're using dynamic quantization; use the static pipeline in `ml/train.py` |

---

## 20. Demo walkthrough

A 5-minute script for presenting the project:

1. **Problem (30 s).** Manual visual inspection is slow and inconsistent. This system classifies,
   localizes and reports defects automatically.
2. **Model card tab (1 min).** The classical baseline gets 45.8%, the CNN 97.3%. Point out that cracks are
   perfect and errors are faint scratches and spots.
3. **Optimization story (1 min).** Dynamic quantization made the model 7× *slower*. Static quantization
   was fast but lost 6 points. Percentile calibration fixed it: 2.8× faster, 3.7× smaller, −0.4 pt.
4. **Inspection tab (1 min).** Click a few samples, show the Grad-CAM overlay and the INT8 agreement
   badge. Find a low-confidence case and show the manual-review flag.
5. **GenAI report (1 min).** Add an operator note and generate a report. The LLM gives a disposition and
   cross-checks the CNN by looking at the image itself.
6. **Detection tab (30 s).** Upload any photo to show pretrained detection on real-world images.
7. **Engineering (30 s).** Open `/docs` for the typed API, mention the tests, Docker and cloud deployment.
