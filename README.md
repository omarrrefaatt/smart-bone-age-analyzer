<div align="center">

<br/>

```
██████╗  ██████╗ ███╗   ██╗███████╗     █████╗  ██████╗ ███████╗
██╔══██╗██╔═══██╗████╗  ██║██╔════╝    ██╔══██╗██╔════╝ ██╔════╝
██████╔╝██║   ██║██╔██╗ ██║█████╗      ███████║██║  ███╗█████╗
██╔══██╗██║   ██║██║╚██╗██║██╔══╝      ██╔══██║██║   ██║██╔══╝
██████╔╝╚██████╔╝██║ ╚████║███████╗    ██║  ██║╚██████╔╝███████╗
╚═════╝  ╚═════╝ ╚═╝  ╚═══╝╚══════╝    ╚═╝  ╚═╝ ╚═════╝ ╚══════╝
```

# 🦴 Smart Bone Age Analyzer

**AI-powered skeletal maturity assessment from hand radiographs**

[![Python](https://img.shields.io/badge/Python-3.9%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org)
[![Flask](https://img.shields.io/badge/Flask-3.x-000000?style=for-the-badge&logo=flask&logoColor=white)](https://flask.palletsprojects.com)
[![License](https://img.shields.io/badge/License-MIT-22c55e?style=for-the-badge)](LICENSE)
[![RSNA Dataset](https://img.shields.io/badge/Dataset-RSNA%20Bone%20Age-6366f1?style=for-the-badge)](https://www.rsna.org/education/ai-resources-and-training/ai-image-challenge/rsna-pediatric-bone-age-challenge-2017)

> _Upload a hand X-ray → get a precise skeletal age estimate in seconds._  
> Built with a custom CNN, a REST API backend, and a polished web interface.

---

</div>

## 📖 Table of Contents

- [What It Does](#-what-it-does)
- [Live Demo Preview](#-live-demo-preview)
- [Architecture Overview](#-architecture-overview)
- [The Model — Deep Dive](#-the-model--deep-dive)
- [Project Structure](#-project-structure)
- [Getting Started](#-getting-started)
- [API Reference](#-api-reference)
- [Training & Checkpoints](#-training--checkpoints)
- [Roadmap](#-roadmap)

---

## ✨ What It Does

Smart Bone Age Analyzer takes a **hand X-ray image** and predicts the patient's **skeletal age** using a deep learning model trained on thousands of pediatric radiographs from the RSNA Bone Age Challenge dataset.

| Input                              | Output                                    |
| ---------------------------------- | ----------------------------------------- |
| Hand X-ray (PNG / JPG)             | Predicted age in **months**               |
| Biological sex (`male` / `female`) | **Years + months** label                  |
| —                                  | **Skeletal stage** classification         |
| —                                  | **Percentile** estimate + confidence note |

### Skeletal Stages Detected

```
🍼 Infant          →  0 – 24 months
🚶 Toddler         →  25 – 60 months
🧒 Child           →  61 – 120 months
📚 Pre-adolescent  →  121 – 156 months
🧑 Adolescent      →  157 – 216 months
```

---

## 🖥️ Live Demo Preview

> Add your screenshots to `assets/screenshots/` and update the paths below.

```
assets/
└── screenshots/
    ├── ui-upload.png      ← Drag-and-drop upload screen
    └── ui-results.png     ← Prediction results card
```

---

## 🏗️ Architecture Overview

The system is composed of three layers that communicate cleanly via HTTP:

```
┌─────────────────────────────────────────────────────────────────┐
│                        FRONTEND  (Browser)                      │
│                                                                 │
│   UI.html ──────── script.js ──────── style.css                 │
│      │                 │                                        │
│   Upload X-ray     POST /predict                                │
│   Select sex       GET  /health                                 │
│   View results          │                                       │
└─────────────────────────┼───────────────────────────────────────┘
                          │  HTTP / multipart form-data
┌─────────────────────────▼───────────────────────────────────────┐
│                      BACKEND  (Flask :3000)                     │
│                                                                 │
│   app.py                                                        │
│   ├── /health         → {"status": "ok"}                        │
│   └── /predict        → Receives image + sex                    │
│        │                                                        │
│        ▼                                                        │
│   Image Preprocessing Pipeline                                  │
│   ├── Convert to Grayscale (L-mode)                             │
│   ├── Resize → 256×256                                          │
│   ├── Normalize (ImageNet μ/σ)                                  │
│   └── Tensor → CUDA / CPU                                       │
│        │                                                        │
│        ▼                                                        │
│   BoneAgeCNN Inference                                          │
│   └── Scaled output × 216.0 → Predicted months                 │
│        │                                                        │
│        ▼                                                        │
│   Post-processing                                               │
│   ├── Convert months → years + months label                     │
│   ├── Assign skeletal stage                                     │
│   ├── Estimate percentile                                       │
│   └── Return JSON response                                      │
└─────────────────────────────────────────────────────────────────┘
                          │
┌─────────────────────────▼───────────────────────────────────────┐
│                      MODEL LAYER                                │
│                                                                 │
│   best_bone_age_model.pth                                       │
│   ├── Backbone    : EfficientNet-B0 (pretrained ImageNet)       │
│   ├── Sex branch  : Linear(1 → 32) → ReLU                       │
│   ├── Fusion head : Concat → Linear → ReLU → Dropout → Linear  │
│   └── Output      : Sigmoid-activated scalar  ∈ [0, 1]         │
│                     × MAX_AGE (216) → months                    │
└─────────────────────────────────────────────────────────────────┘
```

### Data Flow Summary

```
Hand X-Ray Image
       │
       ▼
  Grayscale → 256×256 → Normalize
       │
       ▼
  EfficientNet-B0 backbone  ◄─── Sex flag (0 or 1)
       │                              │
       └──────────┬───────────────────┘
                  ▼
           Fusion FC Layers
                  │
                  ▼
          Sigmoid output × 216
                  │
                  ▼
        Predicted Age (months)
```

---

## 🧠 The Model — Deep Dive

### Base Architecture: EfficientNet-B0

The model uses **EfficientNet-B0** as its convolutional backbone, pretrained on ImageNet and fine-tuned on the RSNA Pediatric Bone Age dataset. EfficientNet was chosen for its compound scaling (depth × width × resolution), giving state-of-the-art accuracy at a fraction of the compute cost of larger models.

### Sex-Aware Fusion

Bone development is strongly sex-dependent. Rather than ignoring this signal, the model incorporates it explicitly:

- A **sex branch** (`Linear(1 → 32) → ReLU`) embeds the binary sex flag into a 32-dimensional representation.
- The CNN feature vector and sex embedding are **concatenated** before the final regression head, allowing the model to learn sex-conditional age predictions.

### Training Setup

| Parameter               | Value                            |
| ----------------------- | -------------------------------- |
| Input resolution        | 256 × 256                        |
| Batch size              | 32                               |
| Optimizer               | Adam                             |
| Learning rate           | 1e-3                             |
| LR scheduler            | ReduceLROnPlateau (patience = 5) |
| Early stopping          | patience = 10                    |
| Dropout                 | 0.3                              |
| Max age (normalization) | 216 months (18 years)            |
| Age bins                | 18                               |
| Random seed             | 42                               |
| GPU (training)          | NVIDIA Tesla T4                  |

### Dataset: RSNA Bone Age Challenge 2017

- **Source:** [RSNA Pediatric Bone Age Challenge](https://www.rsna.org/education/ai-resources-and-training/ai-image-challenge/rsna-pediatric-bone-age-challenge-2017)
- **Splits:** `RSNA_train` / `RSNA_val` / `RSNA_test`
- **Labels:** `boneage_train.csv`, `boneage_val.csv` (image id, bone_age in months, sex)
- **Preprocessing:** Grayscale conversion, normalization with ImageNet mean/std, WeightedRandomSampler to handle age distribution imbalance

### Output Normalization

All predictions are made in normalized space `[0, 1]` and then multiplied by `MAX_AGE = 216.0` to recover the bone age in months:

```python
predicted_months = model_output_sigmoid × 216.0
```

---

## 📁 Project Structure

```
smart-bone-age-analyzer/
│
├── 📄 README.md
├── 📄 requirements.txt
│
├── 🗂️  assets/
│   ├── sample-images/          ← Example hand X-rays for testing
│   ├── screenshots/            ← UI screenshots
│   └── training/               ← Training plots / loss curves
│
├── ⚙️  backend/
│   ├── app.py                  ← Flask inference server (port 3000)
│   ├── models/
│   │   ├── best_bone_age_model.pth   ← ✅ Best weights (use this)
│   │   ├── checkpoint_ep5.pth
│   │   ├── checkpoint_ep10.pth
│   │   ├── checkpoint_ep15.pth
│   │   ├── checkpoint_ep20.pth
│   │   ├── checkpoint_ep25.pth
│   │   ├── checkpoint_ep30.pth
│   │   └── checkpoint_ep35.pth
│   └── notebooks/
│       └── bone-age-assessment-from-hand-x-rays-baseline.ipynb
│
├── 🌐  frontend/
│   ├── UI.html                 ← Main interface
│   ├── script.js               ← Upload logic & API calls
│   └── style.css               ← Styling
│
├── 📂  docs/                   ← Extended documentation
└── 🐍  venv/                   ← Python virtual environment (local)
```

---

## 🚀 Getting Started

### Prerequisites

- Python **3.9+**
- pip
- A modern browser (Chrome / Firefox / Edge)
- _(Optional)_ CUDA-capable GPU for faster inference

---

### Step 1 — Clone the Repository

```bash
git clone https://github.com/your-username/smart-bone-age-analyzer.git
cd smart-bone-age-analyzer
```

---

### Step 2 — Set Up the Python Environment

**Option A — Virtual environment (recommended)**

```bash
python -m venv venv

# Activate on Linux / macOS
source venv/bin/activate

# Activate on Windows
venv\Scripts\activate
```

**Option B — System Python**

```bash
# Skip the venv steps and proceed directly to pip install
```

---

### Step 3 — Install Dependencies

```bash
pip install -r requirements.txt
```

Key packages installed:

| Package                 | Purpose                       |
| ----------------------- | ----------------------------- |
| `torch` + `torchvision` | Deep learning inference       |
| `flask`                 | REST API server               |
| `Pillow`                | Image loading & preprocessing |
| `numpy`                 | Numerical operations          |

---

### Step 4 — Verify the Model File

Make sure the best model weights are in place:

```bash
ls backend/models/
# You should see: best_bone_age_model (1).pth
```

> ⚠️ The model file name contains a space and parentheses. `app.py` already handles this — do not rename the file unless you update the path in `app.py` too.

---

### Step 5 — Start the Backend Server

```bash
python backend/app.py
```

You should see:

```
 * Running on http://0.0.0.0:3000
 * Debug mode: off
```

Verify the server is healthy:

```bash
curl http://localhost:3000/health
# → {"status": "ok"}
```

---

### Step 6 — Open the Frontend

Open `frontend/UI.html` in your browser:

```bash
# macOS
open frontend/UI.html

# Linux
xdg-open frontend/UI.html

# Windows
start frontend/UI.html
```

Or simply drag `UI.html` into your browser window.

---

### Step 7 — Run a Prediction

1. Click **Upload X-ray** and select a hand radiograph (PNG / JPG)
2. Select the patient's biological sex
3. _(Optional)_ Confirm the backend URL is `http://localhost:3000`
4. Click **Analyze** — results appear within seconds

---

## 📡 API Reference

### Health Check

```http
GET /health
```

**Response:**

```json
{ "status": "ok" }
```

---

### Predict Bone Age

```http
POST /predict
Content-Type: multipart/form-data
```

**Request fields:**

| Field   | Type   | Description                   |
| ------- | ------ | ----------------------------- |
| `image` | file   | Hand X-ray (PNG / JPG / JPEG) |
| `sex`   | string | `"male"` or `"female"`        |

**Example with `curl`:**

```bash
curl -X POST http://localhost:3000/predict \
  -F "image=@/path/to/hand_xray.png" \
  -F "sex=male"
```

**Success response (`200 OK`):**

```json
{
  "predicted_months": 134.2,
  "years": 11,
  "months_remainder": 2,
  "age_label": "11 years, 2 months",
  "stage": "Child",
  "sex": "male",
  "percentile": "50th",
  "confidence_note": "Prediction within normal model confidence range."
}
```

---

## 🗃️ Training & Checkpoints

Intermediate checkpoints are saved every 5 epochs during training. To resume training or evaluate a specific checkpoint:

```python
import torch
from your_model_module import BoneAgeCNN

# Load any checkpoint
ckpt = torch.load('backend/models/checkpoint_ep20.pth', map_location='cpu')
model = BoneAgeCNN().to(device)
model.load_state_dict(ckpt['model_state_dict'])
model.eval()

print(f"Best Val MAE: {ckpt['config']['best_val_mae']:.2f} months")
```

To retrain the model from scratch, open the notebook:

```bash
jupyter notebook backend/notebooks/bone-age-assessment-from-hand-x-rays-baseline.ipynb
```

---

## 🗺️ Roadmap

- [ ] Docker containerization for one-command deployment
- [ ] Batch inference support (multiple X-rays at once)
- [ ] DICOM file format support
- [ ] Grad-CAM visualization (highlight which bone regions drive predictions)
- [ ] Model retraining pipeline with custom datasets
- [ ] Cloud deployment guide (AWS / GCP / Azure)
- [ ] `.env`-based configuration for production

---

<div align="center">

**Built for medical AI research**

_If this project helped you, consider leaving a ⭐ on GitHub._

</div>
