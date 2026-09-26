<div align="center">

# 🚗 ANPR — Iranian Automatic Number Plate Recognition

**An end-to-end computer vision system for detecting, tracking, and reading Iranian license plates from images, video, and live streams.**

</div>

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Ultralytics YOLO](https://img.shields.io/badge/Ultralytics-YOLOv8--Pose-00FFFF)](https://github.com/ultralytics/ultralytics)
[![FastAPI](https://img.shields.io/badge/FastAPI-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](./LICENSE)
[![Status](https://img.shields.io/badge/Status-v1.0%20Initial%20Release-orange)]()

<img width="1918" height="867" alt="Image" src="https://github.com/user-attachments/assets/cd6656a8-cb6a-4f53-87cb-c69f08c74dce" />

<img width="1918" height="862" alt="Image" src="https://github.com/user-attachments/assets/aee95d87-26de-4908-a1df-c13e3d74c851" />

<video src="https://github-production-user-asset-6210df.s3.amazonaws.com/108976550/659418603-04551779-b793-46fe-973b-deb6457ef883.mp4?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIAVCODYLSA53PQK4ZA%2F20260926%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20260926T114119Z&X-Amz-Expires=300&X-Amz-Signature=c2bdca4a6a5aa4087f4759da3b9ca1a2efb10901b7558abddc5b33ab030a7c4c&X-Amz-SignedHeaders=host&response-content-type=video%2Fmp4" controls width="800"></video>

## Demo for video on Youtube

[![ANPR Demo](https://img.youtube.com/vi/l7ReHWg6evw/maxresdefault.jpg)](https://www.youtube.com/watch?v=l7ReHWg6evw)

---

## 📖 Overview

This project is a complete, self-contained **Automatic Number Plate Recognition (ANPR)** pipeline built specifically for **Iranian license plates**. It goes beyond a single detection model — it's a full system: plate localization with keypoint-based perspective correction, OCR, temporal voting for reliability, a REST API, and an interactive dashboard.

It was built from the ground up as a resume/portfolio project, with a deliberate focus on **engineering discipline over quick wins**: every major design decision — from why detections are validated geometrically before OCR, to why the temporal voter requires repeated agreement before committing a plate to the database — is intentional, tested, and documented below.

> **This is v1.0 — an initial, working release.** The [Roadmap](#-roadmap--whats-next) section at the bottom is not an afterthought; it's an honest account of what's next, because a resume project that pretends to be "finished" is less convincing than one that shows a clear trajectory.

---

## ✨ Key Features

- 🎯 **Keypoint-based plate detection** — A fine-tuned YOLOv8-Pose model detects each plate's 4 corners directly, enabling accurate perspective correction instead of a naive axis-aligned crop.
- 📐 **Geometric sanity validation** — Detected keypoints are checked for plausible area and aspect ratio before ever reaching OCR, rejecting degenerate/garbage detections early.
- 🔄 **Multi-object tracking** — ByteTrack-based tracking keeps a stable ID per vehicle across frames, even through brief occlusion.
- 🗳️ **Temporal voting for reliability** — A single OCR reading is never trusted blindly. A plate is only confirmed once the *same* valid reading has been observed multiple times across frames — a deliberate trade-off of latency for correctness.
- ✅ **Format-aware validation** — OCR output is validated against the real structure of Iranian plates (2 digits – 1 letter – 3 digits – 2-digit region code) before being accepted.
- 🖼️ **Sharpness-aware frame selection** — For each tracked vehicle, the pipeline actively waits for the sharpest available frame instead of reading the first (possibly blurry) one.
- 🌐 **Full REST API** (FastAPI) — Image and video endpoints, plate search, and statistics.
- 📊 **Interactive dashboard** (Streamlit) — Live inference view with a scrollable recognition history, plus a searchable database of everything ever detected.
- 🗄️ **Persistent storage** — Every confirmed plate is saved with its confidence score and a cropped, perspective-corrected image for auditability.

---

## 🏗️ Architecture

```mermaid
flowchart LR
    A[Image / Video / Webcam] --> B["Plate Detection<br/>(YOLOv8-Pose, 4 keypoints)"]
    B --> C["Geometric Validation<br/>(area + aspect ratio)"]
    C -->|invalid| X["Discarded"]
    C -->|valid| D["Perspective Correction<br/>(Deskew)"]
    D --> E["Sharpness Check"]
    E -->|too blurry| W["Wait for next frame"]
    E -->|sharp enough| F["OCR<br/>(CRNN + CTC)"]
    F --> G["Format Validation<br/>(Iranian plate pattern)"]
    G --> H["Temporal Voting<br/>(video only)"]
    H --> I[("SQLite Database")]
    H --> J["API / Dashboard"]
```

The pipeline is split into two distinct entry points that share the same core components but behave differently by design:

| Mode | Used by | Behavior |
|---|---|---|
| **Stateless** (`process_image`) | Single image upload (API `/detect`, Dashboard image tab) | One shot, no tracking, no history — confidence is based purely on format validity |
| **Stateful** (`process_frame`) | Video / webcam (API `/detect-from-video`, Dashboard video tab, `run.py`) | Tracking + temporal voting across frames — confidence is based on repeated agreement |

This separation matters: an earlier version of this project mixed the two, which caused tracking state from one video to silently leak into the next. Keeping them separate was a deliberate architectural fix, not an accident.

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| Plate & keypoint detection | Ultralytics YOLOv8-Pose (fine-tuned) |
| OCR | Custom CRNN (CNN + BiLSTM) with CTC loss, trained from scratch |
| Tracking | ByteTrack (custom-tuned config) |
| API | FastAPI |
| Dashboard | Streamlit |
| Database | SQLite + SQLAlchemy |
| Core CV | OpenCV |
| Training / experimentation | PyTorch, TensorBoard |

---

## 📁 Project Structure

```
ANPR/
├── data/
│   ├── ocr/                 # OCR training data (labels only committed — see .gitignore)
│   ├── pose/                # Keypoint detection training data
│   └── samples/             # Sample videos for testing
├── outputs/
│   ├── checkpoints/
│   │   ├── ocr/             # Trained OCR model weights
│   │   └── pose/            # Fine-tuned pose model weights
│   └── plates/              # Saved crops of confirmed plates
├── scripts/                 # Dev tooling (dataset prep, frame extraction, etc.)
├── src/
│   ├── api/                 # FastAPI application
│   ├── dashboard/           # Streamlit application
│   ├── database/            # SQLAlchemy models & connection
│   ├── decision/            # Format validation + temporal voting
│   ├── detection/           # Detector, geometry, image quality checks
│   ├── display/             # Display/formatting helpers
│   ├── ocr/                 # CRNN model, training, evaluation
│   ├── pose/                # Pose model fine-tuning
│   ├── tracking/            # Multi-object tracker
│   ├── config.py            # All tunable constants — single source of truth
│   └── pipeline.py          # Orchestrates the full flow
├── run.py                   # Standalone video demo (OpenCV window)
├── requirements.txt
└── README.md
```

---

## 🚀 Getting Started

### Prerequisites
- Python 3.10+
- (Recommended) a CUDA-capable GPU — the pipeline runs on CPU too, at reduced speed

### Installation

```bash
git clone https://github.com/MahdiMashayekhi-AI/Automatic-Number-Plate-Recognition.git
cd Automatic-Number-Plate-Recognition

python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate

pip install -r requirements.txt
```

### Running the components

**Standalone video demo (OpenCV window):**
```bash
python run.py
```

**API server:**
```bash
uvicorn src.api.main:app --reload
```
Interactive API docs available at `http://localhost:8000/docs` once running.

**Dashboard:**
```bash
streamlit run src/dashboard/app.py
```

---

## 🌐 API Reference

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/api/v1/status` | Health check |
| `POST` | `/api/v1/plates/detect` | Detect & read plates in a single uploaded image |
| `POST` | `/api/v1/plates/detect-from-video` | Process an uploaded video, returns confirmed plates |
| `GET` | `/api/v1/plates` | List recently detected plates |
| `GET` | `/api/v1/plates/search?q=` | Search plates by text |
| `GET` | `/api/v1/plates/{id}` | Get a single plate record |
| `GET` | `/api/v1/plates/stats` | Aggregate statistics (totals, unique plates, avg. confidence) |

---

## 📊 Performance

| Component | Metric | Value | Notes |
|---|---|---|---|
| OCR (CRNN) | Sequence Accuracy | `79.30%` | Measured on held-out test set (majority synthetic data — see Limitations) |
| OCR (CRNN) | Character Error Rate (CER) | `5.83%` | |
| Plate Detection (Pose) | Precision | `96.1%` | On held-out validation set after fine-tuning on Iranian plate data |
| Plate Detection (Pose) | Recall | `93.3%` | |
| Plate Detection (Pose) | mAP@50 | `95.4%` | |
| End-to-end, challenging video (informal) | Detection reliability | `~97%` | After fixing a tracker/frame-skip interaction bug — see [Roadmap](#-roadmap--whats-next) for formal benchmark status |

---

## ⚠️ Known Limitations

Being upfront about this is deliberate — a project that names its limitations is more trustworthy than one that doesn't have any (they always do).

- **Training data is partially synthetic.** A meaningful portion of the OCR training set was synthetically generated, not captured from real plates. Real-world accuracy is likely somewhat lower than the benchmark above until this is fully replaced.
- **No formal end-to-end benchmark yet.** OCR and detection are each evaluated separately; a rigorous, held-out, real-world end-to-end accuracy number (image in → correct plate text out) is still on the roadmap.
- **Small/distant plates remain the hardest case.** Plates that occupy very few pixels in the frame are the primary remaining failure mode, an inherent challenge for any single-shot detector at fixed inference resolution.
- **Single-video-at-a-time API.** The video endpoint is designed for one request being processed at a time; concurrent video uploads can share internal tracker state. A task queue is the correct long-term fix.
- **No two-stage (vehicle-then-plate) detection.** The system detects plates directly rather than first localizing the vehicle, which is the more failure-resistant approach used in some production ANPR systems.
- **Live dashboard playback speed depends on hardware.** Real-time-looking playback in the Streamlit dashboard is throttled adaptively but is not a hard real-time guarantee — the video processing API endpoint is unaffected by this since it doesn't need to display anything live.

---

## 🗺️ Roadmap — What's Next

This is v1. Here's what's planned, roughly in priority order:

- [ ] Replace remaining synthetic OCR data with real, human-verified samples
- [ ] Formal end-to-end evaluation pipeline with a held-out, real-world test set
- [ ] Tiled/sliced inference (SAHI) for reliably detecting very small or distant plates
- [ ] Two-stage detection (vehicle localization → plate search within vehicle) to reduce false positives
- [ ] Constrained CTC decoding (using the known Iranian plate character pattern) to improve OCR accuracy without changing the model architecture
- [ ] Task-queue-based API to properly support concurrent video processing
- [ ] Live webcam streaming support in the dashboard

---

## 📦 Dataset & Model Credits

- The fine-tuned pose model builds on a subset of the [`ANPR Dataset Pose`](https://universe.roboflow.com/new-workspace-tpi2p/anpr-mlcmk) dataset (Roboflow, CC BY 4.0), used for general plate-shape/geometry training alongside a self-collected and hand-annotated set of Iranian license plates.
- All Iranian plate imagery used for training was collected and annotated specifically for this project.

---

## 📄 License

This project is licensed under the MIT License — see the [LICENSE](./LICENSE) file for details.

---

## 👤 Author

**[Mahdi Mashayekhi]**
[LinkedIn](https://www.linkedin.com/in/mahdimashayekhi/) · [GitHub](https://github.com/MahdiMashayekhi-AI) · [Email](mailto:mahdimashayekhi.ai@gmail.com)

---

<div align="center">

*If you found this project interesting or have feedback, feel free to open an issue or reach out — I'm always happy to discuss the engineering decisions behind it.*

</div>
