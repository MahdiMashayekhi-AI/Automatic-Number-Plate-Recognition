# 🚗 Iranian Automated Number Plate Recognition (ANPR) System

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=flat&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-005571?style=flat&logo=fastapi)](https://fastapi.tiangolo.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=flat&logo=streamlit&logoColor=white)](https://streamlit.io/)
[![OpenCV](https://img.shields.io/badge/OpenCV-5C3EE8?style=flat&logo=opencv&logoColor=white)](https://opencv.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

An end-to-end, production-ready **Automated Number Plate Recognition (ANPR)** and tracking system tailored for Iranian license plates. Built with state-of-the-art computer vision models, custom deep learning architectures, robust decision-making algorithms, and modern web interfaces.

---

## 🌟 Key Features

- **🎯 Keypoint-Based Plate Deskewing**: Leverages YOLO-Pose keypoints to perform precise geometric perspective transformations, normalizing tilted or perspective-distorted license plates.
- **🔄 Multi-Object Vehicle Tracking**: Utilizes **ByteTrack** integration to track unique vehicle paths across video frames, maintaining persistent track IDs.
- **🔤 Custom CRNN + CTC OCR Engine**: A hybrid Deep Neural Network architecture (CNN + BiLSTM + CTC Loss) specifically trained to decode Persian license plate character sequences without explicit character segmentation.
- **🗳️ Temporal Voting Engine**: Incorporates a sequence-level voting algorithm across multiple consecutive video frames to mitigate OCR flickering, eliminate noise, and guarantee high-precision final text outputs.
- **⚡ FastAPI REST Services**: Asynchronous backend exposing endpoints for live frame inference, database queries, and plate search filters.
- **📊 Interactive Streamlit Dashboard**: A user-friendly web interface supporting real-time video/image processing and live database analytical view.
- **🗄️ Robust Database Integration**: Built with SQLAlchemy & SQLite for safe transaction management, track logging, and timestamp tracking.
- **📜 Structured Logging**: Comprehensive multi-handler logging infrastructure for debugging, monitoring, and audit trails.

---

## 🏗️ System Architecture

```
                               ┌──────────────────────────┐
                               │   Input Video / Image    │
                               └────────────┬─────────────┘
                                            │
                                            ▼
                               ┌──────────────────────────┐
                               │   ByteTrack & YOLO-Pose  │
                               │  (Detection & Tracking)  │
                               └────────────┬─────────────┘
                                            │
                                            ▼
                               ┌──────────────────────────┐
                               │ Perspective Correction   │
                               │   (Crop & Deskewing)     │
                               └────────────┬─────────────┘
                                            │
                                            ▼
                               ┌──────────────────────────┐
                               │   Custom CRNN + CTC      │
                               │    (OCR Character Rec)   │
                               └────────────┬─────────────┘
                                            │
                                            ▼
                               ┌──────────────────────────┐
                               │  Temporal Voting Engine  │
                               │  (Sequence Stabilization)│
                               └────────────┬─────────────┘
                                            │
                      ┌─────────────────────┴─────────────────────┐
                      ▼                                           ▼
          ┌───────────────────────┐                   ┌───────────────────────┐
          │  FastAPI / Streamlit  │                   │  SQLAlchemy Database  │
          │    User Interfaces    │                   │   (ANPR Record Store) │
          └───────────────────────┘                   └───────────────────────┘
```

---

## 📂 Project Structure

```text
.
├── dashboard/
│   └── app.py              # Streamlit interactive web application
├── data/                   # Sample videos and dataset storage
├── outputs/                # Trained model checkpoints (best_model.pt)
├── src/
│   ├── api/
│   │   └── main.py         # FastAPI application & API endpoints
│   ├── database/
│   ├── connection.py   # SQLAlchemy session & context manager
│   │   └── models.py       # ORM Database Schema (DetectedPlate)
│   ├── decision/
│   │   └── voter.py        # Temporal voting algorithm for text stabilization
│   ├── detection/
│   │   ├── detector.py     # YOLO pose detector integration
│   │   └── geometry.py     # Perspective transformation & deskewing
│   ├── ocr/
│   │   ├── dataset.py      # PyTorch Dataset loader
│   │   ├── model.py        # CRNN (CNN + BiLSTM) architecture
│   │   ├── predictor.py    # PlateReader inference wrapper
│   │   ├── train.py        # OCR model training pipeline
│   │   └── utils.py        # CTC decode & Levenshtein CER metric
│   ├── tracking/
│   │   └── tracker.py      # ByteTrack vehicle & plate tracker
│   ├── config.py           # Global hyper-parameters & configuration
│   ├── logger.py           # Centralized logging setup
│   └── pipeline.py         # End-to-End ANPR Pipeline
├── main.py                 # Pipeline execution entry point for videos
├── requirements.txt        # Project dependencies
└── README.md               # Project documentation
```

---

## 🛠️ Tech Stack

- **Core & DL**: Python 3.9+, PyTorch, Torchvision, NumPy
- **Computer Vision**: OpenCV, Ultralytics YOLO (Pose & ByteTrack)
- **OCR Engine**: Custom CRNN (CNN + BiLSTM + PyTorch CTC Loss)
- **Backend & Database**: FastAPI, SQLAlchemy, SQLite, Uvicorn
- **Dashboard**: Streamlit, Pandas
- **Utilities**: Levenshtein Distance, Python Logging

---

## 🚀 Getting Started

### 1. Prerequisites & Installation

Clone the repository and create a virtual environment:

```bash
git clone https://github.com/your-username/iranian-anpr-system.git
cd iranian-anpr-system

python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

Install the dependencies:

```bash
pip install -r requirements.txt
```

### 2. Model Weights Setup

Ensure you place the required model checkpoints in the correct directories:
- `license_plate_keypoint.pt` -> Root directory (YOLO Pose model)
- `outputs/best_model.pt` -> `outputs/` directory (CRNN OCR checkpoint)

---

## 💻 Usage

### 🎥 1. Running Video Pipeline
To process a video file through the end-to-end ANPR pipeline and visualize results in real time:

```bash
python main.py
```

### ⚡ 2. Running FastAPI REST Services
Launch the FastAPI server:

```bash
uvicorn src.api.main:app --reload
```
Open your browser at `http://127.0.0.1:8000/docs` to explore interactive Swagger API documentation.

#### API Endpoints Overview:
| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `GET` | `/` | Service health check |
| `GET` | `/plates` | Fetch recent $N$ detected plates |
| `GET` | `/plates/search` | Search plates by substring or pattern |
| `POST` | `/predict` | Upload an image for instant ANPR inference |

### 📊 3. Running Interactive Streamlit Dashboard
Launch the web dashboard for video upload, real-time inference, and database logs:

```bash
streamlit run dashboard/app.py
```

---

## 🛣️ Roadmap

- [x] YOLO Keypoint Detection & Geometry Correction
- [x] Custom CRNN Model & CTC Training for Persian Characters
- [x] ByteTrack Multi-Object Tracking & Temporal Voting Engine
- [x] FastAPI REST Integration & Thread-Safe SQLite Storage
- [x] Streamlit Interactive Web Dashboard
- [ ] Comprehensive Unit & Integration Testing (`pytest`)
- [ ] Benchmarking Suite (FPS, GPU Memory, CER Metrics)
- [ ] Containerization via Docker & Docker-Compose
- [ ] Continuous Integration / CI-CD with GitHub Actions
- [ ] Inference Acceleration (ONNX Runtime / TensorRT)

---

## 📝 License

Distributed under the MIT License. See `LICENSE` for more information.