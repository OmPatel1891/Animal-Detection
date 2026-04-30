# Real-Time Animal Detection with Arduino Buzzer Alert

> **Edge-deployed computer vision system that detects livestock (cow, ox, pig) from a live camera feed and triggers a hardware buzzer alert via Arduino — bridging deep learning and IoT.**

---

## Overview

Unauthorized animal entry into restricted zones (farms, roads, crops) causes significant economic and safety hazards. This project builds a **real-time detection and alerting system** that runs a TensorFlow classification model on a live webcam stream, identifies target animals with high confidence, and immediately fires a physical alarm through an Arduino-connected buzzer — all from a PySide6 desktop GUI.

---

## Key Features

- Real-time video feed processed at ~30 fps via `QTimer`
- Confidence-thresholded classification (≥ 95%) to suppress false positives
- Automatic image capture with predicted label overlaid on detection
- Hardware alarm via serial communication to Arduino (buzzer trigger)
- Clean PySide6 GUI with live video display and manual override buttons

---

## Target Classes

| Class | Description |
|---|---|
| `cow` | Domestic cattle |
| `ox` | Draft cattle |
| `pig` | Swine |

---

## Tech Stack

| Category | Tools |
|---|---|
| Language | Python 3 |
| ML Framework | TensorFlow / Keras (`.h5` model) |
| GUI | PySide6 (Qt6) |
| Computer Vision | OpenCV |
| Hardware | Arduino via PySerial |
| Libraries | NumPy, datetime, os |

---

## Project Structure

```
Animal-Detection/
├── animal_detect.py    # Main GUI app: live detection + Arduino alert
├── model_train.py      # Model training script
├── aug.py              # Data augmentation pipeline
└── Report.pdf          # Full technical report
```

---

## System Architecture

```
Live Camera (OpenCV)
       │
       ▼
  Frame Preprocessing (resize 224×224, normalize)
       │
       ▼
  TensorFlow Model (.h5)
       │
   ┌───┴────────────────────┐
   │ Confidence ≥ 0.95?      │
   │  YES → Save Image       │
   │      → Serial → Arduino │
   │      → Buzzer Alert     │
   │  NO  → "No Animal"      │
   └─────────────────────────┘
```

---

## How to Run

### Prerequisites

```bash
pip install tensorflow opencv-python PySide6 pyserial numpy
```

### Setup

1. Connect Arduino to `COM6` (update port in `animal_detect.py` if different)
2. Place your trained model at the path specified in `animal_detect.py` (update hardcoded path)
3. Run:

```bash
python animal_detect.py
```

---

## Model Training

Training pipeline lives in `model_train.py`. Data augmentation (flip, rotation, zoom, brightness) is handled in `aug.py`. The model is a CNN trained on labeled livestock images and saved as `animal_detection.h5`.

---

## Real-World Use Case

Deployed at farm perimeters or crop boundaries, this system provides 24/7 automated surveillance and immediate physical alerts — reducing the need for human monitoring and preventing crop damage from livestock intrusions.

---

## Author

**Om Patel** | MS Data Science, University of Michigan Ann Arbor  
[LinkedIn](https://www.linkedin.com/in/om-patel-20507a219/) · [GitHub](https://github.com/OmPatel1891)
