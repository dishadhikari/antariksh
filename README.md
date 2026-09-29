# VYOMA- Vision Based AI for On-Board Mission Operations

**VYOMA** is an offline, edge-AI based onboard experiment monitoring and verification system designed to assist astronauts during scientific experiments by continuously understanding human–object interactions, validating experiment procedures, detecting deviations and providing real-time guidance.
It converts live camera feeds into structured experiment states by combining **computer vision, pose estimation, temporal action recognition and deterministic experiment validation** into one edge-AI pipeline.
The complete inference and decision pipeline is designed to operate **offline on edge hardware**, reducing dependence on continuous communication with ground control.
Built for the **AI Human Activity Recognition for On-board BAS Experiments** SIH 2026 problem statement.

---

## Table of Contents

4. Key Features
6. System Architecture
7. Tech Stack
8. Methodology
9. System Modules
12. Dataset Generation and Experiment Workflow
13. Training Strategy
14. Installation & Setup
16. Usage Guide
17. Project Structure
18. Output & Logging
19. Testing & Evaluation
20. Hardware Deployment
23. Team
24. License

---

# Overview

 Instead of only recognizing human activities, VYOMA understands:


# Key Features

### 1. Temporal Activity & Interaction Recognition
Understands actions from sequences of frames by combining astronaut pose, hand landmarks, object tracking and hand–object interaction.

```text
Approach → Contact → Grasp → Move → Release
                         ↓
                  REMOVE_RED_BOX
# System Architecture

```text
                    FIXED CAMERA
                         │
                         ▼
              ┌────────────────────┐
              │ GStreamer +         │
              │ NVIDIA DeepStream   │
              └─────────┬──────────┘
                        │
                        ▼
              ┌────────────────────┐
              │ PERCEPTION LAYER   │
              │                    │
              │ YOLO26 Detection   │
              │ YOLO26 Pose        │
              │ Hand Landmarks     │
              │ 3D HMR             │
              └─────────┬──────────┘
                        │
                        ▼
              ┌────────────────────┐
              │ TRACKING &         │
              │ SPATIAL ANALYSIS   │
              │                    │
              │ Object Tracking    │
              │ Hand/Object Links  │
              │ Rack Coordinates   │
              │ Motion Features    │
              └─────────┬──────────┘
                        │
                        ▼
              ┌────────────────────┐
              │ TEMPORAL REASONING │
              │                    │
              │ TCN   
              └─────────┬──────────┘
                        │
                        ▼
              ┌────────────────────┐
              │ OUTCOME            │
              │ VERIFICATION       │
              └─────────┬──────────┘
                        │
                        ▼
              ┌────────────────────┐
              │ EXPERIMENT         │
              │ STATE VALIDATION   │
              │                    │
              │ FSM + Confidence   │
              │ Logic              │
              └─────────┬──────────┘
                        │
               ┌────────┼────────┐
               ▼        ▼        ▼
            CORRECT  DEVIATION  UNCERTAIN
               │        │        │
               ▼        ▼        ▼
           Next Step  Voice     Observe
           Guidance   Alert      More
               │        │        │
               └────────┼────────┘
                        ▼
              ┌────────────────────┐
              │ OUTPUT & TELEMETRY │
              │                    │
              │ GUI                │
              │ Voice              │
              │ JSONL / SQLite     │
              │ Local Video        │
              │ RTSP Stream        │
              │ Event Telemetry    │
              └────────────────────┘
```

---

# Tech Stack

| Layer | Technology |
|---|---|
| Primary Programming Language | Python |
| AI Training | PyTorch + Ultralytics |
| Object Detection | YOLO26 |
| Pose Estimation | YOLO26-Pose |
| Segmentation | YOLO26-Seg  |
| 3D Human Understanding | 3D Human Mesh Recovery |
| Tracking | NVIDIA DeepStream Tracker  |
| Temporal Model | TCN / lightweight Transformer |
| Spatial Reasoning | Rack-centric coordinate system |
| Video Pipeline | GStreamer + NVIDIA DeepStream |
| Inference Optimization | ONNX + TensorRT FP16 |
| Edge Hardware | NVIDIA Jetson Orin NX / AGX Orin |
| GUI | PySide6 Qt |
| Voice | Piper TTS |
| Database | SQLite |
| Structured Logs | JSONL |
| Video Encoding | H.264/H.265 |
| Network Streaming | RTSP |
| Operating System | Ubuntu / JetPack |

---

# Methodology

The implementation follows the pipeline:

```text
1. Capture
      ↓
2. Detect
      ↓
3. Track
      ↓
4. Estimate Pose / Hands / 3D Body
      ↓
5. Extract Spatial + Temporal Features
      ↓
6. Recognize Activity
      ↓
7. Verify Physical Outcome
      ↓
8. Validate Experiment Sequence
      ↓
9. Guide / Alert
      ↓
10. Log + Store + Transmit Events
```

---

# System Modules

| Module | Description |
|---|---|
| Camera Manager | Captures and manages local video streams |
| Video Pipeline | Performs GPU-accelerated frame processing |
| Object Detector | Detects experiment-specific objects |
| Pose Estimator | Extracts astronaut body keypoints |
| 3D HMR Module | Estimates 3D human configuration |
| Hand Interaction Module | Detects hand-object interaction |
| Object Tracker | Maintains object identity across frames |
| Temporal Action Model | Recognizes action sequences |
| Outcome Verifier | Confirms physical completion of actions |
| Experiment Engine | Maintains the current experiment state |
| Confidence Engine | Handles uncertain predictions |
| Deviation Detector | Detects skipped/wrong/repeated actions |
| Guidance Engine | Generates next-step instructions |
| Voice Engine | Provides offline voice alerts |
| Telemetry Engine | Generates lightweight mission events |
| Communication Manager | Handles transmission and synchronization |
| Logger | Stores timestamped experiment records |
| GUI | Displays experiment, AI and system status |

---

# 

## Communication Loss

```text
LINK AVAILABLE
      ↓
Transmit Events

LINK LOST
      ↓
Store Events Locally
      ↓
Continue Experiment

LINK RESTORED
      ↓
Synchronize Stored Events
```
# Dataset Generation and Experiment Workflow

We have trained the AI model based on our own recorded videos (synthetic dataset generation). VYOMA uses a combination of **real and synthetic data**. Synthetic data is primarily used to test for the specific experiment, while real data is used for realistic validation.

Record experiment videos using:

- Different body orientations
- Different execution speeds
- Different object positions
- Different lighting conditions
- Correct actions
- Incorrect actions
- Skipped actions

# Installation & Setup

## 1. Clone the Repository
```bash
git clone https://github.com/yourusername/VYOMA.git
cd VYOMA
```

## 2. Create Virtual Environment
```bash
python -m venv venv
```

### Windows
```bash
venv\Scripts\activate
```

### Linux / macOS
```bash
source venv/bin/activate
```

## 3. Install Python Dependencies
```bash
pip install -r requirements.txt
```

## 4. Install NVIDIA / Edge Dependencies
For Jetson deployment, install the required:
- JetPack
- CUDA
- TensorRT
- NVIDIA DeepStream
- GStreamer

according to the target hardware environment.

# Usage Guide

## Step 1 — Start Camera

```bash
python app.py --camera 0
```

---
## Step 2 — Start Experiment

Select the required experiment from the GUI.

The system initializes:

- Camera
- Object detector
- Pose model
- Hand tracking
- 3D HMR
- Experiment state
- Logger
- Video recorder

---

## Step 3 — Perform Experiment

The system continuously:

- Detects objects
- Tracks the astronaut
- Recognizes actions
- Verifies physical outcomes
- Validates experiment sequence

---

## Step 4 — Receive Guidance

The GUI displays:

```text
CURRENT STEP
NEXT STEP
CONFIDENCE
SYSTEM STATUS
EXPERIMENT STATUS
```

Voice alerts are generated for confirmed deviations.

---

## Step 5 — Complete Experiment

At completion, VYOMA generates:

```text
Experiment Video
Experiment Event Log
Experiment Summary
Telemetry Record
```

---

# Output & Logging

## Experiment Event Log

VYOMA stores lightweight event records using JSONL.

Example:

```json
{
  "timestamp": "2026-09-29T12:34:56.421",
  "experiment": "BOX_SEPARATION",
  "step": "STEP_02",
  "action": "REMOVE_RED_BOX",
  "confidence": 0.94,
  "status": "COMPLETED"
}
```

---

## Deviation Event

```json
{
  "timestamp": "2026-09-29T12:36:04.217",
  "experiment": "BOX_SEPARATION",
  "expected_step": "STEP_02",
  "observed_action": "REMOVE_YELLOW_BOX",
  "confidence": 0.91,
  "status": "OUT_OF_SEQUENCE"
}
```

# Testing & Evaluation

## Activity Recognition Metrics
- Accuracy
- Precision
- Recall
- F1-score
- Confusion matrix

## Experiment Validation Metrics
- Correct step verification rate
- False step completion rate
- Wrong-sequence detection rate
- Skipped-step detection rate
- Repeated-step detection rate
- Unknown-action rejection rate
- Recovery success rate

## Edge Metrics

- FPS
- End-to-end latency
- GPU utilization
- CPU utilization
- Memory usage
- Power consumption

# System Modes

## Normal Mode

```text
Detect → Recognize → Verify → Advance
```

## Uncertain Mode

```text
Ambiguous Evidence
      ↓
Continue Observation
      ↓
Additional Evidence
      ↓
Verify / Remain Uncertain
```

## Deviation Mode

```text
Wrong Action
      ↓
Deviation Confirmed
      ↓
Voice Alert
      ↓
State Does Not Advance
      ↓
Astronaut Corrects Action
```

## Communication Loss Mode

```text
Communication Lost
      ↓
Continue Full Onboard Operation
      ↓
Store Events Locally
      ↓
Communication Restored
      ↓
Synchronize Events
```

# Future Improvements
- Multi-camera perception
- Improved 3D human mesh recovery
- More robust orientation-agnostic activity recognition
- Hardware-aware adaptive inference
- Advanced uncertainty estimation
- Additional experiment templates
- Radiation-tolerant deployment hardware
- Integration with spacecraft payload interfaces

# Team
| Name | Role |
|---|---|
| Disha Adhikari | AI  |
| Akshat Porwal | Backend / Edge AI |
| Priya Singh | Experiment Pipeline Validation |
| Mishita Joshi | AI Dataset Training |
| Mansi Rai | GUI |
| Geeriwar Aggarwal | UI |

# License
This project is developed as part of the **Smart India Hackathon 2026** problem statement ID 26174.
