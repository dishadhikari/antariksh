# VYOMA- Vision Based AI for On-Board Mission Operations

**VYOMA** is an **offline, edge-AI based onboard experiment monitoring and verification system** designed to assist astronauts during scientific experiments by combining **computer vision, pose estimation, temporal action recognition and deterministic experiment validation** into one edge-AI pipeline, thus providing real-time guidance.<br><br>
The complete inference and decision pipeline is designed to operate **offline on edge hardware**, reducing dependence on continuous communication with ground control.<br><br>
Built for the **AI Human Activity Recognition for On-board BAS Experiments** SIH 2026 problem statement.

---

## Table of Contents

1. Key Features
2. System Architecture
3. Tech Stack
4. Methodology
5. System Modules
6. Dataset Generation and Experiment Workflow
7. Training Strategy
8. Installation & Setup
9. Usage Guide
10. Project Structure
11. Output & Logging
12. Testing & Evaluation
13. Hardware Deployment
14. Team
15. License
    
#Key Features

## Key Features

### 1. Edge-Native Offline AI Experiment Evaluation
ASTRA performs the complete perception, activity recognition, state validation, and decision pipeline **on-device**, without requiring cloud or continuous ground connectivity. This enables low-latency, autonomous experiment monitoring even during communication outages.

### 2. Runtime AI Experiment Evaluator
Continuously evaluates the experiment **while it is being performed**, combining temporal activity recognition, object state, hand–object interaction, physical outcome verification, and a deterministic experiment state machine to detect:

```text
CORRECT → SKIPPED → REPEATED → OUT-OF-SEQUENCE → UNCERTAIN

```text
STEP_COMPLETED
DEVIATION_DETECTED
STEP_RECOVERED
EXPERIMENT_COMPLETED
```

Events are stored locally during communication loss and synchronized when connectivity is restored, reducing bandwidth requirements and preserving mission records.

### 8. Local Mission Recording & Monitoring
Provides complete onboard monitoring through structured event logs, local H.264/H.265 video recording, RTSP streaming, GUI visualization, and offline voice alerts. The core experiment-monitoring pipeline operates independently of continuous ground connectivity.
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

# Dataset Generation and Experiment Workflow

We have trained the AI model based on our own recorded videos (synthetic dataset generation). VYOMA uses a combination of **real and synthetic data**. Synthetic data is primarily used to test for the specific experiment, while real data is used for realistic validation.

Recorded experiment videos using:

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
# Output & Logging
Experiment Event Log stores lightweight event records using JSONL.

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
- Accuracy- 0.91
- Precision- 0.7
- Recall-0.88
- F1-score- 0.77
- Confusion matrix

## Experiment Validation Metrics
- Correct step verification rate
- False step completion rate
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
| Disha Adhikari | AI and Computer Vision  |
| Akshat Porwal | Backend and Edge AI |
| Priya Singh | Experiment Pipeline Validation |
| Mishita Joshi | AI Dataset and Model Training |
| Mansi Rai | GUI and Frontend Development |
| Geeriwar Aggarwal | UI/UX & System Visualization |

# License
This project is developed as part of the **Smart India Hackathon 2026** problem statement ID 26174.
