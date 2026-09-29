# ASTRA — Autonomous Experiment Verification & Guidance System

**ASTRA (Autonomous Step Tracking, Reasoning & Assistance)** is an offline, edge-AI based system designed to assist astronauts during scientific experiments by continuously understanding human–object interactions, validating experiment procedures, detecting deviations, and providing real-time guidance.

Built for the **AI Human Activity Recognition for On-board BAS Experiments** problem statement.

---

## Table of Contents

1. Overview
2. Problem Statement
3. Proposed Solution
4. Key Features
6. System Architecture
7. Tech Stack
8. Methodology
9. System Modules
10. Experiment Workflow
12. Dataset Generation
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

ASTRA is an **onboard experiment monitoring and verification system** that converts live camera feeds into structured experiment states.

Instead of only recognizing human activities, ASTRA understands:

- What the astronaut is doing
- Which experiment object is being interacted with
- Whether the physical action actually occurred
- Whether the action is valid at the current stage
- What the next expected step is
- Whether the astronaut has skipped, repeated or performed an incorrect action
- When the visual evidence is insufficient to make a reliable decision

The complete inference and decision pipeline is designed to operate **offline on edge hardware**, reducing dependence on continuous communication with ground control.

### Core Concept

> **SEE → UNDERSTAND → VERIFY → GUIDE**

---

# Problem Statement

Future space missions require astronauts to conduct increasingly complex scientific experiments while communication with Earth may be delayed, intermittent or bandwidth constrained.

The proposed system addresses the need for an onboard AI assistant capable of:

- Monitoring experiments through fixed cameras
- Recognizing astronaut activities
- Validating the predefined sequence of an experiment
- Detecting skipped or out-of-sequence actions
- Suggesting the next step
- Providing voice-based alerts
- Generating timestamped experiment logs
- Recording experiment video locally
- Streaming video to a specified IP
- Operating as a standalone offline system

---

# Proposed Solution

ASTRA combines **computer vision, pose estimation, temporal action recognition and deterministic experiment validation** into one edge-AI pipeline.

The system processes a live camera stream and extracts:

- Experiment objects
- Astronaut body pose
- Hand landmarks
- Hand–object interactions
- Object movement
- Rack-relative spatial relationships
- Temporal motion patterns

These signals are fused to infer an action such as:

```text
REMOVE_RED_BOX
PLACE_YELLOW_BOX
OPEN_CONTAINER
PICK_TOOL
PLACE_TOOL
# Temporal Activity Recognition vs Physical Outcome Verification

They are **not the same thing**.

### Temporal Activity Recognition
Answers:

> **“What action is the astronaut performing?”**

It looks at a sequence of frames and identifies the action from motion over time.

Example:

```text
Hand approaches red box
        ↓
Hand contacts red box
        ↓
Box moves
        ↓
Box changes location
        ↓
Interaction confirmed
```

This provides evidence that the astronaut is actually manipulating the required experiment object.

---

## 4. Temporal Activity Recognition

The system analyzes multiple consecutive frames to understand actions over time rather than making decisions from a single frame.

Example:

```text
Approach
   ↓
Contact
   ↓
Grasp
   ↓
Move
   ↓
Release
   ↓
REMOVE_RED_BOX
```

Temporal activity recognition answers:

> **What action is the astronaut performing?**

---

## 5. Physical Outcome Verification

Physical outcome verification determines whether the recognized action actually completed the intended experiment step.

Example:

```text
Before:
Red box = inside container

Astronaut interacts

After:
Red box = outside container
        ↓
Physical Outcome = VERIFIED
```

This prevents an attempted or incomplete action from being incorrectly treated as a completed experiment step.

---

## 6. Experiment Sequence Validation

The system compares recognized actions with the predefined experiment sequence.

Possible states include:

```text
CORRECT
SKIPPED
REPEATED
OUT-OF-SEQUENCE
UNKNOWN
UNCERTAIN
```

---

## 7. Next-Step Guidance

After a step is successfully verified, ASTRA automatically identifies the next expected action and displays it through the GUI.

Example:

```text
CURRENT STEP
✓ Remove Red Box

NEXT STEP
→ Remove Yellow Box
```

---

## 8. Deviation Detection & Voice Alerts

If the astronaut performs an incorrect or out-of-sequence action, the system provides a voice-based alert.

Example:

> "Incorrect sequence. Please remove the red box first."

The alert system operates locally without requiring cloud connectivity.

---

## 9. Confidence-Aware Decision Making

ASTRA does not blindly accept low-confidence predictions.

```text
High Confidence
→ Accept / Advance

Medium Confidence
→ Continue Observation

Low Confidence
→ Mark Uncertain
→ Do Not Advance
```

This reduces false experiment-state transitions.

---

## 10. Unknown Action Detection

Actions that do not correspond to known experiment activities are not forced into an existing class.

Instead:

```text
Unknown Action
     ↓
No State Transition
     ↓
Continue Monitoring
```

This prevents unexpected movements from corrupting the experiment state.

---

## 11. Microgravity-Aware Spatial Understanding

Instead of assuming a fixed floor or gravitational "down" direction, astronaut and object positions are represented relative to the **payload rack**.

This allows spatial reasoning to remain meaningful even when the astronaut is rotated or working in non-Earth-like orientations.

---

## 12. Self-Calibrating Payload Reference

Known rack geometry or visual reference markers can be used to establish the payload coordinate frame when the system starts.

```text
Detect Rack Reference
        ↓
Establish Coordinate Frame
        ↓
Track Astronaut + Objects
Relative to Rack
```

This reduces dependence on manually defined image coordinates.

---

## 13. Predictive Deviation Warning

ASTRA can analyze hand/object movement trajectories to identify a likely incorrect action before the action is fully completed.

Example:

```text
Expected Object → Red Box

Observed Hand Trajectory → Yellow Box

        ↓

Potential Deviation

        ↓

Early Voice Warning
```

This allows the system to prevent some procedural mistakes instead of detecting them only after completion.

---

## 14. Perception Health Monitoring

The system monitors the quality of the visual pipeline for conditions such as:

- Camera obstruction
- Motion blur
- Poor visibility
- Object tracking loss
- Pose tracking loss

When perception becomes unreliable, the system marks the observation as **uncertain** instead of making an unreliable decision.

---

## 15. Graceful Degradation / Fault-Tolerant Perception

If one perception component temporarily becomes unreliable, ASTRA can use other available evidence such as:

- Object tracking
- Hand-object geometry
- Temporal motion
- Previous verified state

The aim is to degrade gracefully instead of completely failing the experiment.

---

## 16. Event-Based Lightweight Telemetry

Rather than depending on continuous raw-video transmission, ASTRA converts experiment activity into compact structured events.

Example:

```text
STEP_COMPLETED
DEVIATION_DETECTED
STEP_RECOVERED
EXPERIMENT_COMPLETED
```

Each event can contain:

- Timestamp
- Step
- Action
- Confidence
- Status
- Relevant metadata

---

## 17. Delay/Disruption-Tolerant Experiment Reporting

When communication with ground control is unavailable, experiment events are stored locally.

```text
Experiment Running
       ↓
Communication Lost
       ↓
Events Stored Locally
       ↓
Experiment Continues
       ↓
Communication Restored
       ↓
Stored Events Synchronized
```

The experiment-monitoring pipeline continues to operate independently of communication availability.

---

## 18. Bandwidth-Aware Data Prioritization

Events are assigned different priorities according to mission relevance.

```text
Normal Event
→ Lightweight telemetry

Warning / Uncertain Event
→ Higher-priority event

Critical Deviation
→ Event + relevant video/context
```

This reduces unnecessary communication load while retaining important mission information.

---

## 19. Local Video Recording & Network Streaming

The system stores the experiment video locally while supporting real-time streaming to a specified IP address.

Supported capabilities include:

- Local recording
- H.264/H.265 encoding
- RTSP streaming
- Timestamp synchronization with experiment events

---

## 20. Timestamped Experiment Logging

All important experiment events are stored as structured records.

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

# Experiment Workflow

Example experiment:

```text
Outer Box
   │
   ├── Red Box
   └── Yellow Box
```

Example procedure:

```text
STEP 1 → Open Outer Box
STEP 2 → Remove Red Box
STEP 3 → Remove Yellow Box
STEP 4 → Place Red Box at Target
STEP 5 → Place Yellow Box at Target
STEP 6 → Complete Experiment
```

During execution:

```text
STEP 1 VERIFIED
      ↓
NEXT → REMOVE RED BOX
      ↓
RED BOX INTERACTION DETECTED
      ↓
PHYSICAL OUTCOME VERIFIED
      ↓
STEP 2 COMPLETED
      ↓
NEXT → REMOVE YELLOW BOX
```

If an incorrect step occurs:

```text
Expected → REMOVE RED BOX
Observed → REMOVE YELLOW BOX
      ↓
OUT-OF-SEQUENCE
      ↓
Voice Alert
      ↓
Current State Remains Unchanged
      ↓
Astronaut Corrects Action
      ↓
STEP VERIFIED
```

---

# Edge & Space Networking

ASTRA follows an **offline-first architecture**.

Critical experiment intelligence remains onboard:

```text
Camera
  ↓
AI Inference
  ↓
Experiment Validation
  ↓
Voice Guidance
  ↓
Local Logging
```

Ground communication is treated as an additional communication layer rather than a dependency for experiment execution.

## Event Prioritization

```text
Routine
→ Step completion telemetry

Warning
→ Deviation / uncertain state

Critical
→ Priority event + relevant video/context
```

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

---

# Dataset Generation

The problem requires a custom focused dataset.

ASTRA uses a combination of **real and synthetic data**.

## Real Dataset

Record experiment videos using:

- Fixed camera positions
- Multiple participants
- Different body orientations
- Different execution speeds
- Different object positions
- Different lighting conditions
- Partial occlusion
- Correct actions
- Incorrect actions
- Skipped actions
- Repeated actions

---

## Object Detection Dataset

Annotate:

- Experiment container
- Red box
- Yellow box
- Tools
- Other experiment-specific objects

Annotations can be created in YOLO format.

---

## Pose / Interaction Dataset

Capture sequences containing:

- Hand positions
- Body keypoints
- Object locations
- Hand-object contact
- Object movement
- Action start/end

---

## Synthetic Dataset

Generate controlled synthetic variations for:

- Orientation changes
- Lighting variations
- Object displacement
- Astronaut pose variations
- Camera viewpoint changes
- Background variations
- Occlusions

Synthetic data is primarily used to improve perception robustness, while real data is used for realistic validation.

---

# Training Strategy

The training process is modular.

```text
Raw Videos
     ↓
Frame Extraction
     ↓
Object Annotation
     ↓
Pose / Interaction Labels
     ↓
YOLO Training
     ↓
Pose / Hand Integration
     ↓
Temporal Feature Generation
     ↓
TCN Training
     ↓
Experiment Validation
     ↓
Real-World Testing
```

## Training Priorities

1. Object detection accuracy
2. Hand-object interaction accuracy
3. Temporal action recognition
4. Correct experiment-state transitions
5. Wrong/skip/repeated-step detection
6. Edge inference latency

---

# Installation & Setup

## 1. Clone the Repository
```bash
git clone https://github.com/yourusername/astra.git
cd astra
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

At completion, ASTRA generates:

```text
Experiment Video
Experiment Event Log
Experiment Summary
Telemetry Record
```

---

# Output & Logging

## Experiment Event Log

ASTRA stores lightweight event records using JSONL.

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

---

# Testing & Evaluation

The system should be evaluated at both **AI level and complete-system level**.

## Computer Vision Metrics

- Precision
- Recall
- mAP
- Pose accuracy
- Tracking accuracy

## Activity Recognition Metrics

- Accuracy
- Precision
- Recall
- F1-score
- Confusion matrix

## Experiment Validation Metrics

The most important system-level metrics are:

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


---

# System Modes

## Normal Mode

```text
Detect → Recognize → Verify → Advance
```

---

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

---

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

---

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
