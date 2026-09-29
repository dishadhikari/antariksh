# ASTRA — Autonomous Experiment Verification & Guidance System

**ASTRA (Autonomous Step Tracking, Reasoning & Assistance)** is an offline, edge-AI based system designed to assist astronauts during scientific experiments by continuously understanding human–object interactions, validating experiment procedures, detecting deviations, and providing real-time guidance.

Built for the **AI Human Activity Recognition for On-board BAS Experiments** problem statement.

---

## Table of Contents

1. Overview
2. Problem Statement
3. Proposed Solution
4. Key Features
5. Innovation & Uniqueness
6. System Architecture
7. Tech Stack
8. Methodology
9. System Modules
10. Experiment Workflow
11. Edge & Space Networking
12. Dataset Generation
13. Training Strategy
14. Installation & Setup
15. Configuration
16. Usage Guide
17. Project Structure
18. Output & Logging
19. Testing & Evaluation
20. Hardware Deployment
21. Performance Targets
22. Future Improvements
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

Hand approaches red box
        ↓
Hand contacts red box
        ↓
Red box moves
        ↓
Hand releases
        ↓
TEMPORAL MODEL
        ↓
"REMOVE_RED_BOX"

## Key Features — What Each Part Actually Does

### 1. Object Detection
**Answers:** “Where are the experiment objects?”

Detects and tracks objects such as the outer box, red box, yellow box and tools.

Example:
`Red box detected → position tracked across frames`

---

### 2. Human Pose & 3D HMR
**Answers:** “Where is the astronaut and how is the body oriented?”

Pose estimation extracts body keypoints, while **3D Human Mesh Recovery (HMR)** estimates the astronaut's 3D body configuration.

This is important because the astronaut may be rotated or working in arbitrary orientations in microgravity.

---

### 3. Hand–Object Interaction
**Answers:** “Is the astronaut actually interacting with the object?”

Combines hand landmarks with object positions.

Example:
`Hand approaches red box → contact → grip → object begins moving`

This provides the evidence required to understand manipulation actions.

---

### 4. Temporal Activity Recognition
**Answers:** “What action is the astronaut performing?”

Analyzes several frames over time to recognize actions such as:

`PICK → MOVE → PLACE → OPEN → REMOVE`

---

### 5. Physical Outcome Verification
**Answers:** “Did the intended action actually complete the experiment step?”

Compares the physical state before and after the action.

Example:

`Red box inside container → interaction → red box outside container → STEP VERIFIED`

---

### 6. Experiment Sequence Validation
**Answers:** “Was the correct action performed at the correct stage?”

The recognized action is compared with the expected experiment sequence.

Possible outcomes:

`CORRECT | SKIPPED | REPEATED | OUT-OF-SEQUENCE | UNCERTAIN`

---

### 7. Next-Step Guidance
**Answers:** “What should the astronaut do next?”

After a step is successfully verified:

`STEP 2 COMPLETED → NEXT: Remove Yellow Box`

The next action is shown on the GUI and can be spoken using offline TTS.

---

### 8. Deviation Detection & Voice Alerts
**Answers:** “Is something going wrong?”

Example:

`Expected: Remove Red Box`
`Observed: Remove Yellow Box`
`→ Deviation detected`
`→ Voice: "Please remove the red box first."`

---

### 9. Confidence-Aware Decision Making
**Answers:** “How certain are we?”

```text
High confidence   → Accept step
Medium confidence → Keep observing
Low confidence    → Mark uncertain / do not advance
