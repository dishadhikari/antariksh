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
