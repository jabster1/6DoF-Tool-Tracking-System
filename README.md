# 6DoF Tool Tracking System

A real-time computer vision system for industrial tool tracking using a fine tuned YOLOv11 model. Designed to monitor tools entering and exiting designated work areas, automatically logging check-in/check-out events to prevent Foreign Object Debris (FOD) incidents and reduce costly equipment loss in aerospace and power generation environments.

---

## Motivation

In industrial environments such as gas turbine maintenance and aerospace manufacturing, a single tool left inside a unit can cause catastrophic equipment failure and unplanned shutdowns — costing millions in downtime and repairs. This system provides an automated, vision-based solution to track tools in real time, log their presence, and alert operators when tools leave or fail to return to a designated area.

---

## Features

- **Real-time multi-class tool detection** via YOLOv11 on live webcam feed
- **Automated inference logging** — tool detections written to external log file with timestamps
- **Targeted tool classes** — drill, pliers, screwdriver (expanding to full industrial toolset)
- **Check-in / Check-out tracking** — monitors tools entering and leaving designated zones
- **C++ deployment pipeline** via ONNX Runtime *(in development)*
- **6-Degree-of-Freedom (6DoF) pose estimation** for precise spatial tool tracking *(in development)*

---

## System Architecture

```
Webcam Feed
     │
     ▼
YOLOv11 Inference (Python)
     │
     ├──▶ Bounding Box + Class Detection
     │
     ├──▶ Zone Entry/Exit Logic
     │
     └──▶ Logging Engine ──▶ External Log File / Enterprise System
```

**Planned C++ Pipeline:**
```
Webcam Feed ──▶ ONNX Runtime (C++ Inference) ──▶ 6DoF Pose Estimator ──▶ Tool Registry
```

---

## Tech Stack

| Component | Technology |
|---|---|
| Object Detection | YOLOv11 (Ultralytics) |
| Computer Vision | OpenCV |
| Language | Python 3.10+ → C++ (in development) |
| Deployment | ONNX Runtime (in development) |
| Logging | Python logging / external file output |

---

## Project Status

| Milestone | Status |
|---|---|
| Real-time YOLOv11 inference on webcam feed | ✅ Complete |
| Multi-class detection (drill, pliers, screwdriver) | ✅ Complete |
| Tool check-in/check-out logging | ✅ Complete |
| Fine-tuning on domain-specific industrial toolset | 🔄 In Progress |
| ONNX model export for C++ deployment | 🔄 In Progress |
| C++ inference pipeline via ONNX Runtime | 🔄 In Progress |
| 6DoF pose estimation integration | 📅 Planned |
| Enterprise system integration | 📅 Planned |

---

## Getting Started

### Prerequisites

```bash
pip install -r requirements.txt
```

### Run

```bash
python tool_tracker.py
```

Make sure your webcam is connected. Detections will be logged to `logs/tool_log.txt`.

---

## Use Case

This system is being developed for deployment in **aerospace and power generation maintenance facilities** where FOD prevention is safety-critical. The goal is a lightweight, real-time pipeline that integrates with existing enterprise asset management systems — flagging missing tools before equipment is closed up and returned to service.

---

## Roadmap

- [ ] Fine-tune YOLOv11 on labeled industrial tool dataset
- [ ] ONNX export and C++ inference pipeline
- [ ] 6DoF pose estimation for spatial awareness
- [ ] Zone-based alert system (tool left in restricted area)
- [ ] Integration with enterprise logging/asset management systems
- [ ] Edge deployment on embedded hardware (Jetson Nano / Raspberry Pi)

---

## Author

**Jaden Barnwell**
M.S. Computer Vision Candidate — University of Central Florida
[GitHub](https://github.com/jabster1) | [LinkedIn](https://linkedin.com/in/jadenbarnwell)

---
