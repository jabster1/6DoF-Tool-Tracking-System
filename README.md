# 6DoF Edge Tool Tracking System

A real-time computer vision system for industrial tool tracking using a fine tuned YOLOv11 model. Designed to monitor tools entering and exiting designated work areas, automatically logging check-in/check-out events to prevent Foreign Object Debris (FOD) incidents and reduce costly equipment loss in aerospace and power generation environments.

---

## Quickstart

mkdir -p build && cd build
cmake ..
make -j$(nproc)
./tool_tracker_trt

Make sure your webcam is connected. Detections will be logged to `logs/tool_log.txt`.

---

## Motivation

In industrial environments such as gas turbine maintenance and aerospace manufacturing, a single tool left inside a unit can cause catastrophic equipment failure and unplanned shutdowns — costing millions in downtime and repairs. This system provides an automated, vision-based solution to track tools in real time, log their presence, and alert operators when tools leave or fail to return to a designated area.

---

## Features

- **Real-time multi-class tool detection** via YOLOv11 on live webcam feed
- **Automated inference logging** — tool detections written to external log file with timestamps
- **Targeted tool classes** — drill, pliers, screwdriver (expanding to full industrial toolset)
- **Check-in / Check-out tracking** — monitors tools entering and leaving designated zones
- **C++ deployment pipeline** via ONNX Runtime
- **Jetson Orin Nano and TensorRT Integration**
- **IMU Sensor Fusion** for camera orientation and movement *(in development)*
- **EKF State Estimation** for better 3D pose detection 

---

## System Architecture

```
Webcam Feed
     │
     ▼
YOLOv11 Inference
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

## Key Files

| File | What it does |
|---|---|
| `tool_tracker_onnx.cpp` | Main C++ pipeline: YOLOv11 inference via ONNX Runtime, ByteTrack IDs, zone check-in/check-out logic |
| `bytetrack.h` | Multi-object tracker linking detections across frames |
| `imu_read.cpp` | BNO055 IMU bring-up over I2C; timestamped accel/gyro logging — the data source for sensor fusion |
| `best.onnx` | Fine-tuned YOLOv11 weights (not in repo — see note below) |

---

## Tech Stack

| Component | Technology |
|---|---|
| Object Detection | YOLOv11 (Ultralytics) |
| Computer Vision | OpenCV |
| Language | Python 3.10+ → C++  |
| Deployment | ONNX Runtime | TensorRT |
| Logging | logging / external file output |

---

## Project Status

| Milestone | Status |
|---|---|
| Real-time YOLOv11 inference on webcam feed | ✅ Complete |
| Multi-class detection (drill, pliers, screwdriver) | ✅ Complete |
| Tool check-in/check-out logging | ✅ Complete |
| Fine-tuning on domain-specific industrial toolset | ✅ Complete  |
| ONNX model export for C++ deployment | ✅ Complete  |
| C++ inference pipeline via ONNX Runtime | ✅ Complete  |
| ByteTrack implementation for linking detection across frames and apply specific ID to tools | ✅ Complete |
| Logging to track and audit tools based on ID | ✅ Complete |
| Kalman Filter Implementation | ✅ Complete |
| TensorRT implementation and Jetson Orin Nano Hardware Integration | ✅ Complete |
| IMU Sensor Fusion Implementation with Arduino BNO055 and soldering/wiring to the Jetson | In Progress |
| EKF State Estimation Extension | In Progress |

---

## Use Case

This system is being developed for deployment in **aerospace and power generation maintenance facilities** where FOD prevention is safety-critical. The goal is a lightweight, real-time pipeline that integrates with existing enterprise asset management systems — flagging missing tools before equipment is closed up and returned to service.

---

## Roadmap

- [ ] Fine-tune YOLOv11 on labeled industrial tool dataset
- [ ] ONNX export and C++ inference pipeline
- [ ] Integration with enterprise logging/asset management systems
- [ ] Edge deployment on embedded hardware (Jetson Nano / TensorRT)
- [ ] IMU BNO055 sensor soldering, wiring to the jetson, and testing
- [ ] IMU sensor fusion
- [ ] EKF state estimation

---

## Author

**Jaden Barnwell**
M.S. Computer Vision Candidate — University of Central Florida
[GitHub](https://github.com/jabster1) | [LinkedIn](https://linkedin.com/in/jaden-barnwell-09734a212)

---
