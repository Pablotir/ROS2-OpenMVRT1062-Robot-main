# Jetson Bot — AI-Guided SLAM Explorer

**Autonomous room mapping robot** powered by an NVIDIA Jetson, mecanum-drive chassis, and on-device vision AI. The robot explores a room, builds a real-time 2D map, and uses a Vision-Language Model (VILA 2.7B) to decide where to explore next—all running locally on the edge, no cloud required.

---

## Table of Contents

1. [Architecture](#1-architecture)
2. [Tech Stack & Definitions](#2-tech-stack--definitions)
3. [Why These Technologies?](#3-why-these-technologies)
4. [Hardware](#4-hardware)
5. [Docker Setup](#5-docker-setup)
6. [Running the Robot](#6-running-the-robot)
7. [AI Navigation + Scene Labelling](#7-ai-navigation--scene-labelling)
8. [Topics Reference](#8-topics-reference)
9. [Launch Arguments](#9-launch-arguments)
10. [Tuning & Calibration](#10-tuning--calibration)
11. [File Structure](#11-file-structure)
12. [Known Limitations & Next Steps](#12-known-limitations--next-steps)

---

## 1. Architecture

The system has three layers: **perception** (camera + VILA AI), **localisation** (odometry + SLAM), and **actuation** (motor commands). They're connected through ROS2 topics inside a single Docker container.

```
┌─────────────────────────────────────────────────────────────────┐
│  dustynv/nano_llm:humble  Docker Container                      │
│                                                                  │
│  USB Camera (720p)                                               │
│    └──► usb_cam ──► /image_raw ──┬──► RTAB-Map ──► /map         │
│                                  │    (visual SLAM)              │
│                                  │                               │
│                                  └──► vila_scene_labeller        │
│                                       (VILA 2.7B, 4-bit AWQ)    │
│                                       ├─► /ai/direction          │
│                                       └─► /ai/semantic_label     │
│                                                │                 │
│  Arduino Mega (USB serial)                     ▼                 │
│    └──► arduino_bridge    exploration_controller                 │
│         ├─► /wheel_ticks ──► mecanum_odometry ──► /odom          │
│         ├─► /ultrasonic_range ──────────────────►(collision gate)│
│         │                                        │               │
│         └──◄ /arduino_cmd ◄── motor_driver ◄── /cmd_vel          │
└─────────────────────────────────────────────────────────────────┘
```

**Data flow in plain English:**

1. The USB camera publishes 720p frames to `/image_raw`
2. **RTAB-Map** uses those frames + wheel odometry to build and refine a 2D map
3. **VILA 2.7B** (resized to 384×384) looks at the same frames and decides: *"go forward, turn left, or turn right?"* and labels the scene (e.g. *"near desk"*)
4. **exploration_controller** takes the AI direction, cross-checks it against the ultrasonic sensor (collision avoidance), and publishes velocity commands
5. **motor_driver** converts those velocities into per-wheel motor commands for the mecanum chassis
6. **arduino_bridge** sends them to the Arduino over serial, and reads back encoder ticks + ultrasonic distance

---

## 2. Tech Stack & Definitions

### Core Robotics Framework

| Technology | What It Is |
|---|---|
| **ROS2 (Humble)** | The Robot Operating System 2 — an open-source middleware for robotics. Provides a publisher/subscriber messaging system ("topics"), launch files, and standard message types. ROS2 Humble is a Long-Term Support (LTS) release. |
| **rclpy** | The Python client library for ROS2. All nodes in this project are written in Python using `rclpy`. |
| **Topics** | Named message channels in ROS2 (e.g. `/odom`, `/cmd_vel`). Nodes publish to or subscribe to topics — this is how components communicate without direct coupling. |
| **Launch Files** | Python scripts that start multiple ROS2 nodes together with configured parameters. One `ros2 launch` command starts the entire robot stack. |

### SLAM & Navigation

| Technology | What It Is |
|---|---|
| **SLAM** | Simultaneous Localisation and Mapping — the robot builds a map of its environment while simultaneously tracking its own position within that map. |
| **RTAB-Map** | Real-Time Appearance-Based Mapping — a visual SLAM library. Uses camera images + wheel odometry to detect "loop closures" (recognising previously visited places) and correct accumulated drift. Outputs a 2D occupancy grid (`/map`). |
| **Nav2** | The ROS2 Navigation Stack — provides path planning, obstacle avoidance costmaps, and goal-based navigation. Used here with `explore_lite` for autonomous frontier exploration. |
| **explore_lite** | A frontier-based exploration planner. Finds edges of the known map ("frontiers") and sends the robot to explore them, progressively mapping the whole room. |
| **Odometry** | Estimating the robot's position by tracking wheel encoder counts. The mecanum odometry node uses the 4-wheel kinematics model to compute x, y, and heading (θ) from encoder ticks. |

### AI / Vision-Language Model

| Technology | What It Is |
|---|---|
| **VILA 2.7B** | A Vision-Language Model (VLM) co-developed by NVIDIA and MIT. It can look at an image and answer natural-language questions about it — like *"what room is this?"* or *"which direction has the most open space?"*. The 2.7-billion parameter version fits in ~2 GB of GPU RAM when quantised. |
| **AWQ (4-bit)** | Activation-Aware Weight Quantisation — a compression technique that shrinks model weights from 16-bit to 4-bit with minimal accuracy loss. This is how VILA fits on a Jetson with only 8 GB of shared GPU/CPU memory. |
| **nano_llm** | NVIDIA's optimised inference library for running LLMs and VLMs on Jetson. Handles model loading, quantisation, and GPU-accelerated generation. Much faster than generic frameworks like Ollama on the Jetson's ARM+CUDA architecture. |
| **NanoLLM Container** | `dustynv/nano_llm:humble` — a pre-built Docker image that bundles nano_llm + ROS2 Humble + CUDA + TensorRT + cuDNN. This is the runtime environment for the robot. |

### Hardware Interfaces

| Technology | What It Is |
|---|---|
| **Mecanum Wheels** | Wheels with 45° angled rollers that allow the robot to move in any direction (forward, sideways, diagonal) without turning. Require 4 independently controlled motors. |
| **Arduino Mega** | The low-level motor controller. Runs a firmware loop that accepts B-commands over serial (e.g. `B100,-100,100,-100`), drives the HiTechnic motor controllers, and reports encoder counts + ultrasonic distance back to the Jetson. |
| **HC-SR04** | An ultrasonic distance sensor. Sends a 40 kHz pulse and measures echo return time → distance in metres. Used as a forward-facing collision limiter (range ~0.02–4 m). |
| **cv_bridge** | A ROS2 library that converts between ROS `sensor_msgs/Image` messages and OpenCV/NumPy arrays. Needed by the VILA node to preprocess camera frames before inference. |

### Containerisation

| Technology | What It Is |
|---|---|
| **Docker** | A container platform that packages software + all dependencies into isolated, reproducible environments. The entire ROS2 + AI stack runs inside a Docker container, so nothing needs to be installed directly on the Jetson's host OS. |
| **Docker Compose** | A tool for defining multi-container setups in a YAML file (`docker-compose.yml`). One command (`docker compose up`) instantiates the container with correct GPU access, device mounts, and volumes. |
| **NVIDIA Container Runtime** | `runtime: nvidia` in Docker Compose — gives the container access to the Jetson's GPU, CUDA libraries, and TensorRT. Without this, GPU-accelerated inference won't work. |
| **Named Volumes** | Docker volumes that persist data across container restarts. Used here for `nano_llm_models` (so the 2+ GB VILA model doesn't re-download every time) and `maps` (saved SLAM maps). |

---

## 3. Why These Technologies?

### Why VILA 2.7B instead of Ollama + Gemma3:4b?

| | Ollama + Gemma3:4b (old) | nano_llm + VILA 2.7B (new) |
|---|---|---|
| **GPU RAM** | ~4 GB (too tight on 8 GB Jetson) | ~1.8–2 GB (leaves 3+ GB for SLAM) |
| **Inference speed** | 5–10 s per frame | 0.5–1.5 s per frame |
| **Architecture** | HTTP server in separate container | Native Python API, same container |
| **Spatial reasoning** | Generic chat model | Trained specifically for edge robotics |
| **ROS2 integration** | Manual HTTP calls + base64 encoding | Direct image input via nano_llm |

### Why Docker instead of native install?

- **Reproducibility:** The exact same environment runs on any Jetson with JetPack 6, regardless of what's installed on the host
- **Dependency isolation:** CUDA, TensorRT, ROS2, Python packages — all pinned inside the container, no conflicts with host packages
- **Easy rollback:** If something breaks, `docker compose down` + `docker compose up` gives a fresh start
- **Live-reload:** The `./src` directory is mounted into the container, so Python code changes on the Jetson (or synced from Windows) take effect immediately after a `colcon build`

### Why ROS2 Humble instead of Iron?

The `dustynv/nano_llm` container images are built on **ROS2 Humble** (the current LTS release). Since all our Python nodes use standard `rclpy` APIs and message types, they work identically on both Iron and Humble — no code changes were needed for the switch.

### Why mecanum odometry instead of visual odometry?

RTAB-Map can do visual odometry (estimating motion from camera images alone), but monocular visual odometry has **scale ambiguity** — it can't tell whether the robot moved 10 cm or 10 m from images alone. Wheel odometry from the encoders provides metric scale, and RTAB-Map uses its visual loop closures to correct accumulated drift. The two are complementary.

---

## 4. Hardware

| Component | Model | Role |
|---|---|---|
| **Compute** | NVIDIA Jetson Orin Nano (8 GB) | Runs ROS2, SLAM, and VILA inference on GPU |
| **Motor Controller** | Arduino Mega 2560 | Receives B-commands over USB serial, drives 4 HiTechnic motor controllers |
| **Drive** | 4× Mecanum wheels (100 mm / 0.0508 m radius) | Omnidirectional movement |
| **Camera** | USB camera (720p, 5 MP) | Visual input for SLAM + AI |
| **Distance** | HC-SR04 ultrasonic sensor | Forward-facing collision avoidance |
| **Encoders** | 1440 ticks/rev (360 CPR × 4 quadrature) | Wheel odometry for position tracking |

### Robot Dimensions (used in odometry & motor kinematics)

| Parameter | Value | Description |
|---|---|---|
| `wheel_radius` | 0.0508 m | 100 mm diameter mecanum wheels |
| `half_wheelbase` | 0.1270 m | Centre-to-wheel distance (front–back) |
| `half_track_width` | 0.2172 m | Centre-to-wheel distance (left–right) |

---

## 5. Docker Setup

### How Docker Works in This Project

The entire robot stack runs inside **one Docker container** based on `dustynv/nano_llm:humble`. This image comes pre-loaded with:
- ROS2 Humble (full desktop)
- nano_llm (NVIDIA's VLM inference library)
- CUDA 12 + TensorRT + cuDNN (GPU acceleration)
- Python 3.10 + pip

Our code is **mounted into the container** via Docker Compose volumes — it's not baked into the image. This means:
- You edit code on your Windows dev machine
- SCP or VSCode Remote SSH it to the Jetson
- The container sees the changes immediately (via the `./src` mount)
- You just `colcon build` inside the container to recompile

### `docker-compose.yml` Explained

```yaml
services:
  robot:
    image: dustynv/nano_llm:humble-r36.3.0    # Pre-built image with ROS2 + nano_llm
    container_name: jetson_bot_vila             # Name for docker exec commands
    network_mode: "host"                        # Share Jetson's network (ROS2 DDS needs this)
    runtime: nvidia                             # GPU passthrough (CUDA, TensorRT)
    devices:
      - /dev/ttyUSB0:/dev/ttyUSB0              # Arduino serial port
      - /dev/video0:/dev/video0                # USB camera (index 0)
      - /dev/video1:/dev/video1                # USB camera (index 1)
    volumes:
      - ./src:/root/ros2_ws/src                # Your ROS2 packages (live-reload)
      - ./maps:/root/maps                      # Persisted SLAM maps
      - ./setup_vila.sh:/root/ros2_ws/setup_vila.sh  # Setup script
      - nano_llm_models:/root/.cache           # Cached VILA model weights
```

Key design choices:
- **`network_mode: "host"`** — ROS2 uses DDS for discovery, which relies on multicast. Bridge networking breaks this. Host networking lets all ROS2 topics be visible from the Jetson host and any laptop on the same network.
- **`runtime: nvidia`** — Without this, the container can't access the Jetson's GPU. nano_llm and TensorRT would fail silently.
- **Named volume `nano_llm_models`** — The VILA model is ~2 GB. This volume persists it across container restarts so it only downloads once.

### First-Time Setup (on the Jetson)

```bash
# 1. Navigate to the project directory
cd ~/ros2_robot    # or wherever ROS2-OpenMVRT1062-Robot-main lives

# 2. Stop any old containers
docker compose down

# 3. Pull the nano_llm container image (~12 GB download, one time)
docker compose pull

# 4. Start the container
docker compose up -d

# 5. Enter the container
docker exec -it jetson_bot_vila bash

# 6. Run the one-time setup script (installs ROS deps, builds workspace, downloads VILA)
bash /root/ros2_ws/setup_vila.sh
```

### What `setup_vila.sh` Does

This script runs **inside the container** and installs everything not included in the base `dustynv/nano_llm` image:

1. **ROS2 apt packages:** `usb-cam`, `rtabmap-ros`, `navigation2`, `nav2-bringup`, `xacro`, `robot-state-publisher`, `tf2-ros`, `cv-bridge`, `explore-lite`
2. **Python packages:** `pyserial` (Arduino), `pillow` (image processing), `numpy`, `opencv-python-headless`
3. **Workspace build:** `colcon build --symlink-install` for both `robot_control` and `jetson_bot_slam`
4. **VILA model download:** Pre-downloads `Efficient-Large-Model/VILA-2.7b` so the first inference is fast

### Deploying Code Changes (Windows → Jetson)

```powershell
# From your Windows machine (PowerShell):
scp -r "ROS2-OpenMVRT1062-Robot-main\src" pablo@<JETSON_IP>:~/ros2_robot/
```

Then inside the container:
```bash
cd /root/ros2_ws
colcon build --symlink-install --packages-select robot_control jetson_bot_slam
source install/setup.bash
```

---

## 6. Running the Robot

All commands below run **inside the Docker container** on the Jetson.

### Daily Start

```bash
# On the Jetson host:
cd ~/ros2_robot
docker compose up -d
docker exec -it jetson_bot_vila bash

# Inside the container:
source /opt/ros/humble/setup.bash
source /opt/ros/install/setup.bash
source /root/ros2_ws/install/setup.bash
ros2 launch jetson_bot_slam ai_slam_explore.launch.py
```

### Stop

```bash
# In the launch terminal:
Ctrl+C
exit
docker compose down
```

### First Time Only (after image re-pull or fresh setup)

```bash
cd ~/ros2_robot
docker compose up -d
docker exec -it jetson_bot_vila bash
bash /root/ros2_ws/setup_vila.sh
```

### What the Robot Does

1. VILA loads in the background (~10–30 s on first run, ~5 s cached)
2. The exploration controller drives the robot forward until an obstacle is within 30 cm
3. On obstacle: backs up → scans left/right → finds open path → resumes forward
4. Every 5 obstacle-turns, VILA is triggered: *"Where am I? Which way should I explore?"*
5. VILA responds with a direction (forward/left/right) and a scene label (e.g. "near desk")
6. AI decisions + camera frames are logged to `./logs/` on the Jetson host

### Enable SLAM Mapping

SLAM is off by default (camera outputs YUV which needs conversion). To enable:

```bash
ros2 launch jetson_bot_slam ai_slam_explore.launch.py use_slam:=true
```

View saved maps: copy `/root/maps/bedroom.db` to your PC and open with the [RTAB-Map desktop app](https://github.com/introlab/rtabmap/releases).

### Save the Map

```bash
ros2 run nav2_map_server map_saver_cli -f /root/maps/bedroom_map
# Maps persist to ./maps/ on the Jetson host via the Docker volume
```

### Monitor (second terminal)

```bash
docker exec -it jetson_bot_vila bash
source /root/ros2_ws/source_all.bash
ros2 topic echo /ai/decision          # VILA decisions (JSON)
ros2 topic echo /ultrasonic_range     # distance sensor
ros2 topic echo /cmd_vel              # motor commands
```

### Keyboard Teleop (manual driving)

```bash
docker exec -it jetson_bot_vila bash
source /root/ros2_ws/source_all.bash
ros2 run teleop_twist_keyboard teleop_twist_keyboard
```

### Docker Maintenance

```bash
# Update container image (clean swap):
docker compose pull && docker compose up -d --remove-orphans && docker image prune -f

# Reclaim disk space:
docker system prune -f --filter 'until=168h'
```

---

## 7. AI Navigation + Scene Labelling

### How VILA Guides Exploration

The `vila_scene_labeller` node is the AI brain of the robot. When triggered:

1. **Grabs** the latest 720p camera frame from `/image_raw`
2. **Resizes** to 384×384 (VILA's native input resolution)
3. **Asks VILA:** *"Which direction should the robot explore? What area is this?"*
4. **Publishes** the direction to `/ai/direction` and scene label to `/ai/semantic_label`
5. The `exploration_controller` reads `/ai/direction` and moves accordingly, using the ultrasonic sensor as a hard collision limiter

The prompt asks VILA to respond in a structured format:
```
DIRECTION: forward
LABEL: open floor near bed
REASON: Clear path ahead with open space to explore
```

### How the AI Decision Flows

```
VILA says "left"  ──► exploration_controller checks ultrasonic
                       ├── ultrasonic > 0.10 m → OK, turn left
                       └── ultrasonic < 0.10 m → emergency stop + reverse
```

The ultrasonic sensor **always overrides** the AI direction for safety. If VILA says "forward" but an obstacle is 5 cm away, the robot reverses.

### Configuration

Edit `src/jetson_bot_slam/config/vila_params.yaml`:

```yaml
vila_scene_labeller:
  ros__parameters:
    model_name:     "Efficient-Large-Model/VILA-2.7b"
    api:            "awq"
    quantization:   "q4f16_ft"
    infer_width:    384        # resize from 720p before inference
    infer_height:   384
    max_new_tokens: 64
    prompt: >-
      You are a navigation assistant for a robot exploring a room...
```

---

## 8. Topics Reference

### Hardware Topics

| Topic | Type | Source | Description |
|---|---|---|---|
| `/image_raw` | `Image` | usb_cam | 720p camera feed |
| `/camera_info` | `CameraInfo` | usb_cam | Camera calibration |
| `/wheel_ticks` | `Int32MultiArray` | arduino_bridge | Encoder counts [BL, BR, FL, FR] |
| `/ultrasonic_range` | `Range` | arduino_bridge | HC-SR04 distance (metres) |
| `/odom` | `Odometry` | mecanum_odometry | Robot position from encoders |
| `/cmd_vel` | `Twist` | exploration_controller | Velocity commands |
| `/arduino_cmd` | `String` | motor_driver | Raw B-commands to Arduino |

### AI Topics

| Topic | Type | Source | Description |
|---|---|---|---|
| `/ai/direction` | `String` | vila_scene_labeller | `forward`, `left`, `right`, or `stop` |
| `/ai/semantic_label` | `String` | vila_scene_labeller | Scene label, e.g. "near desk" |
| `/ai/decision` | `String` | vila_scene_labeller | Full JSON with direction, label, reason, timing |

### Map Topics

| Topic | Type | Source | Description |
|---|---|---|---|
| `/map` | `OccupancyGrid` | rtabmap | 2D occupancy grid map |
| `/explore/frontiers` | `MarkerArray` | explore_lite | Current frontier points |

---

## 9. Launch Arguments

```bash
ros2 launch jetson_bot_slam ai_slam_explore.launch.py \
  serial_port:=/dev/ttyUSB0 \
  camera_device:=/dev/video0 \
  vila_model:=Efficient-Large-Model/VILA-2.7b \
  use_slam:=true \
  move_speed:=0.20
```

| Argument | Default | Description |
|---|---|---|
| `serial_port` | `/dev/ttyUSB0` | Arduino serial port |
| `camera_device` | `/dev/video0` | USB camera device |
| `rviz` | `false` | Launch RViz2 (needs display attached) |
| `vila_model` | `Efficient-Large-Model/VILA-2.7b` | VLM model name |
| `vila_api` | `awq` | Quantisation backend |
| `vila_quantization` | `q4f16_ft` | Quantisation precision |
| `use_slam` | `false` | Enable RTAB-Map SLAM (needs rgb camera) |
| `move_speed` | `0.20` | Forward speed (m/s) |
| `turn_speed` | `0.55` | Turn speed (rad/s) |
| `obstacle_distance` | `0.30` | Begin turning at this range (m) |
| `emergency_stop_dist` | `0.08` | Emergency reverse below this range (m) |
| `label_every` | `5` | Run AI scene label every N obstacle-turns |

---

## 10. Tuning & Calibration

### Odometry Calibration

If the robot drifts (e.g. driving straight curves left):

```bash
# Reset encoders
ros2 topic pub /arduino_cmd std_msgs/msg/String "data: 'RESET'" --once

# Echo odom while driving 1 m forward via teleop
ros2 topic echo /odom --field pose.pose.position
```

Adjust in `ai_slam_explore.launch.py`:
- `half_wheelbase` — increase if forward distance is under-reported
- `half_track_width` — increase if rotation angle is under-reported

### Camera Calibration

For better SLAM loop closures:
```bash
ros2 run camera_calibration cameracalibrator \
  --size 8x6 --square 0.025 \
  image:=/image_raw camera:=/camera
```

---

## 11. File Structure

```
ROS2-OpenMVRT1062-Robot-main/          ← this project (mirrors ~/ros2_robot on Jetson)
├── docker-compose.yml                 — Container config (nano_llm + GPU + devices)
├── setup_vila.sh                      — One-time setup script (run inside container)
├── ros2.Dockerfile                    — Legacy Dockerfile (kept for reference)
├── entrypoint.sh                      — Legacy entrypoint (kept for reference)
├── maps/                              — Saved SLAM maps (persisted via Docker volume)
└── src/
    ├── robot_control/                 — AI + exploration package
    │   └── robot_control/
    │       ├── ai_navigator.py            — Old Ollama scene labeller (replaced by VILA)
    │       ├── exploration_controller.py  — Ultrasonic-reactive exploration + AI fusion
    │       ├── usb_camera_node.py         — USB camera publisher
    │       └── ...
    └── jetson_bot_slam/               — Hardware + SLAM package
        ├── jetson_bot_slam/
        │   ├── arduino_bridge_node.py       — Serial ↔ ROS2 bridge (encoders + ultrasonic)
        │   ├── mecanum_odometry_node.py     — 4-wheel odometry → /odom + TF
        │   ├── motor_driver_node.py         — cmd_vel → B-commands (mecanum IK)
        │   └── vila_scene_labeller_node.py  — VILA 2.7B AI navigation + labelling ★
        ├── launch/
        │   ├── robot_bringup.launch.py      — Hardware bringup only
        │   ├── slam.launch.py               — RTAB-Map SLAM
        │   ├── explore.launch.py            — Nav2 + frontier exploration
        │   └── ai_slam_explore.launch.py    — Full stack: SLAM + VILA + exploration ★
        ├── config/
        │   ├── nav2_params.yaml             — Nav2 / DWB planner tuning
        │   └── vila_params.yaml             — VILA model + prompt config ★
        ├── urdf/
        │   └── jetson_bot.urdf.xacro        — Robot URDF (TF tree)
        └── rviz/
            └── slam_view.rviz               — RViz2 visualisation config
```

★ = new files added for VILA integration

---

## 12. AI Decision Logging

Every session automatically creates a timestamped log folder at `./logs/<YYYYMMDD_HHMMSS>/`:

```
logs/20260325_164500/
├── decisions_full.txt          # every AI decision with timestamp + motor command
└── captures/
    ├── capture_001.jpg         # camera frame at decision #5
    ├── capture_001.json        # metadata (label, direction, reason, motor cmd)
    ├── capture_002.jpg         # camera frame at decision #10
    ├── capture_002.json
    └── ...                     # up to 10 captures (1 per 5 decisions)
```

### `decisions_full.txt` format

```
2026-03-25 16:45:12.345 | #001 | dir=forward  | label="open floor" | reason="clear path ahead" | motor=(linear=0.200 angular=0.000) | infer=1.23s
2026-03-25 16:45:30.678 | #002 | dir=left     | label="near desk"  | reason="wall on right"   | motor=(linear=0.000 angular=0.550) | infer=0.89s
```

Each line includes: timestamp, decision number, AI direction, scene label, AI reasoning, current motor command, and inference time.

### Capture metadata (`capture_NNN.json`)

```json
{
  "timestamp": "2026-03-25 16:45:30.678",
  "decision_num": 5,
  "direction": "left",
  "label": "near desk",
  "reason": "wall on right side",
  "motor_cmd": "linear=0.000 angular=0.550",
  "inference_s": 0.89,
  "image_file": "capture_001.jpg",
  "capture_num": 1
}
```

### Configuration

| Parameter | Default | Description |
|---|---|---|
| `capture_every` | `5` | Save an image every N AI decisions |
| `max_captures` | `10` | Maximum number of images to save per session |
| `log_dir` | `/root/ros2_ws/logs` | Log output directory (mounted to `./logs/` on host) |

---

## 13. Known Limitations & Next Steps

| Limitation | Mitigation |
|---|---|
| Single ultrasonic sensor (forward only) | Rotate frequently; future: add side sensors or LiDAR |
| Camera outputs YUV (RTAB-Map needs RGB) | SLAM disabled by default; add image_proc converter |
| No IMU → rotation relies on encoders | Mecanum wheels slip on carpet; future: add MPU6050 IMU |
| Loop closure needs texture-rich scenes | Good for bedrooms; struggles in featureless corridors |

**Planned upgrades:**
1. **RPLIDAR A1** → 360° LiDAR for proper SLAM + obstacle detection (rear-right mount)
2. **5DOF arm** → front-left mounted claw arm for object interaction
3. **2nd ultrasonic sensor** → right-side collision detection
4. **IMU** (MPU6050) → fuse with wheel odom in `robot_localization` EKF for better heading
5. **Semantic map overlay** → plot AI scene labels onto the SLAM map grid
