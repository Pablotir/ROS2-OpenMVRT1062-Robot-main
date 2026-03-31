# Jetson Bot — AI-Guided SLAM Explorer

**Autonomous room mapping robot** powered by an NVIDIA Jetson, mecanum-drive chassis, STL-27L 360° LiDAR, and on-device vision AI. The robot explores rooms, builds a real-time 2D map with LiDAR SLAM, and uses a Vision-Language Model (VILA 2.7B) to identify rooms and decide where to explore next—all running locally on the edge, no cloud required.

---

## Table of Contents

1. [Architecture](#1-architecture)
2. [Tech Stack & Definitions](#2-tech-stack--definitions)
3. [Why These Technologies?](#3-why-these-technologies)
4. [Hardware](#4-hardware)
5. [Docker Setup](#5-docker-setup)
6. [Running the Robot](#6-running-the-robot)
7. [Upgrading Files on the Jetson](#7-upgrading-files-on-the-jetson)
8. [AI Navigation + Scene Labelling](#8-ai-navigation--scene-labelling)
9. [Room Identification](#9-room-identification)
10. [AI Decision Logging](#10-ai-decision-logging)
11. [Topics Reference](#11-topics-reference)
12. [Launch Arguments (All Flags)](#12-launch-arguments-all-flags)
13. [Tuning & Calibration](#13-tuning--calibration)
14. [File Structure & Descriptions](#14-file-structure--descriptions)
15. [Known Limitations & Next Steps](#15-known-limitations--next-steps)

---

## 1. Architecture

The system has four layers: **perception** (LiDAR + camera + VILA AI), **localisation** (odometry + LiDAR SLAM), **intelligence** (room identification), and **actuation** (motor commands). All connected through ROS2 topics inside a single Docker container.

```
┌──────────────────────────────────────────────────────────────────────┐
│  dustynv/nano_llm:humble  Docker Container                          │
│                                                                      │
│  STL-27L LiDAR (360°)                                               │
│    └──► ldlidar_stl_ros2 ──► /scan ──┬──► RTAB-Map ──► /map        │
│                                      │    (LiDAR ICP SLAM)          │
│                                      │                               │
│                                      └──► exploration_controller    │
│                                           (360° obstacle zones)      │
│                                                                      │
│  USB Camera (640x480)                                                │
│    └──► usb_cam ──► /image_raw ──► image_proc ──► /image_rect_color │
│                  │                                                   │
│                  └──► vila_scene_labeller                            │
│                       (VILA 2.7B, 4-bit AWQ)                        │
│                       ├─► /ai/direction                              │
│                       ├─► /ai/semantic_label                         │
│                       └─► /ai/room (room identification)             │
│                                                                      │
│  Arduino Mega (USB serial)                                           │
│    └──► arduino_bridge                                               │
│         ├─► /wheel_ticks ──► mecanum_odometry ──► /odom             │
│         ├─► /ultrasonic_range ──► exploration_controller (rear only) │
│         │                                        │                   │
│         └──◄ /arduino_cmd ◄── motor_driver ◄── /cmd_vel             │
└──────────────────────────────────────────────────────────────────────┘
```

**Data flow in plain English:**
1. The **STL-27L LiDAR** scans 360° at 21,600 samples/sec and publishes `/scan`
2. **RTAB-Map** uses the LiDAR scan + wheel odometry to build a 2D map
3. The **exploration controller** reads all 4 LiDAR zones (front/left/right/rear) to avoid obstacles, with the rear ultrasonic as a backup during reverse
4. Every 5 turns, **VILA 2.7B** analyses a camera frame, labels the scene, and suggests a direction
5. The **room identifier** accumulates VILA labels over time to infer which room the robot is in
6. **Motor driver** converts velocity commands into individual wheel speeds for the mecanum chassis

---

## 2. Tech Stack & Definitions

| Component | Technology | Purpose |
|---|---|---|
| **Robot OS** | ROS2 Humble | Message passing between nodes (pub/sub topics) |
| **Container** | Docker + `dustynv/nano_llm:humble-r36.3.0` | Reproducible environment with CUDA/TensorRT |
| **AI Model** | VILA 2.7B (4-bit AWQ via nano_llm) | On-device vision-language model for scene labels |
| **LiDAR** | LDROBOT STL-27L (360°, 21.6k samples/s) | Primary obstacle detection + SLAM |
| **SLAM** | RTAB-Map (ICP mode) | Builds 2D occupancy grid from LiDAR + odometry |
| **Camera** | USB camera (640×480, YUYV) | Visual input for VILA scene analysis |
| **Microcontroller** | Arduino Mega 2560 | Motor control, encoder reading, ultrasonic sensor |
| **Ultrasonic** | HC-SR04 (rear-mounted) | Backup safety sensor for reverse maneuvers |
| **Drive** | 4× mecanum wheels | Omnidirectional movement |
| **Compute** | NVIDIA Jetson Orin Nano (JetPack 6) | GPU inference + ROS2 processing |

### What is each thing?

- **ROS2**: A robotics middleware — nodes communicate via typed "topics". Think of it as a message bus where each sensor/actuator is a separate process.
- **RTAB-Map**: A SLAM library. SLAM = Simultaneous Localization And Mapping. It builds a map while figuring out where the robot is on that map.
- **VILA 2.7B**: NVIDIA's Vision-Language Model. Given an image, it answers questions about what it sees. We use it for scene labelling and exploration guidance.
- **nano_llm**: NVIDIA's optimized inference library for LLMs on Jetson (handles TensorRT, quantization, GPU memory).
- **ICP**: Iterative Closest Point — the algorithm RTAB-Map uses to match consecutive LiDAR scans for mapping.
- **Mecanum wheels**: Special wheels with rollers that allow the robot to move sideways (strafe) without turning.
- **TF**: Transform tree — the coordinate system that tells ROS2 where each sensor is physically located on the robot.
- **URDF/xacro**: XML files that describe the robot's physical structure (dimensions, sensor positions).

---

## 3. Why These Technologies?

| Choice | Alternative Considered | Why We Chose This |
|---|---|---|
| VILA 2.7B via nano_llm | Ollama + Gemma3 4B | 2× less GPU RAM (1.8 vs 4 GB), native Jetson CUDA |
| STL-27L LiDAR | RPLIDAR A1, LD06 | 21.6k samples/s, 25m range, enclosed design, same price |
| RTAB-Map (ICP) | SLAM Toolbox, Cartographer | Best multi-sensor fusion, visual+LiDAR+odom support |
| Single Docker container | Two containers (Ollama + ROS2) | Lower RAM overhead, simpler networking |
| HC-SR04 (rear) | Removing it entirely | Catches glass/transparent surfaces LiDAR misses |

---

## 4. Hardware

| Part | Model | Connection | Details |
|---|---|---|---|
| Compute | NVIDIA Jetson Orin Nano | — | JetPack 6 (L4T R36.x), 8GB RAM |
| LiDAR | LDROBOT STL-27L | USB (UART `/dev/ttyUSB1`) | 360°, 21.6k samples/s, 25m range |
| Camera | USB webcam | USB (`/dev/video0`) | 640×480 YUYV, 15fps |
| Microcontroller | Arduino Mega 2560 | USB (`/dev/ttyUSB0`) | 115200 baud serial |
| Motors | 4× DC motors + encoders | Arduino | 1440 ticks/rev per wheel |
| Drive | 4× mecanum wheels | — | 2 in (50.8mm) radius |
| Ultrasonic | HC-SR04 | Arduino pins TRIG=12 ECHO=13 | Rear-mounted, facing backward |
| Chassis | Custom (13.5"×18.5"×13") | — | 14.37" tall with LiDAR mounted |

### Wiring

| Jetson USB Port | Device |
|---|---|
| Bottom-left (USB0) | Arduino Mega |
| Above Arduino (USB1) | STL-27L LiDAR |
| Any remaining USB | USB camera |

---

## 5. Docker Setup

### Prerequisites
- JetPack 6 installed on Jetson
- Docker + NVIDIA Container Runtime configured
- Clone this repo to `~/ros2_robot/` on the Jetson

### First-Time Setup

```bash
cd ~/ros2_robot
docker compose up -d                           # pulls ~13.6 GB image
docker exec -it jetson_bot_vila bash            # enter container
bash /root/ros2_ws/setup_vila.sh               # install deps + build + download VILA model
```

This takes 15-30 minutes. It:
1. Installs ROS2 Humble apt packages (RTAB-Map, Nav2, usb_cam, image_proc)
2. Clones `ldlidar_stl_ros2` and `explore_lite` from GitHub
3. Sets serial port permissions for LiDAR
4. Installs Python packages (pyserial, pillow, opencv)
5. Builds all ROS2 packages with `colcon build`
6. Pre-downloads the VILA 2.7B model weights

---

## 6. Running the Robot

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

### Monitor (second terminal)

```bash
docker exec -it jetson_bot_vila bash
source /root/ros2_ws/source_all.bash
ros2 topic echo /ai/decision          # VILA decisions (JSON)
ros2 topic echo /ai/room              # inferred room name
ros2 topic echo /scan --once           # LiDAR scan (one shot)
ros2 topic echo /ultrasonic_range     # rear ultrasonic
ros2 topic echo /cmd_vel              # motor commands
```

### Keyboard Teleop (manual driving)

```bash
docker exec -it jetson_bot_vila bash
source /root/ros2_ws/source_all.bash
ros2 run teleop_twist_keyboard teleop_twist_keyboard
```

### Disable SLAM (AI + exploration only)

```bash
ros2 launch jetson_bot_slam ai_slam_explore.launch.py use_slam:=false
```

### Disable Room Hints

```bash
ros2 launch jetson_bot_slam ai_slam_explore.launch.py room_hints_enabled:=false
```

### View the Map

1. Copy the map database from Jetson: `scp pablo@<JETSON_IP>:~/ros2_robot/maps/bedroom.db ~/Desktop/`
2. Download [RTAB-Map](https://github.com/introlab/rtabmap/releases) for Windows
3. Open RTAB-Map → **File → Open Database** → select `bedroom.db`
4. View the 2D map, robot trajectory, and loop closures

### Docker Maintenance

```bash
# Update container image:
docker compose pull && docker compose up -d --remove-orphans && docker image prune -f

# Reclaim disk space:
docker system prune -f --filter 'until=168h'
```

---

## 7. Upgrading Files on the Jetson

When you edit files on your PC and need to update the Jetson:

### Option A: Git Push/Pull (recommended)

```bash
# On your PC:
git add -A && git commit -m "update" && git push

# On the Jetson host:
cd ~/ros2_robot && git pull
```

### Option B: SCP individual files

```bash
# From your PC (PowerShell):
scp .\src\robot_control\robot_control\exploration_controller.py pablo@<JETSON_IP>:~/ros2_robot/src/robot_control/robot_control/
scp .\src\jetson_bot_slam\jetson_bot_slam\vila_scene_labeller_node.py pablo@<JETSON_IP>:~/ros2_robot/src/jetson_bot_slam/jetson_bot_slam/
scp .\src\jetson_bot_slam\launch\ai_slam_explore.launch.py pablo@<JETSON_IP>:~/ros2_robot/src/jetson_bot_slam/launch/
scp .\src\jetson_bot_slam\urdf\jetson_bot.urdf.xacro pablo@<JETSON_IP>:~/ros2_robot/src/jetson_bot_slam/urdf/
scp .\docker-compose.yml pablo@<JETSON_IP>:~/ros2_robot/
scp .\setup_vila.sh pablo@<JETSON_IP>:~/ros2_robot/
```

### After Updating — Rebuild

```bash
# On the Jetson, inside the container:
docker exec -it jetson_bot_vila bash
source /opt/ros/humble/setup.bash
source /opt/ros/install/setup.bash
cd /root/ros2_ws
colcon build --symlink-install --packages-select ldlidar_stl_ros2 robot_control jetson_bot_slam
source install/setup.bash
```

> **Note**: If you only changed `.py` files and used `--symlink-install`, you may NOT need to rebuild — the symlinks point to the live source. But URDF/launch/config changes often need a rebuild.

### After Updating docker-compose.yml

```bash
# On the Jetson host (not inside container):
cd ~/ros2_robot
docker compose down
docker compose up -d
```

### After Updating setup_vila.sh (first-time deps changed)

```bash
docker exec -it jetson_bot_vila bash
bash /root/ros2_ws/setup_vila.sh
```

---

## 8. AI Navigation + Scene Labelling

### How VILA Works

1. **Trigger**: Every 5 obstacle-turns, `exploration_controller` publishes `True` on `/robot/movement_complete`
2. **Capture**: `vila_scene_labeller` grabs the latest camera frame from `/image_raw`
3. **Resize**: 640×480 → 384×384 for inference
4. **Inference**: VILA 2.7B analyses the image with this prompt:
   > "You are a navigation assistant... Choose forward/left/right/stop and label the scene"
5. **Parse**: Extract `DIRECTION`, `LABEL`, and `REASON` from the response
6. **Publish**: Direction → `/ai/direction`, label → `/ai/semantic_label`, room → `/ai/room`
7. **Log**: Everything written to `./logs/<session>/decisions_full.txt`

### VILA Response Format

```
DIRECTION: forward
LABEL: open floor near desk
REASON: clear path ahead with open space
```

### Memory & Speed

- GPU RAM: ~1.8–2 GB (4-bit AWQ quantization)
- Inference: ~0.5–1.5s per frame on Jetson Orin Nano
- Does NOT block the exploration controller — runs in a background thread

---

## 9. Room Identification

When `room_hints_enabled:=true` (default), the VILA node accumulates scene labels to infer which room the robot is in. **Zero extra GPU cost** — this is pure Python keyword matching on VILA's existing output.

### How It Works

1. VILA outputs a label like "near bed" → keyword "bed" matches `bedroom` category
2. Labels accumulate in a sliding window of 10
3. **1 match** → "bedroom area" (tentative)
4. **2+ matches from same category** → "bedroom" (confirmed)
5. **No matches for 5+ labels** → check LiDAR geometry:
   - Narrow sides + long axis → "hallway"
   - Open all around → "open area"

### Room Categories

| Room | Keywords |
|---|---|
| bedroom | bed, pillow, mattress, dresser, drawer, nightstand, wardrobe, closet |
| kitchen | stove, oven, sink, fridge, counter, cabinet, microwave |
| bathroom | toilet, shower, bathtub, sink, mirror, towel |
| living | couch, sofa, tv, coffee table, bookshelf, armchair |
| dining | dining table, chair, plate, candle |
| hallway | hallway, corridor, narrow, doorway, passage |
| office | desk, monitor, keyboard, mouse, computer |

### Hallway Detection (LiDAR Geometry)

The VILA node subscribes to `/scan` and checks:
- **Narrow**: average of (left zone min + right zone min) < 1.5m
- **Long**: front or rear max distance > 3× the narrow dimension
- If both conditions met AND no room objects detected → "hallway"

### Disable Room Hints

```bash
ros2 launch jetson_bot_slam ai_slam_explore.launch.py room_hints_enabled:=false
```

---

## 10. AI Decision Logging

Every session creates a timestamped log folder at `./logs/<YYYYMMDD_HHMMSS>/`:

```
logs/20260328_164500/
├── decisions_full.txt          # every AI decision with room + motor command
└── captures/
    ├── capture_001.jpg         # camera frame at decision #5
    ├── capture_001.json        # metadata (label, room, direction, reason)
    ├── capture_002.jpg         # camera frame at decision #10
    └── ...                     # up to 10 captures (1 per 5 decisions)
```

### `decisions_full.txt` format

```
2026-03-28 16:45:12.345 | #001 | dir=forward  | label="open floor" | room="bedroom" | reason="clear path ahead" | motor=(linear=0.200 angular=0.000) | infer=1.23s
2026-03-28 16:45:30.678 | #002 | dir=left     | label="near desk"  | room="bedroom" | reason="wall on right"   | motor=(linear=0.000 angular=0.550) | infer=0.89s
```

### Capture metadata (`capture_NNN.json`)

```json
{
  "timestamp": "2026-03-28 16:45:30.678",
  "decision_num": 5,
  "direction": "left",
  "label": "near desk",
  "room": "bedroom",
  "reason": "wall on right side",
  "motor_cmd": "linear=0.000 angular=0.550",
  "inference_s": 0.89,
  "image_file": "capture_001.jpg",
  "capture_num": 1
}
```

Logs are accessible from the Jetson host at `~/ros2_robot/logs/` (Docker volume mount with user 1000:1000 permissions).

---

## 11. Topics Reference

| Topic | Type | Publisher | Subscriber | Description |
|---|---|---|---|---|
| `/scan` | `LaserScan` | ldlidar | exploration_ctrl, rtabmap, vila | 360° LiDAR scan |
| `/image_raw` | `Image` | usb_cam | image_proc, vila | Camera frame (YUYV) |
| `/image_rect_color` | `Image` | image_proc | (available) | Converted RGB8 frame |
| `/odom` | `Odometry` | mecanum_odom | rtabmap | Wheel odometry |
| `/cmd_vel` | `Twist` | exploration_ctrl | motor_driver, vila (log) | Velocity commands |
| `/ultrasonic_range` | `Range` | arduino_bridge | exploration_ctrl | Rear ultrasonic (backup) |
| `/ai/direction` | `String` | vila | (logged) | AI navigation suggestion |
| `/ai/semantic_label` | `String` | vila | (map annotation) | Scene label |
| `/ai/room` | `String` | vila | (logged) | Inferred room name |
| `/ai/decision` | `String` | vila | (monitoring) | Full decision JSON |
| `/robot/movement_complete` | `Bool` | exploration_ctrl | vila | AI trigger |
| `/wheel_ticks` | `Int32MultiArray` | arduino_bridge | mecanum_odom | Encoder ticks |
| `/arduino_cmd` | `String` | motor_driver | arduino_bridge | Motor speed commands |
| `/map` | `OccupancyGrid` | rtabmap | (visualization) | 2D SLAM map |

---

## 12. Launch Arguments (All Flags)

Every argument can be overridden on the command line:

```bash
ros2 launch jetson_bot_slam ai_slam_explore.launch.py <arg>:=<value>
```

### Hardware Ports

| Argument | Default | Description |
|---|---|---|
| `serial_port` | `/dev/ttyUSB0` | Arduino Mega serial port |
| `lidar_port` | `/dev/ttyUSB1` | STL-27L LiDAR UART port |
| `camera_device` | `/dev/video0` | USB camera device |

### AI Model

| Argument | Default | Description |
|---|---|---|
| `vila_model` | `Efficient-Large-Model/VILA-2.7b` | VILA model name |
| `vila_api` | `awq` | Quantisation backend |
| `vila_quantization` | `q4f16_ft` | Quantisation precision |

### Exploration

| Argument | Default | Description |
|---|---|---|
| `move_speed` | `0.20` | Forward speed (m/s) |
| `turn_speed` | `0.55` | Turn speed (rad/s) |
| `obstacle_distance` | `0.30` | Start backing up at this range (m) |
| `emergency_stop_dist` | `0.08` | Emergency reverse below this range (m) |
| `rear_safety_dist` | `0.15` | Stop reversing if rear obstacle closer than this (m) |
| `backup_s` | `2.0` | Reverse duration before turning (seconds) |
| `min_turn_s` | `3.0` | Minimum turn time before checking if clear |
| `max_turn_s` | `10.0` | Switch turn direction after this (corner escape) |
| `label_every` | `5` | Trigger AI scene label every N turns |

### SLAM & Features

| Argument | Default | Description |
|---|---|---|
| `use_slam` | `true` | Enable RTAB-Map LiDAR SLAM |
| `room_hints_enabled` | `true` | Enable AI room identification from labels |
| `rviz` | `false` | Launch RViz2 (needs display) |

---

## 13. Tuning & Calibration

### Obstacle Distance

If the robot collides too often, increase `obstacle_distance`:
```bash
ros2 launch jetson_bot_slam ai_slam_explore.launch.py obstacle_distance:=0.50
```

### Turn Behavior

The exploration controller automatically picks the turn direction based on which side (left vs right LiDAR zone) has more open space. No manual left/right preference needed.

### Camera Calibration

`usb_cam` publishes a default `camera_info`. For better VILA accuracy, calibrate your camera:
```bash
ros2 run camera_calibration cameracalibrator --size 8x6 --square 0.025
```

### LiDAR Orientation

If the LiDAR scan is rotated (obstacles appear in wrong direction), check the `laser_scan_dir` parameter in the launch file. Set `True` for CCW positive (ROS convention).

---

## 14. File Structure & Descriptions

```
ros2_robot/
├── docker-compose.yml            # Container config: image, devices, volumes
├── setup_vila.sh                 # One-time setup: apt deps, LiDAR driver, build
├── README.md                     # This file
├── maps/                         # Persisted SLAM maps (bedroom.db)
├── logs/                         # AI decision logs + captured frames
│
├── src/
│   ├── jetson_bot_slam/          # SLAM + AI package
│   │   ├── jetson_bot_slam/
│   │   │   ├── arduino_bridge.py        # Serial comms with Arduino (sensors + motors)
│   │   │   ├── mecanum_odometry.py      # Wheel encoder → /odom + TF
│   │   │   ├── motor_driver.py          # /cmd_vel → individual wheel speeds
│   │   │   └── vila_scene_labeller_node.py ★  # VILA AI + room identification
│   │   ├── launch/
│   │   │   └── ai_slam_explore.launch.py ★    # Full stack launch (LiDAR+SLAM+AI)
│   │   ├── config/
│   │   │   └── vila_params.yaml         # VILA model parameters
│   │   ├── urdf/
│   │   │   └── jetson_bot.urdf.xacro ★  # Robot TF tree (sensor positions)
│   │   └── rviz/
│   │       └── slam_view.rviz           # RViz config for SLAM visualization
│   │
│   └── robot_control/            # Exploration + motor control package
│       └── robot_control/
│           └── exploration_controller.py ★  # 360° LiDAR obstacle avoidance
│
└── (external, cloned by setup_vila.sh)
    ├── ldlidar_stl_ros2/         # STL-27L LiDAR ROS2 driver
    └── m-explore-ros2/           # Frontier-based exploration (future use)
```

★ = key files modified for LiDAR + room identification integration

### File Descriptions

| File | Purpose |
|---|---|
| `docker-compose.yml` | Defines the Docker container: base image (`dustynv/nano_llm:humble`), GPU passthrough, USB device mounts (`/dev/ttyUSB0` Arduino, `/dev/ttyUSB1` LiDAR, `/dev/video0` camera), volume mounts for source code, maps, logs, and model cache |
| `setup_vila.sh` | One-time setup script. Installs ROS2 apt packages, clones LiDAR driver + explore_lite, installs Python deps, builds workspace with colcon, downloads VILA model |
| `exploration_controller.py` | Main driving logic. Subscribes to `/scan` (LiDAR) and splits into 4 zones (front ±60°, left 60-120°, right 240-300°, rear 120-240°). State machine: FORWARD → BACKING → TURNING. Picks turn direction based on which side has more space. Rear ultrasonic as backup during reverse |
| `vila_scene_labeller_node.py` | AI brain. Loads VILA 2.7B, triggered every 5 turns. Captures camera frame, runs inference, parses direction + label. Room identification: accumulates labels in sliding window, matches keywords to room types, uses LiDAR geometry for hallway detection. Logs all decisions + saves images |
| `ai_slam_explore.launch.py` | Launch file. Starts all 9 nodes: robot_state_publisher, arduino_bridge, mecanum_odometry, motor_driver, ldlidar, usb_cam, image_proc, rtabmap, vila_scene_labeller, exploration_controller. Configurable via launch arguments |
| `jetson_bot.urdf.xacro` | Robot physical description (URDF). Defines TF tree: base_link → wheels, camera, LiDAR (rear-right), ultrasonic (rear-centre). Required by RTAB-Map to correctly place sensor data on the map |
| `vila_params.yaml` | VILA model config: model name, quantization, prompt template, inference resolution |

---

## 15. Known Limitations & Next Steps

| Limitation | Mitigation |
|---|---|
| Camera outputs YUV (needs conversion) | `image_proc` converts to RGB8 automatically |
| No IMU → rotation relies on encoders | Mecanum wheels slip on carpet; future: add MPU6050 IMU |
| Loop closure needs texture-rich scenes | LiDAR SLAM doesn't need visual textures |
| Room hints are keyword-based | Works well for common rooms; add custom keywords to `ROOM_HINTS` for unusual rooms |
| Single rear ultrasonic | 360° LiDAR covers most directions; ultrasonic catches glass/mirrors |

**Planned upgrades:**
1. **5DOF arm** → front-left mounted claw arm for object interaction
2. **RealSense D435i** → depth camera + built-in IMU for 3D mapping
3. **IMU** (MPU6050) → fuse with wheel odom for better heading accuracy
4. **Semantic map overlay** → plot AI room labels onto the SLAM map grid
5. **Dynamic ROOM_HINTS** → learn new object→room associations from logs
