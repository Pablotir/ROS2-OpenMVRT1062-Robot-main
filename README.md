# Jetson Bot — AI-Guided SLAM Explorer & Manipulator

**Autonomous room mapping robot** powered by an NVIDIA Jetson, mecanum-drive chassis, STL-27L 360° LiDAR, Intel RealSense D405, and on-device vision AI. The robot explores rooms, builds a real-time 2D map with SLAM Toolbox, uses a Vision-Language Model (VILA 2.7B) to classify and semantically annotate rooms, and leverages a SO-ARM101 robotic arm to interact with objects—all running locally on the edge, no cloud required.

---

## Table of Contents

1. [Architecture](#1-architecture)
2. [Tech Stack & Definitions](#2-tech-stack--definitions)
3. [Hardware](#3-hardware)
4. [Docker Setup](#4-docker-setup)
5. [Running the Robot](#5-running-the-robot)
6. [Exploration & Semantic Layers](#6-exploration--semantic-layers)
7. [Arm Movements & Object Grabbing](#7-arm-movements--object-grabbing)
8. [Topics Reference](#8-topics-reference)
9. [File Structure & Descriptions](#9-file-structure--descriptions)

---

## 1. Architecture

The system has five layers: **perception** (LiDAR + RealSense D405 + VILA AI), **localisation** (SLAM Toolbox), **intelligence** (semantic room identification), **actuation** (RoboClaw motor controllers), and **manipulation** (SO-ARM101). All connected through ROS2 topics inside a single Docker container.

```text
┌──────────────────────────────────────────────────────────────────────┐
│  dustynv/nano_llm:humble  Docker Container                          │
│                                                                      │
│  STL-27L LiDAR (360°)                                               │
│    └──► ldlidar_stl_ros2 ──► /scan ──┬──► SLAM Toolbox ──► /map      │
│                                      │    (2D SLAM)                  │
│                                      │                               │
│                                      └──► exploration_controller    │
│                                           (State Machine)            │
│                                                                      │
│  Intel RealSense D405                                                │
│    ├──► RGB ──┬──► vila_scene_labeller (Semantic Layer/Room ID)      │
│    │          │                                                      │
│    │          └──► object_detection (Grasp pose computation)         │
│    └──► Depth ──► arm_controller (3D object localization)            │
│                                                                      │
│  Dual RoboClaw 2x15A (USB)                                           │
│    └──► roboclaw_ros                                                 │
│         ├─► /odom (from Quadrature Encoders)                         │
│         └──◄ /cmd_vel                                                │
│                                                                      │
│  SO-ARM101 Robotic Arm (USB)                                         │
│    └──► arm_controller ◄── Pick & Place Commands                     │
└──────────────────────────────────────────────────────────────────────┘
```

---

## 2. Tech Stack & Definitions

| Component | Technology | Purpose |
|---|---|---|
| **Robot OS** | ROS2 Humble | Message passing between nodes (pub/sub topics) |
| **Container** | Docker + `dustynv/nano_llm:humble` | Reproducible environment with CUDA/TensorRT |
| **AI Model** | VILA 2.7B (4-bit AWQ via nano_llm) | On-device vision-language model for semantic room labels |
| **LiDAR** | LDROBOT STL-27L | Primary obstacle detection + SLAM |
| **SLAM** | SLAM Toolbox | Builds stable 2D occupancy grid & pose graph |
| **Camera** | Intel RealSense D405 | RGB + Depth for VILA, obstacle detection, and arm grasping |
| **Motor Drivers**| Dual RoboClaw 2x15A | Precise closed-loop velocity control via USB |
| **Arm** | SO-ARM101 6-DOF | Pick-and-place manipulation |
| **Drive** | 4× mecanum wheels | Omnidirectional/Differential movement |
| **Compute** | NVIDIA Jetson Orin Nano | GPU inference + ROS2 processing |

---

## 3. Hardware

| Part | Model | Connection |
|---|---|---|
| Compute | NVIDIA Jetson Orin Nano | — |
| LiDAR | LDROBOT STL-27L | USB (`/dev/ttyUSB_lidar`) |
| Camera | Intel RealSense D405 | USB 3.0 |
| Motor Drivers | 2x RoboClaw 2x15A | USB (`/dev/roboclaw_left`, `...right`) |
| Motors | 4× DC motors + encoders | Wired to RoboClaws (Quadrature mode) |
| Arm | SO-ARM101 6-DOF | USB (`/dev/arm_controller`) |

### USB Power Budget (Jetson)
The system leverages a shared-ground, multi-battery setup. The USB connections from the Jetson to the RoboClaws and Arm controller are **data-only**.
- STL-27L LiDAR: ~200mA
- RealSense D405: Powered via USB 3.0
- RoboClaws & Arm: Powered by separate NiMH/LiPo batteries, drawing minimal USB current (~100mA each).

---

## 4. Docker Setup

### Prerequisites
- JetPack 6 installed on Jetson
- Docker + NVIDIA Container Runtime configured
- Clone this repo to `~/ros2_robot/`

### Setup & Build
```bash
cd ~/ros2_robot
docker compose up -d
docker exec -it jetson_bot_vila bash
bash /root/ros2_ws/setup_vila.sh
```

---

## 5. Running the Robot

### Start the Stack
```bash
cd ~/ros2_robot
docker compose up -d
docker exec -it jetson_bot_vila bash
source /opt/ros/humble/setup.bash
source /root/ros2_ws/install/setup.bash
ros2 launch jetson_bot_slam ai_slam_explore.launch.py
```

---

## 6. Exploration & Semantic Layers

### State Machine Controller
The exploration process relies on a robust state machine (`exploration_controller.py`):
- **`STATE_HALLWAY`**: Strict parallel alignment and centering using PD controllers.
- **`STATE_ROOM_PERIMETER`**: Strictly follow a single wall at a set distance to map the room outline.
- **`STATE_CROSSING`**: Drive across open spaces to find new frontiers.

### Semantic Mapping (VILA AI)
While the robot traces specific rooms (during `STATE_ROOM_PERIMETER`), the RealSense D405 captures images.
1. The frames are fed to **VILA 2.7B**, which classifies the semantic meaning of the area (e.g., "Bedroom", "Kitchen").
2. The system aggregates these labels (e.g., consensus of 4 "Bedroom" tags).
3. Once consensus is reached, the 2D map generated by SLAM Toolbox is annotated with a semantic label and an `[X, Y]` pin.
4. This enables high-level commands like **"Go to the Bedroom"**, using Nav2 to route the robot directly to the annotated region.

---

## 7. Arm Movements & Object Grabbing

The **SO-ARM101 6-DOF robotic arm** extends the robot's capabilities to physical interaction.
- **Semantic Navigation Integration**: The robot first uses semantic maps to navigate to the target room.
- **Perception with RealSense D405**: Once in the room, the D405’s RGB-D data is used alongside an object detection model to locate graspable items and compute their 3D target grasp poses.
- **Action Primitives**: The arm uses predefined primitives (pre-grasp, grasp, lift, release) to manipulate the object, safely avoiding joint limits and collisions.

---

## 8. Topics Reference

| Topic | Type | Publisher | Subscriber | Description |
|---|---|---|---|---|
| `/scan` | `LaserScan` | ldlidar | exploration_ctrl, slam_toolbox | 360° LiDAR scan |
| `/camera/color/image_raw` | `Image` | realsense2_camera | vila, arm_vision | D405 RGB frames |
| `/camera/depth/image_rect_raw`| `Image` | realsense2_camera | arm_vision | D405 Depth frames |
| `/odom` | `Odometry` | roboclaw_ros | slam_toolbox, nav2 | Wheel odometry |
| `/cmd_vel` | `Twist` | exploration_ctrl, nav2| roboclaw_ros | Velocity commands |
| `/ai/semantic_label` | `String` | vila | map_annotator | Scene label |
| `/ai/room` | `String` | vila | map_annotator | Inferred room name |
| `/map` | `OccupancyGrid` | slam_toolbox | nav2 | 2D SLAM map |

---

## 9. File Structure & Descriptions

```text
ros2_robot/
├── docker-compose.yml            
├── setup_udev_rules.sh           # Udev rules for LiDAR, RoboClaws, Arm
├── README.md                     
├── src/
│   ├── jetson_bot_slam/          
│   │   ├── vila_scene_labeller_node.py  # VILA AI semantic labeling & room consensus
│   │   ├── launch/ai_slam_explore.launch.py 
│   │   └── urdf/jetson_bot.urdf.xacro   
│   ├── robot_control/            
│   │   ├── exploration_controller.py    # State machine (Hallway, Perimeter, Crossing)
│   │   ├── roboclaw_driver.py           # Closed-loop velocity control for RoboClaws
│   │   └── arm_controller.py            # SO-ARM101 Pick & Place logic
```
