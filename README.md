# Jetson Bot — AI-Guided SLAM Explorer

ROS2 Iron · Docker · Mecanum drive · RTAB-Map · Ollama VLM

## How it works

```
USB Camera ─► usb_camera_node ─► /image_raw ──────────────► RTAB-Map (builds 2D map)
                                ─► /camera/usb_raw ─► ai_navigator
                                                         │  sends frame to Ollama gemma3:4b
                                                         │  "can robot move forward? left or right?"
                                                         ▼
                                               /ai/direction
                                                         │
                                               exploration_controller
                                                         │  publishes /cmd_vel Twist
                                                         ▼
                                               motor_driver_node (mecanum IK)
                                                         ▼
                                               arduino_bridge_node ──► Arduino ──► Motors

Arduino ─► /wheel_ticks ─► mecanum_odometry ─► /odom ──► RTAB-Map (metric scale)
        ─► /ultrasonic_range (HC-SR04)
```

Two ROS2 packages work together in one workspace:
- **`robot_control`** — AI navigator (Ollama VLM), camera node, exploration controller
- **`jetson_bot_slam`** — Arduino bridge, mecanum odometry, motor driver, RTAB-Map launch

---

## Setup on the Jetson

### 1. Copy `jetson_bot_slam` into this project's `src/`

From your Windows machine:
```powershell
# From the "1Nvidia Robot" folder
scp -r "ros2_ws\src\jetson_bot_slam" pablo@<JETSON_IP>:~/ROS2-OpenMVRT1062-Robot-main/src/
```

After this, `src/` should contain **both** packages:
```
src/
├── robot_control/       ← existing
└── jetson_bot_slam/     ← newly added
```

### 2. Create the maps folder
```bash
mkdir -p maps
```

### 3. Pull the Ollama vision model (once)
```bash
docker compose --profile ollama up -d ollama
docker compose run --rm ollama ollama pull gemma3:4b
docker compose --profile ollama stop   # stop after pull
```

### 4. Build the Docker image
```bash
docker compose build
```
This installs RTAB-Map, Nav2, tf2, cv_bridge, builds the workspace, and clones explore_lite.

---

## Running

### Start Ollama (VLM server)
```bash
docker compose --profile ollama up -d ollama
```

### Start the robot (in a new terminal)
```bash
docker compose run --rm robot bash
```

### Inside the container — launch everything
```bash
ros2 launch robot_control slam_ai_explore.launch.py
```

The robot will:
1. Start building a map with RTAB-Map
2. Ask Ollama to look at the camera and decide where to go
3. Move in that direction for 3 seconds
4. Repeat — each loop the map grows

**Save the map when done:**
```bash
ros2 run nav2_map_server map_saver_cli -f /root/maps/bedroom_map
# Maps persist to ./maps/ on the Jetson host via the volume mount
```

### Optional: keyboard teleop while mapping
```bash
# In a second docker exec session:
docker exec -it <container_id> bash
ros2 run teleop_twist_keyboard teleop_twist_keyboard
```

---

## Launch arguments

| Argument | Default | Description |
|---|---|---|
| `serial_port` | `/dev/ttyUSB0` | Arduino serial port |
| `camera_device` | `/dev/video1` | USB camera (try video0 or video1) |
| `ollama_host` | `http://localhost:8080` | Ollama server URL |
| `ollama_model` | `gemma3:4b` | Vision-capable model |
| `slam_goal` | `explore the bedroom...` | Natural-language goal for the VLM |
| `move_duration` | `3.0` | Seconds per exploration step |
| `move_speed` | `0.25` | Forward speed in m/s |

Example:
```bash
ros2 launch robot_control slam_ai_explore.launch.py \
  camera_device:=/dev/video0 \
  move_speed:=0.20 \
  slam_goal:="map the bedroom completely"
```

---

## What changed from the original

| File | Change |
|---|---|
| `exploration_controller.py` | Now publishes `geometry_msgs/Twist` on `/cmd_vel` instead of incorrect raw motor strings — fixes mecanum sign errors |
| `usb_camera_node.py` | Added continuous 5 fps `/image_raw` + `/camera_info` stream for RTAB-Map alongside the existing on-demand AI stream |
| `package.xml` | Added proper ROS2 message dependencies |
| `ros2.Dockerfile` | Added RTAB-Map, Nav2, tf2, cv_bridge; added workspace `colcon build`; clones explore_lite from source |
| `docker-compose.yml` | Added video1 device, maps volume, Ollama service (optional profile) |
| `launch/slam_ai_explore.launch.py` | **New** — integrated launch that starts all nodes from both packages |
