# Jetson Bot SLAM — Bedroom Explorer

Autonomous bedroom mapping and exploration using:
- **Mecanum-drive** robot with HiTechnic motor controllers (Arduino Mega)
- **USB camera** (index 1, Jetson USB port 1) → RTAB-Map visual loop closure
- **HC-SR04 ultrasonic sensor** → obstacle detection / Nav2 costmap
- **RTAB-Map** (monocular + wheel odometry) → SLAM
- **Nav2** + **explore_lite** → autonomous frontier exploration
- **ROS2 Iron** inside Docker (JetPack 6+, Ubuntu 22.04)

---

## Architecture

```
Arduino (serial)
  │ DATA pos1 pos2 pos3 pos4 dist_m
  ▼
arduino_bridge_node ──► /wheel_ticks  (Int32MultiArray [BL,BR,FL,FR])
                    ──► /ultrasonic_range (sensor_msgs/Range)
                    ◄── /arduino_cmd (String)  B-commands from motor_driver

mecanum_odometry_node ◄── /wheel_ticks
                      ──► /odom  (nav_msgs/Odometry)
                      ──► TF: odom → base_link

motor_driver_node ◄── /cmd_vel (geometry_msgs/Twist from Nav2 / teleop)
                  ──► /arduino_cmd

usb_cam ──► /image_raw, /camera_info

RTAB-Map ◄── /image_raw, /camera_info, /odom
         ──► /map  (nav_msgs/OccupancyGrid)
         ──► TF: map → odom  (loop closure corrections)

Nav2     ◄── /map, /odom, /ultrasonic_range
         ──► /cmd_vel

explore_lite ◄── /global_costmap/costmap
             ──► Nav2 goals  (autonomous frontier navigation)
```

---

## 1. One-Time Setup

> **You are using a Docker-based ROS2 Iron workflow.**
> Do **NOT** run `sudo apt install` directly on the Jetson host for these deps —
> they go into the **Dockerfile** and are installed when the image is rebuilt.

### 1.1 Copy the package into your existing Docker project

**Do not create a second `ros2_ws/` directory.**
Drop `jetson_bot_slam/` alongside your existing packages in the one `src/` folder.

On your Windows machine (from `1Nvidia Robot\` folder in PowerShell):
```powershell
# Replace <JETSON_IP> and <PROJECT_PATH> with your values
scp -r "ros2_ws\src\jetson_bot_slam" pablo@<JETSON_IP>:<PROJECT_PATH>/src/
```
Or if you use VSCode Remote SSH, copy the folder directly in the file explorer.

The result on the Jetson should look like:
```
your_project/
├── Dockerfile
├── entrypoint.sh
└── src/
    ├── your_existing_package/   ← already there
    └── jetson_bot_slam/         ← newly added (no overlap)
```

### 1.2 Update the Dockerfile

Add these lines to the existing `apt-get install` block in your Dockerfile:
```dockerfile
ARG ROS_DISTRO=iron
FROM ros:${ROS_DISTRO}-ros-base
ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    python3-pip python3-opencv ffmpeg libsm6 libxext6 \
    libatlas-base-dev libopenblas-dev libhdf5-dev \
    build-essential cmake git curl nano tmux \
    libusb-1.0-0-dev libjpeg-dev \
    ros-iron-vision-opencv \
    ros-iron-rtabmap-ros \
    ros-iron-navigation2 \
    ros-iron-nav2-bringup \
    ros-iron-usb-cam \
    ros-iron-xacro \
    ros-iron-robot-state-publisher \
    ros-iron-tf2-ros \
    ros-iron-tf2-geometry-msgs \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

RUN python3 -m pip install --no-cache-dir \
    "opencv-python-headless" \
    "numpy<2,>=1.26" \
    pyserial pillow apriltag requests
```

> **`explore_lite` note:** `ros-iron-explore-lite` may not be in the apt index.
> If install fails, clone it from source **in the Dockerfile** before `colcon build`:
> ```dockerfile
> RUN cd $ROS_WS/src && \
>     git clone --branch iron https://github.com/robo-friends/m-explore-ros2.git
> ```
> If the `iron` branch doesn't exist, use `ros2` (the default branch targets Humble+).

### 1.3 Rebuild the Docker image (on the Jetson)
```bash
cd ~/your_project
docker build -t jetson_bot:slam .
```

### 1.4 Build the ROS2 workspace (inside the container)
```bash
# Start container — pass serial and camera devices through
docker run -it --rm \
  --device=/dev/ttyUSB0 \
  --device=/dev/video0 \
  --device=/dev/video1 \
  --network=host \
  jetson_bot:slam bash

# Inside the container:
cd $ROS_WS
colcon build --symlink-install
source install/setup.bash
```

### 1.5 Verify Arduino serial port (on Jetson host, not inside Docker)
```bash
ls /dev/ttyUSB*
sudo usermod -aG dialout $USER   # Allow serial passthrough; re-login after
```

### 1.6 Verify camera (inside container)
```bash
ros2 run usb_cam usb_cam_node_exe --ros-args -p video_device:=/dev/video1
# In another tmux pane / docker exec:
ros2 topic hz /image_raw   # should read ~30 Hz
```

### 1.7 Calibrate the camera (recommended for better loop closure)
```bash
ros2 run camera_calibration cameracalibrator \
  --size 8x6 --square 0.025 \
  image:=/image_raw camera:=/camera
```

---

## 2. Running

All commands below are run **inside the Docker container** on the Jetson.

**Start the container** (do this once per session, then use `tmux` for multiple panes):
```bash
docker run -it --rm \
  --device=/dev/ttyUSB0 \
  --device=/dev/video0 \
  --device=/dev/video1 \
  --network=host \
  jetson_bot:slam bash

# Inside container — source the workspace
source $ROS_WS/install/setup.bash
```

### Option A: SLAM + manual driving (recommended first run)
```bash
# Pane 1 — Robot + SLAM (headless; Jetson has no monitor)
ros2 launch jetson_bot_slam slam.launch.py rviz:=false

# Pane 2 — Keyboard teleop
ros2 run teleop_twist_keyboard teleop_twist_keyboard cmd_vel:=/cmd_vel
```

Drive the robot around the room. The map builds in real time.

**Monitor from your laptop** (same local network, `--network=host` exposes all topics):
```bash
# On your laptop (needs ROS2 Iron):
rviz2 -d <path-to>/slam_view.rviz
```

**Save the map when done:**
```bash
ros2 run nav2_map_server map_saver_cli -f /root/bedroom_map
# bedroom_map.pgm + bedroom_map.yaml saved inside the container
# Copy to host:  docker cp <container_id>:/root/bedroom_map.pgm .
```

### Option B: Fully autonomous exploration
```bash
ros2 launch jetson_bot_slam explore.launch.py rviz:=false
```
The robot picks unexplored frontier cells and navigates to them automatically.

**Stop exploration:**
```bash
ros2 topic pub /explore/resume std_msgs/msg/Bool "data: false" --once
```

### Option C: If a monitor is attached to the Jetson
```bash
docker run -it --rm \
  --device=/dev/ttyUSB0 \
  --device=/dev/video0 --device=/dev/video1 \
  --network=host \
  -e DISPLAY=$DISPLAY \
  -v /tmp/.X11-unix:/tmp/.X11-unix \
  jetson_bot:slam bash

ros2 launch jetson_bot_slam explore.launch.py rviz:=true
```

---

## 3. Tuning Odometry

If the robot drifts (e.g. straight-forward curves left) after a few metres,
the `half_wheelbase` / `half_track_width` values need calibration.

**Test procedure:**
```bash
# Reset encoders
ros2 topic pub /arduino_cmd std_msgs/msg/String "data: 'RESET'" --once

# Echo odom
ros2 topic echo /odom --field pose.pose.position

# Drive the robot exactly 1 m forward via teleop
# Check /odom reports ~1.0 m in x
```

**Tune in `robot_bringup.launch.py`:**
```python
'half_wheelbase':   0.1270,   # increase if forward is under-reported
'half_track_width': 0.2172,   # increase if rotation is under-reported
```

For rotation calibration:
- Drive a 360° turn; odom should report Δθ ≈ 2π
- Adjust `half_track_width` until this is accurate

---

## 4. Topic Reference

| Topic | Type | Description |
|-------|------|-------------|
| `/wheel_ticks` | `Int32MultiArray` | Raw encoder counts [BL, BR, FL, FR] |
| `/odom` | `Odometry` | Mecanum wheel odometry |
| `/ultrasonic_range` | `Range` | HC-SR04 distance (m) |
| `/image_raw` | `Image` | USB camera feed |
| `/camera_info` | `CameraInfo` | Camera calibration |
| `/map` | `OccupancyGrid` | RTAB-Map 2-D occupancy grid |
| `/cmd_vel` | `Twist` | Velocity commands to robot |
| `/arduino_cmd` | `String` | Raw commands to Arduino |
| `/arduino_raw` | `String` | Raw lines from Arduino (debug) |
| `/explore/frontiers` | `MarkerArray` | Current frontier points |

---

## 5. Known Limitations & Next Steps

| Limitation | Mitigation |
|---|---|
| Single ultrasonic sensor (forward only) | Rotate robot frequently; add more sensors or a LiDAR |
| Monocular SLAM has scale ambiguity | Wheel odometry provides scale; camera calibration helps |
| No IMU → rotation odometry relies on encoders | Mecanum wheels slip on carpet; add an IMU (e.g. MPU6050) for heading |
| Loop closure requires texture-rich scene | Good for bedroom; struggles in featureless corridors |

**Next upgrades:**
1. **Add an IMU** → fuse with wheel odom in `robot_localization` EKF
2. **Add a LiDAR** (e.g. RPLIDAR A1, $99) → replace ultrasonic for much better mapping
3. **Replace RTAB-Map** with **Cartographer** once a LiDAR is available
4. **3-D map** → enable `Grid/3D: true` in RTAB-Map for point cloud output

---

## 6. File Structure

```
your_ros_project/            ← your existing Docker project on the Jetson
├── Dockerfile
├── entrypoint.sh
└── src/
    ├── your_existing_pkg/   ← untouched; no overlap
    └── jetson_bot_slam/     ← this package
        ├── package.xml
        ├── setup.py / setup.cfg
        ├── jetson_bot_slam/
        │   ├── arduino_bridge_node.py     — Serial ↔ ROS2 bridge
        │   ├── mecanum_odometry_node.py   — Wheel odometry + TF
        │   └── motor_driver_node.py       — cmd_vel → B-commands
        ├── launch/
        │   ├── robot_bringup.launch.py    — Sensors + drivers
        │   ├── slam.launch.py             — RTAB-Map SLAM
        │   └── explore.launch.py          — Nav2 + frontier exploration
        ├── config/
        │   └── nav2_params.yaml           — Nav2 / DWB tuning
        ├── urdf/
        │   └── jetson_bot.urdf.xacro      — Robot description / TF tree
        └── rviz/
            └── slam_view.rviz             — RViz2 config
```

On your Windows machine, the staging copy lives at:
```
1Nvidia Robot\ros2_ws\src\jetson_bot_slam\
```
Use `scp` or VSCode Remote SSH to sync it to the Jetson.
