# 🤖 VILA AI Semantic Mapping & Navigation Roadmap

## 🎯 The End Goal
To build a fully autonomous indoor robot that intelligently maps your apartment using 2D LiDAR (SLAM Toolbox) in the foreground while simultaneously capturing images in the background. These images are fed into the **VILA 2.7B LLM/VLM**, which classifies the semantic meaning of the area (e.g., "Bedroom", "Kitchen", "Hallway"). 

By aggregating image classifications while the robot traces specific rooms, the robot annotates the 2D map with semantic labels.
Ultimately, this allows you to drop the robot anywhere in the apartment and issue high-level semantic commands like **"Go to the Bedroom"**, prompting the Nav2 stack to plan a path from its current location to the centroid of the nearest annotated "Bedroom" region.

The final evolution of this robot includes a **SO-ARM101 robotic arm** for pick-and-place tasks, triggered by semantic navigation — the robot finds a target room, navigates to it, and uses the arm to interact with objects in the environment.

---

## 🛠️ Step-by-Step Implementation Plan

---

### ✅ Phase 1: Robust Autonomous Exploration (State Machine)
*COMPLETED — Pending hardware retest with new RoboClaw motor controllers.*

> **Hardware Upgrade Note:** The Arduino Mega + HiTechnic motor controllers have been retired and replaced with **dual RoboClaw 2x15A** motor controllers connected via USB directly to the Jetson. The Arduino udev mapping has been removed from `setup_udev_rules.sh`. All old HiTechnic/Arduino ROS2 code is pending deletion once RoboClaw integration is confirmed working.

- [x] Integrate **STL-27L LiDAR** and establish proper URDF coordinate frames.
- [x] Configure mecanum wheel odometry (or laser odometry fallback).
- [x] **Replace RTAB-Map with SLAM Toolbox**: Optimize 2D SLAM for stability, preventing map tearing and odometry drift.
- [x] **State Machine Controller**: Refactor `exploration_controller.py` into distinct states:
  - `STATE_HALLWAY`: Strict parallel alignment and centering using PD controllers.
  - `STATE_ROOM_PERIMETER`: Strictly follow a single wall at a set distance.
  - `STATE_CROSSING`: Drive across open spaces to find new frontiers.
- [x] **Disable Strafing (Optional)**: Lock kinematics to traditional differential drive (straight + turn) for cleaner robotic movement if strafing causes slipping.
- [ ] 🔁 **RETEST**: Re-validate full autonomous exploration with dual RoboClaw controllers replacing the old Arduino-based drive system.

---

### 🔧 Phase 1.5: RoboClaw Hardware Integration (Current Work)
*Migrating motor control from Arduino/HiTechnic to dual RoboClaw 2x15A.*

- [x] Source and configure dual **RoboClaw 2x15A** motor controllers.
- [x] Wire Tetrix TorqueNADO motors + encoders to RoboClaws (Quadrature mode).
- [x] Run **Velocity PID Autotune** on all 4 motors (2 per controller).
- [x] Configure both controllers in BasicMicro Motion Studio:
  - Packet Serial mode, Address 128, Baudrate 115200
  - Quadrature encoder mode
  - Multi-Unit Mode: **OFF** (separate USB connections, not daisy-chained)
  - Min Main Battery: **9.0V** (GoBilda 12V NiMH protection)
  - Save all settings to EEPROM
- [ ] Obtain USB serial numbers from Jetson and fill in `setup_udev_rules.sh` for `roboclaw_left` and `roboclaw_right`.
- [ ] Run `setup_udev_rules.sh` on Jetson host to apply new udev symlinks.
- [ ] Write RoboClaw ROS2 driver node (closed-loop velocity control via `SpeedM1M2`).
- [ ] Delete all legacy **HiTechnic** and **Arduino** motor controller code from the ROS2 workspace.
- [ ] Validate closed-loop wheel control and odometry accuracy.

#### 📌 USB Power Budget (Jetson)
| Device | Est. Draw | Notes |
|---|---|---|
| STL-27L LiDAR | ~200mA | Data only via USB |
| RoboClaw Left | ~100mA | Power from battery; USB is data only |
| RoboClaw Right | ~100mA | Power from battery; USB is data only |
| SO-ARM101 Controller | ~100mA | Power from arm LiPo; USB is data only |
| **Total** | **~500mA** | ✅ Well within limits — no powered hub needed |

> **Previous USB failure** (LiDAR + Arduino + USB camera + keyboard dongle) was likely caused by the camera drawing 400–500mA alone. The new setup is data-only USB for all motor controllers — significantly lighter load.

#### 📌 Common Ground Wiring (Multi-Battery Setup)
All power systems (Jetson, RoboClaws/NiMH, SO-ARM101 LiPo) must share a **common ground** to prevent floating ground issues.

| System | Ground Connection Point |
|---|---|
| Jetson Orin/Nano | Any GND pin on the **40-pin GPIO header** (Pins 6, 9, 14, 20, 25, 30, 34, or 39) |
| RoboClaw GND | Battery negative terminal (already tied to motor GND internally) |
| SO-ARM101 LiPo GND | Arm controller GND terminal |

> **Do NOT connect grounds through USB cables** — always use a dedicated wire to a GPIO GND pin on the Jetson header.

---

### Phase 2: Lifelong Mapping & Serialization (Solution C)
*Ensuring we can pause, save, and continuously build upon existing maps across multiple days without losing data.*
- [ ] **Pose Graph Saving**: Implement a script/trigger to save the `slam_toolbox` Pose Graph (Serialization) instead of just the 2D image.
- [ ] **Graceful Shutdown (E-Stop)**: Add a software kill-switch to freeze the wheels while keeping SLAM alive so the map can be serialized safely.
- [ ] **Map Deserialization**: Launch `slam_toolbox` in Lifelong Mapping mode to load the previous day's Pose Graph, find its current location, and continue expanding the map.

---

### Phase 3: AI Scene Labelling Integration
*Linking the physical space to semantic understanding using the camera and VILA.*
- [ ] **Enable Camera Stream**: Activate the `use_camera:=true` flag in the launch file to start publishing frames from the USB camera.
- [ ] **VILA Labeller Service**: Start the `vila_scene_labeller` node.
- [ ] **State-Triggered Inference**: Modify the exploration State Machine to trigger VILA *only* when the robot is in `STATE_ROOM_PERIMETER`. Suppress it in hallways to save GPU.
- [ ] **Location Tagging & Consensus**: As the robot traces the room perimeter, collect 4-5 labels (N, S, E, W). Once it leaves the room, run a consensus (e.g. 4 'Bedroom' tags) to officially mark that map zone as "Bedroom" and drop an [X,Y] pin.

---

### Phase 4: Semantic "Go-To" Navigation
*Executing semantic commands based on the AI-annotated map.*
- [ ] **Semantic Waypoint Server**: Create a ROS 2 dictionary or JSON file that stores `{ "Bedroom": [x, y], "Kitchen": [x, y] }` over time.
- [ ] **Command Interface**: Build a ROS 2 service or terminal trigger for semantic input (`target_room="Bedroom"`).
- [ ] **Dynamic Path Planning**: Look up the `[x, y]` coordinates, and send a standard Nav2 `NavigateToPose` action goal to drive there.

---

### Phase 5: SO-ARM101 Pick & Place Integration
*Extending the robot's capability from pure navigation to physical object interaction.*

**Hardware:**
- SO-ARM101 6-DOF robotic arm (currently under construction)
- Dedicated LiPo battery pack powering the arm independently
- Servo motor controller connected to Jetson via USB (`/dev/arm_controller`)
- Common ground shared with Jetson via 40-pin GPIO header GND pin

**Software Goals:**
- [ ] Obtain and fill in `SERIAL_ARM` in `setup_udev_rules.sh` once arm controller is connected.
- [ ] Write ROS2 arm controller node interfacing with `/dev/arm_controller`.
- [ ] Define a standard set of **pick** and **place** action primitives (pre-grasp, grasp, lift, release).
- [ ] Integrate arm actions with semantic navigation: robot navigates to target room, then triggers arm action.
- [ ] **Object detection integration**: Use USB camera + VILA or a lightweight detection model to identify graspable objects and compute target grasp poses.
- [ ] Implement safety constraints (joint limits, collision checking, workspace boundaries).
