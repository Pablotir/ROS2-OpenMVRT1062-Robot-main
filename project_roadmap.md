# 🤖 VILA AI Semantic Mapping & Navigation Roadmap

## 🎯 The End Goal
To build a fully autonomous indoor robot that intelligently maps your apartment using 2D LiDAR (SLAM Toolbox) in the foreground while simultaneously capturing images in the background. These images are fed into the **VILA 2.7B LLM/VLM**, which classifies the semantic meaning of the area (e.g., "Bedroom", "Kitchen", "Hallway"). 

By aggregating image classifications while the robot traces specific rooms, the robot annotates the 2D map with semantic labels.
Ultimately, this allows you to drop the robot anywhere in the apartment and issue high-level semantic commands like **"Go to the Bedroom"**, prompting the Nav2 stack to plan a path from its current location to the centroid of the nearest annotated "Bedroom" region.

---

## 🛠️ Step-by-Step Implementation Plan

### Phase 1: Robust Autonomous Exploration (State Machine)
*The foundation. Rebuilding the exploration math to be consistent, predictable, and structurally sound using a Finite State Machine.*
- [x] Integrate **STL-27L LiDAR** and establish proper URDF coordinate frames.
- [x] Configure mecanum wheel odometry (or laser odometry fallback).
- [x] **Replace RTAB-Map with SLAM Toolbox**: Optimize 2D SLAM for stability, preventing map tearing and odometry drift.
- [ ] **State Machine Controller**: Refactor `exploration_controller.py` into distinct states:
  - `STATE_HALLWAY`: Strict parallel alignment and centering using PD controllers.
  - `STATE_ROOM_PERIMETER`: Strictly follow a single wall at a set distance.
  - `STATE_CROSSING`: Drive across open spaces to find new frontiers.
- [ ] **Disable Strafing (Optional)**: Lock kinematics to traditional differential drive (straight + turn) for cleaner robotic movement if strafing causes slipping.

### Phase 2: Lifelong Mapping & Serialization (Solution C)
*Ensuring we can pause, save, and continuously build upon existing maps across multiple days without losing data.*
- [ ] **Pose Graph Saving**: Implement a script/trigger to save the `slam_toolbox` Pose Graph (Serialization) instead of just the 2D image.
- [ ] **Graceful Shutdown (E-Stop)**: Add a software kill-switch to freeze the wheels while keeping SLAM alive so the map can be serialized safely.
- [ ] **Map Deserialization**: Launch `slam_toolbox` in Lifelong Mapping mode to load the previous day's Pose Graph, find its current location, and continue expanding the map.

### Phase 3: AI Scene Labelling Integration
*Linking the physical space to semantic understanding using the camera and VILA.*
- [ ] **Enable Camera Stream**: Activate the `use_camera:=true` flag in the launch file to start publishing frames from the USB camera.
- [ ] **VILA Labeller Service**: Start the `vila_scene_labeller` node.
- [ ] **State-Triggered Inference**: Modify the exploration State Machine to trigger VILA *only* when the robot is in `STATE_ROOM_PERIMETER`. Suppress it in hallways to save GPU.
- [ ] **Location Tagging & Consensus**: As the robot traces the room perimeter, collect 4-5 labels (N, S, E, W). Once it leaves the room, run a consensus (e.g. 4 'Bedroom' tags) to officially mark that map zone as "Bedroom" and drop an [X,Y] pin.

### Phase 4: Semantic "Go-To" Navigation
*Executing semantic commands based on the AI-annotated map.*
- [ ] **Semantic Waypoint Server**: Create a ROS 2 dictionary or JSON file that stores `{ "Bedroom": [x, y], "Kitchen": [x, y] }` over time.
- [ ] **Command Interface**: Build a ROS 2 service or terminal trigger for semantic input (`target_room="Bedroom"`).
- [ ] **Dynamic Path Planning**: Look up the `[x, y]` coordinates, and send a standard Nav2 `NavigateToPose` action goal to drive there.
