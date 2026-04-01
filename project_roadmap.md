# 🤖 VILA AI Semantic Mapping & Navigation Roadmap

## 🎯 The End Goal
To build a fully autonomous indoor robot that intelligently maps your apartment using 2D LiDAR (SLAM Toolbox) in the foreground while simultaneously capturing images in the background. These images are fed into the **VILA 2.7B LLM/VLM**, which classifies the semantic meaning of the area (e.g., "Bedroom", "Kitchen", "Hallway"). 

By aggregating 5-10 image classifications for specific spatial zones, the robot annotates the 2D map with semantic labels.
Ultimately, this allows you to drop the robot anywhere in the apartment and issue high-level semantic commands like **"Go to the Bedroom"**, prompting the Nav2 stack to plan a path from its current location to the centroid of the nearest annotated "Bedroom" region.

---

## 🛠️ Step-by-Step Implementation Plan

### Phase 1: Robust Autonomous Exploration (Currently In Progress)
*The foundation. The robot must be able to explore and build a clean map without human intervention.*
- [x] Integrate **STL-27L LiDAR** and establish proper URDF coordinate frames.
- [x] Configure mecanum wheel odometry (or laser odometry fallback).
- [x] **Replace RTAB-Map with SLAM Toolbox**: Optimize 2D SLAM for stability, preventing map tearing and odometry drift.
- [ ] **Frontier Exploration**: Run the `explore_lite` node to autonomously drive the robot toward unknown areas until the entire apartment is mapped.
- [ ] **Save the Map**: Once the apartment is fully mapped, save the 2D occupancy grid using the Nav2 map saver.

### Phase 2: AI Scene Labelling Integration
*Linking the physical space to semantic understanding using the camera and VILA.*
- [ ] **Enable Camera Stream**: Activate the `use_camera:=true` flag in the launch file to start publishing frames from the USB camera.
- [ ] **VILA Labeller Service**: Start the `vila_scene_labeller` node. Configure it to silently process a frame every *N* seconds while the robot explores.
- [ ] **Location Tagging**: Every time VILA returns a label (e.g., "Living Room"), tag the robot's current X, Y coordinate (from `slam_toolbox` TF) with that label.
- [ ] **Clustering & Consensus**: Accumulate these tagged coordinate points. Once a cluster of 5-10 points in a 2-meter radius all agree on a label, officially mark that zone on the map as "Living Room".

### Phase 3: Semantic "Go-To" Navigation
*Executing semantic commands based on the AI-annotated map.*
- [ ] **Semantic Waypoint Server**: Create a simple ROS 2 dictionary or JSON file that stores `{ "Bedroom": [x, y], "Kitchen": [x, y] }` based on the consensus from Phase 2.
- [ ] **Command Interface**: Build a ROS 2 service or a simple terminal trigger that accepts a string input (e.g., `target_room="Bedroom"`).
- [ ] **Dynamic Path Planning**: When the string is received, look up the `[x, y]` coordinates for that room, and send a standard Nav2 `NavigateToPose` action goal to drive the robot there.
- [ ] **Testing**: Drop the robot in an arbitrary room, ensure AMCL (localization) finds where it is, and tell it to go to another room.
