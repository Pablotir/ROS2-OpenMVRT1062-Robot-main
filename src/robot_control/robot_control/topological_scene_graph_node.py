#!/usr/bin/env python3
"""
topological_scene_graph_node.py — Layer 1 Topological Scene Graph ROS 2 Node
=============================================================================
Implements Layer 1 of the Modular OVMM Master Architecture:
  1. Partitions 2D /map occupancy grids into topological room regions & doorways.
  2. Tracks robot pose across region boundaries using TF2.
  3. Triggers ON-DEMAND snapshots from the Dell front camera upon region crossing.
  4. Classifies semantic room labels from detected landmarks.
  5. Persists regions, centroids, bounding polygons, landmarks, and doorways to SQLite.
  6. Publishes RViz MarkerArrays visualizing room boundaries, centroids, and labels.

Subscribes:
  /map                    nav_msgs/OccupancyGrid
  /camera/usb_raw         sensor_msgs/Image (or /image_raw)

Publishes:
  /scene_graph/markers    visualization_msgs/MarkerArray (3D room polygons, labels, doorways)
  /scene_graph/current_room std_msgs/String
  /camera/request_frame   std_msgs/Bool (on-demand snapshot trigger)

Services:
  /scene_graph/trigger_snapshot  std_srvs/Trigger
  /scene_graph/get_scene_summary std_srvs/Trigger
"""

import os
import cv2
import json
import time
import math
import numpy as np
from typing import Optional, List, Dict, Any, Tuple

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy
from nav_msgs.msg import OccupancyGrid
from sensor_msgs.msg import Image
from std_msgs.msg import String, Bool
from geometry_msgs.msg import Point, PoseStamped
from visualization_msgs.msg import Marker, MarkerArray
from std_srvs.srv import Trigger
from cv_bridge import CvBridge

import tf2_ros
from tf2_ros import TransformException

from .scene_graph_db import SceneGraphDB
from .topological_partitioner import TopologicalPartitioner, TopologicalRegionData
from .map_manager import get_maps_base_dir, get_latest_map_session


# Semantic room landmark dictionaries for zero-shot tagging
ROOM_LANDMARK_TAXONOMY = {
    'kitchen': ['refrigerator', 'fridge', 'sink', 'microwave', 'oven', 'toaster', 'dining table', 'kettle', 'bottle', 'cup', 'bowl'],
    'bedroom': ['bed', 'nightstand', 'wardrobe', 'clock', 'pillow', 'blanket', 'dresser'],
    'living_room': ['couch', 'sofa', 'tv', 'television', 'chair', 'coffee table', 'potted plant', 'remote', 'bookshelf'],
    'bathroom': ['toilet', 'sink', 'bathtub', 'shower', 'towel', 'mirror'],
    'office': ['desk', 'laptop', 'computer', 'monitor', 'keyboard', 'office chair', 'printer', 'book'],
}


class TopologicalSceneGraphNode(Node):
    """ROS 2 Node for Layer 1 Topological Scene Graphing & On-Demand Tagging."""

    def __init__(self):
        super().__init__('topological_scene_graph_node')

        # ── Parameters ────────────────────────────────────────────────────────
        self.declare_parameter('maps_dir', '/root/maps')
        self.declare_parameter('partition_interval_s', 6.0)     # re-partition grid every 6s
        self.declare_parameter('min_room_area_sqm', 1.5)
        self.declare_parameter('max_door_width_m', 1.4)
        self.declare_parameter('camera_topic', '/camera/usb_raw') # on-demand Dell camera stream

        self._maps_dir = get_maps_base_dir(self.get_parameter('maps_dir').value)
        self._partition_interval = float(self.get_parameter('partition_interval_s').value)
        self._min_room_area = float(self.get_parameter('min_room_area_sqm').value)
        self._max_door_width = float(self.get_parameter('max_door_width_m').value)
        camera_topic = self.get_parameter('camera_topic').value

        # SQLite Database Path
        self._db_path = os.path.join(self._maps_dir, 'scene_graph.db')
        self._db = SceneGraphDB(self._db_path)
        self.get_logger().info(f"Topological Scene Graph initialized | DB: {self._db_path}")

        # Snapshot storage directory
        self._snapshots_dir = os.path.join(self._maps_dir, 'snapshots')
        os.makedirs(self._snapshots_dir, exist_ok=True)

        # Partitioner engine
        self._partitioner = TopologicalPartitioner(
            min_room_area_sqm=self._min_room_area,
            max_door_width_m=self._max_door_width
        )

        self._bridge = CvBridge()

        # State variables
        self._latest_map_msg: Optional[OccupancyGrid] = None
        self._last_partition_t = 0.0
        self._current_region_id: int = -1
        self._tagged_regions: set = set()
        self._pending_snapshot_region: Optional[int] = None
        self._last_snapshot_t = 0.0

        # TF2 Listener for tracking robot pose in map frame
        self._tf_buffer = tf2_ros.Buffer()
        self._tf_listener = tf2_ros.TransformListener(self._tf_buffer, self)

        # ── Publishers ────────────────────────────────────────────────────────
        self._markers_pub = self.create_publisher(MarkerArray, '/scene_graph/markers', 10)
        self._room_pub = self.create_publisher(String, '/scene_graph/current_room', 10)
        self._req_frame_pub = self.create_publisher(Bool, '/camera/request_frame', 10)

        # ── Subscribers ───────────────────────────────────────────────────────
        map_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST, depth=1)

        self.create_subscription(OccupancyGrid, '/map', self._on_map, map_qos)
        self.create_subscription(Image, camera_topic, self._on_camera_frame, 10)
        # Fallback subscriber to continuous /image_raw if available
        self.create_subscription(Image, '/image_raw', self._on_camera_frame, 10)

        # ── Services ──────────────────────────────────────────────────────────
        self.create_service(Trigger, '/scene_graph/trigger_snapshot', self._srv_trigger_snapshot)
        self.create_service(Trigger, '/scene_graph/get_scene_summary', self._srv_get_scene_summary)

        # ── Control Loop Timers ────────────────────────────────────────────────
        # 2 Hz pose tracking & region crossing supervisor loop
        self.create_timer(0.5, self._tracking_loop)
        # 1 Hz marker visualizer loop
        self.create_timer(1.0, self._publish_rviz_markers)

    # ── Map Subscriber & Topological Partitioning ───────────────────────────

    def _on_map(self, msg: OccupancyGrid):
        """Cache map message and run topological partitioning when interval elapsed."""
        self._latest_map_msg = msg
        now = time.monotonic()
        if (now - self._last_partition_t) >= self._partition_interval:
            self._partition_map()

    def _partition_map(self):
        """Execute Voronoi distance transform & watershed partitioning on current map."""
        if self._latest_map_msg is None:
            return

        self._last_partition_t = time.monotonic()
        try:
            regions, chokepoints, _ = self._partitioner.partition_occupancy_grid(self._latest_map_msg)
            if not regions:
                return

            # Upsert detected topological regions into SQLite
            for r in regions:
                label = 'hallway' if r.is_hallway else 'unlabeled'
                confidence = 0.5 if r.is_hallway else 0.0
                self._db.upsert_region(
                    region_id=r.region_id,
                    centroid_x=r.centroid_x,
                    centroid_y=r.centroid_y,
                    polygon_coords=r.polygon_coords,
                    label=label,
                    area_sqm=r.area_sqm,
                    confidence=confidence
                )

            # Record detected doorways/chokepoints
            for cp in chokepoints:
                self._db.add_chokepoint(x=cp['x'], y=cp['y'], width_m=cp['width_m'])

            self.get_logger().info(
                f"Topological partitioning: {len(regions)} regions, {len(chokepoints)} doorways -> Saved to SQLite")

        except Exception as e:
            self.get_logger().warn(f"Partitioning error: {e}")

    # ── Robot Tracking & Region Boundary Supervisor ─────────────────────────

    def _tracking_loop(self):
        """Look up robot pose in map frame and check for topological boundary crossings."""
        try:
            t = self._tf_buffer.lookup_transform('map', 'base_link', rclpy.time.Time())
            rx = t.transform.translation.x
            ry = t.transform.translation.y
        except TransformException:
            return

        # Find which region currently contains the robot
        current_region = self._db.get_region_for_pose(rx, ry)

        if current_region is None:
            return

        r_id = current_region['region_id']
        r_label = current_region['label']

        # Publish current room name topic
        room_msg = String()
        room_msg.data = r_label
        self._room_pub.publish(room_msg)

        # Detect transition into a new region
        if r_id != self._current_region_id:
            old_id = self._current_region_id
            self._current_region_id = r_id
            self.get_logger().info(
                f"Topological Boundary Crossed: Region {old_id} -> Region {r_id} ({r_label})")

            # If this region has not yet been tagged with a camera snapshot, trigger one!
            if r_id not in self._tagged_regions and (r_label == 'unlabeled' or current_region['confidence'] < 0.6):
                self._trigger_on_demand_snapshot(r_id)

    def _trigger_on_demand_snapshot(self, region_id: int):
        """Wake up the Dell front camera for a single RGB keyframe snapshot."""
        now = time.monotonic()
        if (now - self._last_snapshot_t) < 3.0:
            return  # Throttle snapshots

        self._last_snapshot_t = now
        self._pending_snapshot_region = region_id

        # Send request signal to on-demand camera node
        req = Bool()
        req.data = True
        self._req_frame_pub.publish(req)
        self.get_logger().info(f"Dell Camera: On-demand snapshot requested for Region {region_id}")

    # ── Camera Snapshot & Semantic Landmark Classifier ──────────────────────

    def _on_camera_frame(self, msg: Image):
        """Process incoming camera snapshot and semantically tag the pending region."""
        if self._pending_snapshot_region is None:
            return

        target_region = self._pending_snapshot_region
        self._pending_snapshot_region = None

        try:
            cv_img = self._bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().warn(f"Failed to convert snapshot image: {e}")
            return

        # Save snapshot JPEG to disk
        snapshot_filename = f"snapshot_region_{target_region}_{int(time.time())}.jpg"
        snapshot_path = os.path.join(self._snapshots_dir, snapshot_filename)
        cv2.imwrite(snapshot_path, cv_img)

        # Execute Landmark Recognition & Room Classification
        label, confidence, landmarks = self._classify_room_from_snapshot(cv_img)

        # Store landmarks in SQLite
        for lm in landmarks:
            self._db.add_landmark(
                region_id=target_region,
                class_name=lm['class'],
                confidence=lm['conf'],
                bbox=lm['bbox'],
                snapshot_path=snapshot_path
            )

        # Update region label in SQLite database
        self._db.update_region_label(target_region, label=label, confidence=confidence)
        self._tagged_regions.add(target_region)

        self.get_logger().info(
            f"Region {target_region} Semantically Tagged -> '{label}' (confidence: {confidence:.2f}) | "
            f"Landmarks: {[lm['class'] for lm in landmarks]}")

    def _classify_room_from_snapshot(self, img: np.ndarray) -> Tuple[str, float, List[Dict[str, Any]]]:
        """
        Classifies room type from image landmarks.
        Uses object color/feature heuristics and room taxonomy matching.
        """
        landmarks: List[Dict[str, Any]] = []
        room_scores: Dict[str, float] = {k: 0.0 for k in ROOM_LANDMARK_TAXONOMY.keys()}

        # Image color analysis heuristics
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        h, w = img.shape[:2]

        # Detect dominant color regions and objects
        # Silver / White appliance detector (Refrigerator / Oven in Kitchen)
        white_mask = cv2.inRange(hsv, np.array([0, 0, 180]), np.array([180, 40, 255]))
        white_ratio = np.count_nonzero(white_mask) / (h * w)
        if white_ratio > 0.15:
            landmarks.append({'class': 'refrigerator', 'conf': 0.75, 'bbox': [0.1, 0.1, 0.9, 0.9]})
            room_scores['kitchen'] += 1.5

        # Dark monitor / TV detector (Office / Living Room)
        dark_mask = cv2.inRange(hsv, np.array([0, 0, 0]), np.array([180, 255, 50]))
        dark_ratio = np.count_nonzero(dark_mask) / (h * w)
        if dark_ratio > 0.12:
            landmarks.append({'class': 'tv_display', 'conf': 0.70, 'bbox': [0.2, 0.2, 0.8, 0.8]})
            room_scores['living_room'] += 1.0
            room_scores['office'] += 1.0

        # Evaluate highest scoring room
        best_room = max(room_scores, key=room_scores.get)
        best_score = room_scores[best_room]

        if best_score > 0.5:
            conf = min(0.95, 0.5 + best_score * 0.2)
            return best_room, conf, landmarks

        # Default fallback
        return 'room', 0.40, landmarks

    # ── Service Callbacks ───────────────────────────────────────────────────

    def _srv_trigger_snapshot(self, request, response):
        """Manually trigger a camera snapshot and tag the current region."""
        if self._current_region_id <= 0:
            response.success = False
            response.message = "Robot is not inside a recognized topological region."
            return response

        self._trigger_on_demand_snapshot(self._current_region_id)
        response.success = True
        response.message = f"Snapshot triggered for active Region {self._current_region_id}."
        return response

    def _srv_get_scene_summary(self, request, response):
        """Return JSON summary of all segmented regions and centroids."""
        regions = self._db.get_all_regions()
        doors = self._db.get_all_chokepoints()
        summary = {
            'database_path': self._db_path,
            'region_count': len(regions),
            'doorway_count': len(doors),
            'regions': regions,
            'doorways': doors
        }
        response.success = True
        response.message = json.dumps(summary, indent=2)
        return response

    # ── RViz2 Marker Visualization ──────────────────────────────────────────

    def _publish_rviz_markers(self):
        """Publish 3D polygon outlines, text labels, and doorway markers for RViz."""
        regions = self._db.get_all_regions()
        doors = self._db.get_all_chokepoints()

        if not regions and not doors:
            return

        marker_array = MarkerArray()
        now = self.get_clock().now().to_msg()

        # Marker colors palette for distinct regions
        COLORS = [
            (0.2, 0.8, 0.2, 0.8),  # Green
            (0.2, 0.4, 0.9, 0.8),  # Blue
            (0.9, 0.5, 0.1, 0.8),  # Orange
            (0.8, 0.2, 0.8, 0.8),  # Purple
            (0.2, 0.8, 0.8, 0.8),  # Cyan
            (0.9, 0.8, 0.2, 0.8),  # Yellow
        ]

        marker_id = 0

        # 1. Room Polygons & Floating 3D Text Labels
        for idx, r in enumerate(regions):
            poly = r['polygon']
            r_id = r['region_id']
            label = r['label'].capitalize()
            conf = r['confidence']
            cx, cy = r['centroid_x'], r['centroid_y']

            color = COLORS[idx % len(COLORS)]

            # A. Polygon Outline Marker (LINE_STRIP)
            poly_marker = Marker()
            poly_marker.header.frame_id = 'map'
            poly_marker.header.stamp = now
            poly_marker.ns = 'room_boundaries'
            poly_marker.id = marker_id
            marker_id += 1
            poly_marker.type = Marker.LINE_STRIP
            poly_marker.action = Marker.ADD
            poly_marker.scale.x = 0.05  # Line thickness 5cm
            poly_marker.color.r, poly_marker.color.g, poly_marker.color.b, poly_marker.color.a = color

            for pt in poly:
                p = Point()
                p.x = float(pt[0])
                p.y = float(pt[1])
                p.z = 0.05
                poly_marker.points.append(p)

            # Close the loop
            if poly:
                p = Point()
                p.x = float(poly[0][0])
                p.y = float(poly[0][1])
                p.z = 0.05
                poly_marker.points.append(p)

            marker_array.markers.append(poly_marker)

            # B. Floating Text Label at Centroid
            text_marker = Marker()
            text_marker.header.frame_id = 'map'
            text_marker.header.stamp = now
            text_marker.ns = 'room_labels'
            text_marker.id = marker_id
            marker_id += 1
            text_marker.type = Marker.TEXT_VIEW_FACING
            text_marker.action = Marker.ADD
            text_marker.pose.position.x = cx
            text_marker.pose.position.y = cy
            text_marker.pose.position.z = 0.45  # Hover 45cm above ground
            text_marker.scale.z = 0.22         # Text size
            text_marker.color.r, text_marker.color.g, text_marker.color.b, text_marker.color.a = (1.0, 1.0, 1.0, 1.0)
            text_marker.text = f"{label} (R{r_id})\n[{conf*100:.0f}%]"
            marker_array.markers.append(text_marker)

            # C. Centroid Sphere Marker
            sphere_marker = Marker()
            sphere_marker.header.frame_id = 'map'
            sphere_marker.header.stamp = now
            sphere_marker.ns = 'room_centroids'
            sphere_marker.id = marker_id
            marker_id += 1
            sphere_marker.type = Marker.SPHERE
            sphere_marker.action = Marker.ADD
            sphere_marker.pose.position.x = cx
            sphere_marker.pose.position.y = cy
            sphere_marker.pose.position.z = 0.10
            sphere_marker.scale.x = 0.15
            sphere_marker.scale.y = 0.15
            sphere_marker.scale.z = 0.15
            sphere_marker.color.r, sphere_marker.color.g, sphere_marker.color.b, sphere_marker.color.a = color
            marker_array.markers.append(sphere_marker)

        # 2. Doorway / Chokepoint Markers (Yellow Cylinders)
        for idx, d in enumerate(doors):
            door_marker = Marker()
            door_marker.header.frame_id = 'map'
            door_marker.header.stamp = now
            door_marker.ns = 'doorways'
            door_marker.id = marker_id
            marker_id += 1
            door_marker.type = Marker.CYLINDER
            door_marker.action = Marker.ADD
            door_marker.pose.position.x = d['x']
            door_marker.pose.position.y = d['y']
            door_marker.pose.position.z = 0.15
            door_marker.scale.x = 0.20
            door_marker.scale.y = 0.20
            door_marker.scale.z = 0.30
            door_marker.color.r, door_marker.color.g, door_marker.color.b, door_marker.color.a = (1.0, 0.9, 0.0, 0.9)
            marker_array.markers.append(door_marker)

        self._markers_pub.publish(marker_array)


def main(args=None):
    rclpy.init(args=args)
    node = TopologicalSceneGraphNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
