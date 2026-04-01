#!/usr/bin/env python3
"""
vila_scene_labeller_node.py  — VILA 2.7B AI navigator + scene labeller
=======================================================================
Runs NVIDIA VILA 2.7B (4-bit AWQ) via nano_llm on the Jetson GPU.
Triggered by exploration_controller every N steps via /robot/movement_complete.
Grabs a camera frame, asks VILA to:
  1) Identify what area/room this is (scene label for map annotation)
  2) Determine the best direction to explore (forward/left/right/stop)

The LiDAR is the primary collision sensor — the AI chooses the *optimal
exploration direction*, not the safe one. exploration_controller fuses
AI direction with LiDAR zone readings to produce the final /cmd_vel.

Room Identification
-------------------
When room_hints_enabled=true, accumulates VILA labels in a sliding window
and infers room type from keyword matches. Also subscribes to /scan for
hallway geometry detection (narrow left+right, long front/rear).
Zero extra inference cost — pure post-processing.

Logging
-------
Creates /root/ros2_ws/logs/<session_timestamp>/:
  decisions_full.txt    — every AI decision + room + motor cmd
  captures/             — camera frame images saved every 5th decision (max 10)
  captures/NNN.json     — metadata for each saved frame

Subscribes
----------
/robot/movement_complete  std_msgs/Bool       trigger from exploration_controller
/image_raw                sensor_msgs/Image   camera feed (640x480)
/cmd_vel                  geometry_msgs/Twist  latest motor command (for logging)
/scan                     sensor_msgs/LaserScan  for hallway geometry (optional)

Publishes
---------
/ai/semantic_label        std_msgs/String    e.g. "near desk"
/ai/direction             std_msgs/String    e.g. "forward" | "left" | "right" | "stop"
/ai/decision              std_msgs/String    full JSON for logging
/ai/room                  std_msgs/String    inferred room (e.g. "bedroom", "hallway")

Requires
--------
- Running inside dustynv/nano_llm Docker container on Jetson (nano_llm + ROS2 Humble)
- VILA model downloaded: Efficient-Large-Model/VILA-2.7b
"""
import math
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, LaserScan
from geometry_msgs.msg import Twist
from std_msgs.msg import String, Bool
import json
import time
import os
import threading
from datetime import datetime
from collections import deque

# ── nano_llm import (only available inside the Jetson container) ──────────
_NANO_LLM_AVAILABLE = False
try:
    from nano_llm import NanoLLM
    from nano_llm.utils import cuda_image
    _NANO_LLM_AVAILABLE = True
except ImportError:
    pass

# cv_bridge + PIL for image conversion
try:
    from cv_bridge import CvBridge
    import cv2
    from PIL import Image as PILImage
    import numpy as np
    _CV_AVAILABLE = True
except ImportError:
    _CV_AVAILABLE = False


# ═══════════════════════════════════════════════════════════════════════════
# Room Identification — keyword accumulator + LiDAR geometry
# ═══════════════════════════════════════════════════════════════════════════

# Object keywords → room category
ROOM_HINTS = {
    'bedroom':   ['bed', 'pillow', 'mattress', 'dresser', 'drawer', 'nightstand',
                  'wardrobe', 'closet', 'blanket', 'duvet', 'headboard'],
    'kitchen':   ['stove', 'oven', 'sink', 'fridge', 'refrigerator', 'counter',
                  'cabinet', 'microwave', 'dishwasher', 'pan', 'pot'],
    'bathroom':  ['toilet', 'shower', 'bathtub', 'sink', 'mirror', 'towel',
                  'tile', 'faucet'],
    'living':    ['couch', 'sofa', 'tv', 'television', 'coffee table',
                  'bookshelf', 'armchair', 'remote', 'carpet', 'rug'],
    'dining':    ['dining table', 'chair', 'plate', 'candle', 'tablecloth'],
    'hallway':   ['hallway', 'corridor', 'narrow', 'doorway', 'passage', 'entry'],
    'office':    ['desk', 'monitor', 'keyboard', 'mouse', 'computer', 'office chair'],
    'garage':    ['car', 'tool', 'workbench', 'garage', 'concrete'],
    'laundry':   ['washer', 'dryer', 'laundry', 'iron', 'detergent'],
}


class RoomIdentifier:
    """Accumulates VILA labels and LiDAR geometry to infer room type."""

    def __init__(self, window_size: int = 10, logger=None):
        self._window = deque(maxlen=window_size)
        self._no_object_streak = 0
        self._current_room = 'unknown'
        self._logger = logger

        # LiDAR geometry (updated externally)
        self.lidar_left_min = 9.9
        self.lidar_right_min = 9.9
        self.lidar_front_max = 0.0
        self.lidar_rear_max = 0.0

    def update(self, label: str) -> str:
        """Process a new VILA label and return the inferred room."""
        label_lower = label.lower()

        # Scan for keywords
        hits = {}  # room_category → count of matching keywords in this label
        for room, keywords in ROOM_HINTS.items():
            for kw in keywords:
                if kw in label_lower:
                    hits[room] = hits.get(room, 0) + 1

        if hits:
            self._no_object_streak = 0
            best_room = max(hits, key=hits.get)
            self._window.append(best_room)
        else:
            self._no_object_streak += 1
            self._window.append(None)

        # Count accumulated evidence across the window
        counts = {}
        for entry in self._window:
            if entry:
                counts[entry] = counts.get(entry, 0) + 1

        if counts:
            top_room = max(counts, key=counts.get)
            top_count = counts[top_room]

            if top_count >= 2:
                # Strong evidence: 2+ matching labels → room name
                self._current_room = top_room
            elif top_count == 1:
                # Single match → "X area"
                self._current_room = f'{top_room} area'
        elif self._no_object_streak >= 5:
            # No objects for 5+ labels — check LiDAR geometry for hallway
            if self._is_hallway_geometry():
                self._current_room = 'hallway'
            else:
                self._current_room = 'open area'

        return self._current_room

    def _is_hallway_geometry(self) -> bool:
        """Check if LiDAR indicates hallway shape (narrow sides, long axis)."""
        narrow_avg = (self.lidar_left_min + self.lidar_right_min) / 2.0
        long_axis = max(self.lidar_front_max, self.lidar_rear_max)

        # Hallway: sides < 1.5m average AND one axis > 3× the narrow dimension
        return narrow_avg < 1.5 and long_axis > (narrow_avg * 3.0)


# ═══════════════════════════════════════════════════════════════════════════
# VILA Scene Labeller Node
# ═══════════════════════════════════════════════════════════════════════════

class VilaSceneLabeller(Node):
    """ROS2 node: VILA 2.7B for exploration guidance + scene labelling."""

    def __init__(self):
        super().__init__('vila_scene_labeller')

        # ── Parameters ────────────────────────────────────────────────────
        self.declare_parameter('model_name',    'Efficient-Large-Model/VILA-2.7b')
        self.declare_parameter('api',           'awq')
        self.declare_parameter('quantization',  'q4f16_ft')
        self.declare_parameter('infer_width',   384)
        self.declare_parameter('infer_height',  384)
        self.declare_parameter('max_new_tokens', 64)
        self.declare_parameter('log_dir',       '/root/ros2_ws/logs')
        self.declare_parameter('capture_every',  5)     # save image every N decisions
        self.declare_parameter('max_captures',   10)    # max number of images to save
        self.declare_parameter('room_hints_enabled', True)  # toggle room identification
        self.declare_parameter('prompt', (
            'You are a navigation assistant for a robot exploring a room and '
            'building a map. Look at this image and answer in EXACTLY this format:\n'
            'DIRECTION: [forward|left|right|stop]\n'
            'LABEL: [short scene description, 5 words max]\n'
            'REASON: [one short sentence]\n\n'
            'Choose "forward" if the path ahead is clear (>1 meter of open space). '
            'Choose "left" or "right" to explore unexplored areas or avoid obstacles '
            'like walls, furniture, or clutter. '
            'Choose "stop" only if something dangerous is directly in front. '
            'For LABEL, describe the area (e.g. "near desk", "open floor", "doorway", '
            '"corner by wardrobe").'
        ))

        self.model_name     = self.get_parameter('model_name').value
        self.api            = self.get_parameter('api').value
        self.quantization   = self.get_parameter('quantization').value
        self.infer_width    = self.get_parameter('infer_width').value
        self.infer_height   = self.get_parameter('infer_height').value
        self.max_new_tokens = self.get_parameter('max_new_tokens').value
        self.prompt         = self.get_parameter('prompt').value
        self._capture_every = int(self.get_parameter('capture_every').value)
        self._max_captures  = int(self.get_parameter('max_captures').value)
        self._room_hints_on = bool(self.get_parameter('room_hints_enabled').value)

        # ── State ─────────────────────────────────────────────────────────
        self._model = None
        self._model_ready = False
        self._latest_frame = None
        self._processing = False
        self._last_label = 'unknown'
        self._bridge = CvBridge() if _CV_AVAILABLE else None
        self._decision_count = 0
        self._capture_count = 0
        self._latest_cmd_vel = None

        # Room identification
        self._room_id = RoomIdentifier(
            window_size=10, logger=self.get_logger()) if self._room_hints_on else None
        self._current_room = 'unknown'

        # ── Logging setup ─────────────────────────────────────────────────
        log_base = self.get_parameter('log_dir').value
        session_ts = datetime.now().strftime('%Y%m%d_%H%M%S')
        self._session_dir = os.path.join(log_base, session_ts)
        self._captures_dir = os.path.join(self._session_dir, 'captures')
        os.makedirs(self._captures_dir, exist_ok=True)

        self._decisions_log_path = os.path.join(
            self._session_dir, 'decisions_full.txt')

        with open(self._decisions_log_path, 'w') as f:
            f.write(f'# VILA Decision Log — Session {session_ts}\n')
            f.write(f'# Model: {self.model_name}  API: {self.api}\n')
            f.write(f'# Room hints: {"ON" if self._room_hints_on else "OFF"}\n')
            f.write(f'# Format: timestamp | #N | direction | label | room | '
                    f'reason | motor_cmd | inference_s\n')
            f.write('=' * 110 + '\n')

        self.get_logger().info(
            f'Logging to {self._session_dir} | '
            f'capture every {self._capture_every} decisions, '
            f'max {self._max_captures} images | '
            f'room_hints={"ON" if self._room_hints_on else "OFF"}')

        # ── Subscribers ───────────────────────────────────────────────────
        self.create_subscription(
            Bool, '/robot/movement_complete', self._on_trigger, 10)
        self.create_subscription(
            Image, '/image_raw', self._on_image, 1)
        self.create_subscription(
            Twist, '/cmd_vel', self._on_cmd_vel, 10)

        # LiDAR subscription for hallway geometry (lightweight — just read zone ranges)
        if self._room_hints_on:
            self.create_subscription(
                LaserScan, '/scan', self._on_scan_for_geometry, 1)

        # ── Publishers ────────────────────────────────────────────────────
        self._label_pub = self.create_publisher(String, '/ai/semantic_label', 10)
        self._dir_pub   = self.create_publisher(String, '/ai/direction',      10)
        self._dec_pub   = self.create_publisher(String, '/ai/decision',       10)
        self._room_pub  = self.create_publisher(String, '/ai/room',           10)

        # ── Load model in background ──────────────────────────────────────
        if _NANO_LLM_AVAILABLE:
            self.get_logger().info(
                f'Loading VILA model: {self.model_name} (api={self.api}) ...')
            self._load_thread = threading.Thread(
                target=self._load_model, daemon=True)
            self._load_thread.start()
        else:
            self.get_logger().warn(
                'nano_llm not available — running in FALLBACK mode. '
                'Will publish "forward" + "vila_unavailable".')

        self.get_logger().info(
            f'VILA AI navigator ready | '
            f'inference size={self.infer_width}x{self.infer_height}')

    # ── Model loading ─────────────────────────────────────────────────────
    def _load_model(self):
        """Load the VILA model on a background thread (GPU-heavy)."""
        try:
            t0 = time.time()
            self._model = NanoLLM.from_pretrained(
                self.model_name,
                api=self.api,
                quantization=self.quantization,
            )
            elapsed = time.time() - t0
            self._model_ready = True
            self.get_logger().info(
                f'VILA model loaded in {elapsed:.1f}s — ready for inference')
        except Exception as e:
            self.get_logger().error(f'Failed to load VILA model: {e}')
            self._model_ready = False

    # ── Callbacks ─────────────────────────────────────────────────────────
    def _on_image(self, msg: Image):
        self._latest_frame = msg

    def _on_cmd_vel(self, msg: Twist):
        self._latest_cmd_vel = msg

    def _on_scan_for_geometry(self, msg: LaserScan):
        """Extract LiDAR geometry for hallway detection (lightweight)."""
        if not self._room_id:
            return
        left_vals, right_vals = [], []
        front_vals, rear_vals = [], []

        for i, r in enumerate(msg.ranges):
            if not (msg.range_min <= r <= msg.range_max):
                continue
            angle = msg.angle_min + i * msg.angle_increment
            angle = math.atan2(math.sin(angle), math.cos(angle))
            deg = math.degrees(angle)

            if -60.0 <= deg <= 60.0:
                front_vals.append(r)
            elif 60.0 < deg <= 120.0:
                left_vals.append(r)
            elif -120.0 <= deg < -60.0:
                right_vals.append(r)
            else:
                rear_vals.append(r)

        self._room_id.lidar_left_min = min(left_vals) if left_vals else 9.9
        self._room_id.lidar_right_min = min(right_vals) if right_vals else 9.9
        self._room_id.lidar_front_max = max(front_vals) if front_vals else 0.0
        self._room_id.lidar_rear_max = max(rear_vals) if rear_vals else 0.0

    def _on_trigger(self, msg: Bool):
        if not msg.data or self._processing:
            return
        if self._latest_frame is None:
            self.get_logger().warn('Triggered but no camera frame available yet')
            return
        if not _NANO_LLM_AVAILABLE:
            self._publish_result('forward', 'vila_unavailable', '', 0.0, None)
            return
        if not self._model_ready:
            self.get_logger().warn('Model still loading, skipping this trigger')
            return

        self.get_logger().info('Triggered — running VILA navigation inference...')
        frame_msg = self._latest_frame
        threading.Thread(
            target=self._analyze_scene, args=(frame_msg,), daemon=True).start()

    # ── Inference ─────────────────────────────────────────────────────────
    def _analyze_scene(self, frame_msg: Image):
        self._processing = True
        try:
            pil_image = self._ros_image_to_pil(frame_msg)
            if pil_image is None:
                self.get_logger().error('Failed to convert ROS image')
                return

            full_res_image = pil_image.copy()
            pil_image = pil_image.resize(
                (self.infer_width, self.infer_height), PILImage.LANCZOS)

            t0 = time.time()
            response = self._model.generate(
                self.prompt, image=pil_image, max_new_tokens=self.max_new_tokens)
            elapsed = time.time() - t0

            raw_response = str(response).strip()
            self.get_logger().info(f'VILA response ({elapsed:.1f}s): {raw_response}')

            direction = self._parse_direction(raw_response)
            label = self._parse_label(raw_response)
            reason = self._parse_reason(raw_response)

            # Room identification (zero inference cost)
            if self._room_id:
                self._current_room = self._room_id.update(label)
            else:
                self._current_room = 'disabled'

            self._publish_result(direction, label, reason, elapsed, full_res_image)

        except Exception as e:
            self.get_logger().error(f'Scene analysis failed: {e}')
            self._publish_result('forward', 'unknown', f'error: {e}', 0.0, None)
        finally:
            self._processing = False

    # ── Response parsing ──────────────────────────────────────────────────
    def _parse_direction(self, response: str) -> str:
        response_lower = response.lower()
        if 'direction:' in response_lower:
            after = response_lower.split('direction:')[-1].strip()
            for d in ['forward', 'left', 'right', 'stop']:
                if after.startswith(d):
                    return d
        if any(w in response_lower for w in
               ['blocked', 'wall', 'obstacle', 'furniture', 'cannot', 'no path']):
            return 'left'
        if 'stop' in response_lower and 'danger' in response_lower:
            return 'stop'
        if 'right' in response_lower:
            return 'right'
        if 'left' in response_lower:
            return 'left'
        return 'forward'

    def _parse_label(self, response: str) -> str:
        response_lower = response.lower()
        if 'label:' in response_lower:
            after = response_lower.split('label:')[1].strip()
            for sep in ['\n', 'reason:', 'direction:']:
                if sep in after:
                    after = after.split(sep)[0]
            label = after.strip().strip('"').strip()
            if label:
                return label
        for room in ['bedroom', 'kitchen', 'bathroom', 'hallway', 'living room']:
            if room in response_lower:
                return room
        return 'unknown area'

    def _parse_reason(self, response: str) -> str:
        response_lower = response.lower()
        if 'reason:' in response_lower:
            after = response.split('eason:')[-1].strip()
            for sep in ['\n', 'DIRECTION:', 'LABEL:']:
                if sep in after:
                    after = after.split(sep)[0]
            return after.strip()
        return ''

    # ── Helpers ───────────────────────────────────────────────────────────
    def _ros_image_to_pil(self, msg: Image):
        if not _CV_AVAILABLE:
            self.get_logger().error('cv_bridge / OpenCV not available')
            return None
        try:
            cv_img = self._bridge.imgmsg_to_cv2(msg, desired_encoding='rgb8')
            return PILImage.fromarray(cv_img)
        except Exception as e:
            self.get_logger().error(f'Image conversion error: {e}')
            return None

    def _publish_result(self, direction: str, label: str,
                        reason: str, inference_s: float,
                        full_res_image=None):
        self._last_label = label
        self._decision_count += 1
        now_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]

        cmd_str = 'none'
        if self._latest_cmd_vel:
            lx = self._latest_cmd_vel.linear.x
            az = self._latest_cmd_vel.angular.z
            cmd_str = f'linear={lx:.3f} angular={az:.3f}'

        # ── Publish to ROS topics ─────────────────────────────────────
        lmsg = String()
        lmsg.data = label
        self._label_pub.publish(lmsg)

        dmsg = String()
        dmsg.data = direction
        self._dir_pub.publish(dmsg)

        # Room topic
        rmsg = String()
        rmsg.data = self._current_room
        self._room_pub.publish(rmsg)

        decision_data = {
            'timestamp': now_str,
            'decision_num': self._decision_count,
            'direction': direction,
            'label': label,
            'room': self._current_room,
            'reason': reason,
            'motor_cmd': cmd_str,
            'model': self.model_name,
            'inference_s': round(inference_s, 2),
        }
        jmsg = String()
        jmsg.data = json.dumps(decision_data)
        self._dec_pub.publish(jmsg)

        # ── Log to decisions_full.txt ─────────────────────────────────
        log_line = (
            f'{now_str} | #{self._decision_count:03d} | '
            f'dir={direction:<8s} | label="{label}" | '
            f'room="{self._current_room}" | '
            f'reason="{reason}" | motor=({cmd_str}) | '
            f'infer={inference_s:.2f}s\n'
        )
        try:
            with open(self._decisions_log_path, 'a') as f:
                f.write(log_line)
        except Exception as e:
            self.get_logger().warn(f'Failed to write decision log: {e}')

        # ── Save camera frame ─────────────────────────────────────────
        if (full_res_image is not None
                and self._decision_count % self._capture_every == 0
                and self._capture_count < self._max_captures):
            self._capture_count += 1
            img_name = f'capture_{self._capture_count:03d}.jpg'
            meta_name = f'capture_{self._capture_count:03d}.json'
            img_path = os.path.join(self._captures_dir, img_name)
            meta_path = os.path.join(self._captures_dir, meta_name)

            try:
                full_res_image.save(img_path, 'JPEG', quality=90)
                with open(meta_path, 'w') as f:
                    json.dump({
                        **decision_data,
                        'image_file': img_name,
                        'capture_num': self._capture_count,
                    }, f, indent=2)
                self.get_logger().info(
                    f'📸 Saved capture {self._capture_count}/{self._max_captures}: '
                    f'{img_name} | room="{self._current_room}" | dir={direction}')
            except Exception as e:
                self.get_logger().warn(f'Failed to save capture: {e}')

        self.get_logger().info(
            f'→ direction={direction} | label="{label}" | '
            f'room="{self._current_room}" | {inference_s:.1f}s')


def main(args=None):
    rclpy.init(args=args)
    node = VilaSceneLabeller()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.get_logger().info(
        f'Session complete: {node._decision_count} decisions, '
        f'{node._capture_count} captures saved to {node._session_dir}')
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
