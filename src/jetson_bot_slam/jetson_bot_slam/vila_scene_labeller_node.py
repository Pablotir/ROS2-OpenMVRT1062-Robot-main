#!/usr/bin/env python3
"""
vila_scene_labeller_node.py  — VILA 2.7B AI navigator + scene labeller
=======================================================================
Runs NVIDIA VILA 2.7B (4-bit AWQ) via nano_llm on the Jetson GPU.
Triggered by exploration_controller every N steps via /robot/movement_complete.
Grabs a camera frame, asks VILA to:
  1) Identify what area/room this is (scene label for map annotation)
  2) Determine the best direction to explore (forward/left/right/stop)

The ultrasonic sensor on the Arduino remains the primary collision limiter —
the AI chooses the *optimal exploration direction*, not the safe one.
exploration_controller fuses AI direction with ultrasonic readings to produce
the final /cmd_vel.

This replaces the old Ollama-based ai_navigator.py (gemma3:4b).

Memory footprint: ~1.8-2 GB GPU RAM (vs ~4 GB for gemma3:4b via Ollama).
Inference speed:  ~0.5-1.5s per frame on Jetson Orin Nano.

Logging
-------
Creates /root/ros2_ws/logs/<session_timestamp>/:
  decisions_full.txt    — every AI decision with timestamp + label + direction + motor cmd
  captures/             — camera frame images saved every 5th decision (max 10)
  captures/NNN.json     — metadata for each saved frame

Subscribes
----------
/robot/movement_complete  std_msgs/Bool      trigger from exploration_controller
/image_raw                sensor_msgs/Image  camera feed (640x480, also used by RTAB-Map)
/cmd_vel                  geometry_msgs/Twist latest motor command (for logging)

Publishes
---------
/ai/semantic_label        std_msgs/String    e.g. "near desk"
/ai/direction             std_msgs/String    e.g. "forward" | "left" | "right" | "stop"
/ai/decision              std_msgs/String    full JSON for logging

Requires
--------
- Running inside dustynv/nano_llm Docker container on Jetson (nano_llm + ROS2 Humble)
- VILA model downloaded: Efficient-Large-Model/VILA-2.7b
"""
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
from std_msgs.msg import String, Bool
import json
import time
import os
import threading
from datetime import datetime

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

        # ── State ─────────────────────────────────────────────────────────
        self._model = None
        self._model_ready = False
        self._latest_frame = None       # most recent ROS Image message
        self._processing = False
        self._last_label = 'unknown'
        self._bridge = CvBridge() if _CV_AVAILABLE else None
        self._decision_count = 0        # total decisions made
        self._capture_count = 0         # total images saved
        self._latest_cmd_vel = None     # last /cmd_vel for motor logging

        # ── Logging setup ─────────────────────────────────────────────────
        log_base = self.get_parameter('log_dir').value
        session_ts = datetime.now().strftime('%Y%m%d_%H%M%S')
        self._session_dir = os.path.join(log_base, session_ts)
        self._captures_dir = os.path.join(self._session_dir, 'captures')
        os.makedirs(self._captures_dir, exist_ok=True)

        self._decisions_log_path = os.path.join(
            self._session_dir, 'decisions_full.txt')

        # Write log header
        with open(self._decisions_log_path, 'w') as f:
            f.write(f'# VILA Decision Log — Session {session_ts}\n')
            f.write(f'# Model: {self.model_name}  API: {self.api}\n')
            f.write(f'# Format: timestamp | #N | direction | label | '
                    f'reason | motor_cmd (linear.x, angular.z) | inference_s\n')
            f.write('=' * 100 + '\n')

        self.get_logger().info(
            f'Logging to {self._session_dir} | '
            f'capture every {self._capture_every} decisions, '
            f'max {self._max_captures} images')

        # ── Subscribers ───────────────────────────────────────────────────
        self.create_subscription(
            Bool, '/robot/movement_complete', self._on_trigger, 10)
        self.create_subscription(
            Image, '/image_raw', self._on_image, 1)  # queue=1, keep latest only
        self.create_subscription(
            Twist, '/cmd_vel', self._on_cmd_vel, 10)  # track motor commands

        # ── Publishers ────────────────────────────────────────────────────
        self._label_pub = self.create_publisher(String, '/ai/semantic_label', 10)
        self._dir_pub   = self.create_publisher(String, '/ai/direction',      10)
        self._dec_pub   = self.create_publisher(String, '/ai/decision',       10)

        # ── Load model in background (takes 10-30s on first run) ──────────
        if _NANO_LLM_AVAILABLE:
            self.get_logger().info(
                f'Loading VILA model: {self.model_name} (api={self.api}) ...')
            self._load_thread = threading.Thread(
                target=self._load_model, daemon=True)
            self._load_thread.start()
        else:
            self.get_logger().warn(
                'nano_llm not available — running in FALLBACK mode. '
                'Will publish "forward" + "vila_unavailable". '
                'Make sure you are inside the dustynv/nano_llm container.')

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
        """Store the latest camera frame (lightweight — no processing here)."""
        self._latest_frame = msg

    def _on_cmd_vel(self, msg: Twist):
        """Track latest motor command for logging."""
        self._latest_cmd_vel = msg

    def _on_trigger(self, msg: Bool):
        """Called when exploration_controller signals movement is complete."""
        if not msg.data or self._processing:
            return

        if self._latest_frame is None:
            self.get_logger().warn('Triggered but no camera frame available yet')
            return

        if not _NANO_LLM_AVAILABLE:
            # Fallback: publish defaults so the robot keeps exploring
            self._publish_result('forward', 'vila_unavailable', '', 0.0, None)
            return

        if not self._model_ready:
            self.get_logger().warn('Model still loading, skipping this trigger')
            return

        self.get_logger().info('Triggered — running VILA navigation inference...')
        # Run inference in background so ROS callbacks stay responsive
        frame_msg = self._latest_frame
        threading.Thread(
            target=self._analyze_scene, args=(frame_msg,), daemon=True).start()

    # ── Inference ─────────────────────────────────────────────────────────
    def _analyze_scene(self, frame_msg: Image):
        """Run VILA inference: get both direction and scene label."""
        self._processing = True
        try:
            # Convert ROS Image → PIL Image
            pil_image = self._ros_image_to_pil(frame_msg)
            if pil_image is None:
                self.get_logger().error('Failed to convert ROS image')
                return

            # Keep a copy of the full-res frame for capture logging
            full_res_image = pil_image.copy()

            # Resize for inference (640x480 → 384x384)
            pil_image = pil_image.resize(
                (self.infer_width, self.infer_height),
                PILImage.LANCZOS,
            )

            # Run VILA inference
            t0 = time.time()
            response = self._model.generate(
                self.prompt,
                image=pil_image,
                max_new_tokens=self.max_new_tokens,
            )
            elapsed = time.time() - t0

            raw_response = str(response).strip()
            self.get_logger().info(
                f'VILA response ({elapsed:.1f}s): {raw_response}')

            # Parse structured response
            direction = self._parse_direction(raw_response)
            label = self._parse_label(raw_response)
            reason = self._parse_reason(raw_response)

            self._publish_result(
                direction, label, reason, elapsed, full_res_image)

        except Exception as e:
            self.get_logger().error(f'Scene analysis failed: {e}')
            # Publish safe default so robot doesn't freeze
            self._publish_result('forward', 'unknown', f'error: {e}', 0.0, None)
        finally:
            self._processing = False

    # ── Response parsing ──────────────────────────────────────────────────
    def _parse_direction(self, response: str) -> str:
        """Extract navigation direction from VILA response."""
        response_lower = response.lower()

        # Look for explicit DIRECTION: tag
        if 'direction:' in response_lower:
            after = response_lower.split('direction:')[-1].strip()
            for d in ['forward', 'left', 'right', 'stop']:
                if after.startswith(d):
                    return d

        # Fallback: keyword scan
        if any(w in response_lower for w in
               ['blocked', 'wall', 'obstacle', 'furniture', 'cannot', 'no path']):
            return 'left'
        if 'stop' in response_lower and 'danger' in response_lower:
            return 'stop'
        if 'right' in response_lower:
            return 'right'
        if 'left' in response_lower:
            return 'left'

        return 'forward'   # default: keep exploring

    def _parse_label(self, response: str) -> str:
        """Extract scene label from VILA response."""
        response_lower = response.lower()

        # Look for explicit LABEL: tag
        if 'label:' in response_lower:
            after = response_lower.split('label:')[1].strip()
            # Take everything up to the next tag or newline
            for sep in ['\n', 'reason:', 'direction:']:
                if sep in after:
                    after = after.split(sep)[0]
            label = after.strip().strip('"').strip()
            if label:
                return label

        # Fallback: try to extract from context keywords
        for room in ['bedroom', 'kitchen', 'bathroom', 'hallway', 'living room']:
            if room in response_lower:
                return room

        return 'unknown area'

    def _parse_reason(self, response: str) -> str:
        """Extract reasoning from VILA response."""
        response_lower = response.lower()
        if 'reason:' in response_lower:
            after = response.split('eason:')[-1].strip()  # case-insensitive split
            for sep in ['\n', 'DIRECTION:', 'LABEL:']:
                if sep in after:
                    after = after.split(sep)[0]
            return after.strip()
        return ''

    # ── Helpers ───────────────────────────────────────────────────────────
    def _ros_image_to_pil(self, msg: Image):
        """Convert a ROS2 sensor_msgs/Image to a PIL Image."""
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
        """Publish direction, scene label, full decision JSON, and log to disk."""
        self._last_label = label
        self._decision_count += 1
        now_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]

        # Current motor command
        cmd_str = 'none'
        if self._latest_cmd_vel:
            lx = self._latest_cmd_vel.linear.x
            az = self._latest_cmd_vel.angular.z
            cmd_str = f'linear={lx:.3f} angular={az:.3f}'

        # ── Publish to ROS topics ─────────────────────────────────────
        # Scene label (for map annotation)
        lmsg = String()
        lmsg.data = label
        self._label_pub.publish(lmsg)

        # Navigation direction (for exploration_controller)
        dmsg = String()
        dmsg.data = direction
        self._dir_pub.publish(dmsg)

        # Full decision JSON (for real-time monitoring)
        decision_data = {
            'timestamp': now_str,
            'decision_num': self._decision_count,
            'direction': direction,
            'label': label,
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
            f'reason="{reason}" | motor=({cmd_str}) | '
            f'infer={inference_s:.2f}s\n'
        )
        try:
            with open(self._decisions_log_path, 'a') as f:
                f.write(log_line)
        except Exception as e:
            self.get_logger().warn(f'Failed to write decision log: {e}')

        # ── Save camera frame (every N decisions, max M captures) ─────
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
                    f'{img_name} | label="{label}" | dir={direction}')
            except Exception as e:
                self.get_logger().warn(f'Failed to save capture: {e}')

        self.get_logger().info(
            f'→ direction={direction} | label="{label}" | {inference_s:.1f}s')


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
