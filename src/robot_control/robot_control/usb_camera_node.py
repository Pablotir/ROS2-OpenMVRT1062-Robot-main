#!/usr/bin/env python3
"""
usb_camera_node.py  (updated)
==============================
Captures from a USB camera and serves two streams:

1. /image_raw  + /camera_info  — continuous at `slam_fps` Hz (default 5)
   Used by RTAB-Map for SLAM and loop closure.

2. /camera/usb_raw             — on-demand, triggered by /camera/request_frame (Bool)
   Used by ai_navigator for vision-LLM queries (keeps the request/response pattern).

Parameters
----------
device      string  /dev/video1 (or /dev/video0)
width       int     320
height      int     240
slam_fps    float   5.0   (frames/s for the RTAB-Map stream)
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from std_msgs.msg import Bool
from cv_bridge import CvBridge
import cv2


class USBCameraNode(Node):
    def __init__(self):
        super().__init__('usb_camera_node')

        self.declare_parameter('device',    '/dev/video1')
        self.declare_parameter('width',     320)
        self.declare_parameter('height',    240)
        self.declare_parameter('slam_fps',  5.0)

        device   = self.get_parameter('device').value
        width    = self.get_parameter('width').value
        height   = self.get_parameter('height').value
        slam_fps = self.get_parameter('slam_fps').value

        self._width  = width
        self._height = height

        # ── Publishers ────────────────────────────────────────────────────────
        # Stream 1 — continuous for RTAB-Map
        self._slam_img_pub  = self.create_publisher(Image,      '/image_raw',    10)
        self._slam_info_pub = self.create_publisher(CameraInfo, '/camera_info',  10)

        # Stream 2 — on-demand for AI navigator
        self._ai_img_pub = self.create_publisher(Image, '/camera/usb_raw', 10)

        # ── Subscribers ───────────────────────────────────────────────────────
        self.create_subscription(Bool, '/camera/request_frame',
                                 self._on_frame_request, 10)

        self._bridge = CvBridge()

        # ── Camera ────────────────────────────────────────────────────────────
        self._cap = cv2.VideoCapture(device)
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH,  width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        # Ask for a buffer size of 1 so we always get the freshest frame
        self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        if not self._cap.isOpened():
            self.get_logger().error(f'Failed to open camera: {device}')
            return

        self.get_logger().info(
            f'USB camera ready: {device} @ {width}x{height} | '
            f'SLAM stream: {slam_fps} fps')

        # ── Timers ────────────────────────────────────────────────────────────
        self.create_timer(1.0 / slam_fps, self._slam_publish)

        self._frame_count = 0

    # ── RTAB-Map continuous stream ─────────────────────────────────────────────

    def _slam_publish(self):
        """Capture and publish one frame on /image_raw for RTAB-Map."""
        frame = self._capture()
        if frame is None:
            return

        now = self.get_clock().now().to_msg()
        ros_img = self._bridge.cv2_to_imgmsg(frame, encoding='bgr8')
        ros_img.header.stamp    = now
        ros_img.header.frame_id = 'camera_optical_link'
        self._slam_img_pub.publish(ros_img)

        # Publish a basic (uncalibrated) CameraInfo so RTAB-Map doesn't stall.
        # Replace with a real calibration file once you run the calibrator.
        info = self._make_camera_info(now)
        self._slam_info_pub.publish(info)

    # ── AI on-demand stream ────────────────────────────────────────────────────

    def _on_frame_request(self, msg: Bool):
        """Capture one frame and publish to /camera/usb_raw for ai_navigator."""
        if not msg.data:
            return
        frame = self._capture()
        if frame is None:
            self.get_logger().warn('Failed to capture frame on request')
            return

        ros_img = self._bridge.cv2_to_imgmsg(frame, encoding='bgr8')
        ros_img.header.stamp    = self.get_clock().now().to_msg()
        ros_img.header.frame_id = 'camera_optical_link'
        self._ai_img_pub.publish(ros_img)

        self._frame_count += 1
        self.get_logger().info(f'AI frame {self._frame_count} published on request')

    # ── Helpers ────────────────────────────────────────────────────────────────

    def _capture(self):
        """Grab the freshest frame from the camera buffer."""
        if not self._cap or not self._cap.isOpened():
            return None
        # Drain the queue to get the latest frame
        for _ in range(2):
            self._cap.grab()
        ret, frame = self._cap.retrieve()
        return frame if ret else None

    def _make_camera_info(self, stamp) -> CameraInfo:
        """
        Build a default (uncalibrated) CameraInfo.
        fx = fy ≈ width (rough 60° HFOV approximation).
        Run camera_calibration to replace these with real values.
        """
        info = CameraInfo()
        info.header.stamp    = stamp
        info.header.frame_id = 'camera_optical_link'
        info.width  = self._width
        info.height = self._height
        info.distortion_model = 'plumb_bob'

        fx = float(self._width)       # rough estimate — calibrate for real use
        fy = float(self._width)
        cx = self._width  / 2.0
        cy = self._height / 2.0

        info.k = [fx, 0., cx,
                  0., fy, cy,
                  0., 0., 1.]
        info.r = [1., 0., 0.,
                  0., 1., 0.,
                  0., 0., 1.]
        info.p = [fx, 0., cx, 0.,
                  0., fy, cy, 0.,
                  0., 0., 1., 0.]
        info.d = [0., 0., 0., 0., 0.]
        return info

    def destroy_node(self):
        if self._cap:
            self._cap.release()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = USBCameraNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
