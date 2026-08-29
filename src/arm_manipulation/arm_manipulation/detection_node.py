import os
import rclpy
from rclpy.node import Node
import numpy as np
import cv2
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import PointStamped
from std_msgs.msg import Float64, String
from vision_msgs.msg import Detection2DArray, Detection2D, ObjectHypothesisWithPose, BoundingBox2D
from cv_bridge import CvBridge
from ultralytics import YOLO

class DetectionNode(Node):
    def __init__(self):
        super().__init__('detection_node')
        
        # Parameters
        self.declare_parameter('model_path', '/root/ros2_ws/models/yolo11n-seg.engine')
        self.declare_parameter('model_fallback_path', '/root/ros2_ws/models/yolo11n-seg.pt')
        self.declare_parameter('target_class_id', 39) # 39 = bottle in COCO
        self.declare_parameter('target_label', 'bottle')
        self.declare_parameter('confidence_threshold', 0.5)
        self.declare_parameter('imgsz', 640)
        self.declare_parameter('publish_rate', 30.0)
        
        self.target_class_id = self.get_parameter('target_class_id').value
        self.target_label = self.get_parameter('target_label').value
        self.conf_thresh = self.get_parameter('confidence_threshold').value
        
        self.bridge = CvBridge()
        
        self._load_model()
        
        # State
        self.latest_depth = None
        self.latest_camera_info = None
        
        # COCO Class name mapping dictionary
        self.class_name_map = {
            'bottle': 39,
            'cup': 41,
            'can': 39,
            'ball': 32,
            'sports ball': 32,
            'bowl': 45,
            'apple': 47,
            'banana': 46,
            'remote': 65,
            'cell phone': 67,
            'book': 73,
            'box': 39
        }
        
        # Subscribers
        self.create_subscription(CameraInfo, '/camera/color/camera_info', self.info_callback, 10)
        self.create_subscription(Image, '/camera/depth/image_rect_raw', self.depth_callback, 10)
        self.create_subscription(Image, '/camera/color/image_raw', self.color_callback, 10)
        self.create_subscription(String, '/arm/set_target', self.set_target_callback, 10)
        
        # Publishers
        self.det_pub = self.create_publisher(Detection2DArray, '/arm/detections', 10)
        self.target_pub = self.create_publisher(PointStamped, '/arm/target_point', 10)
        self.img_pub = self.create_publisher(Image, '/arm/detection_image', 10)
        self.surf_pub = self.create_publisher(Float64, '/arm/surface_depth', 10)

    def _load_model(self):
        engine_path = self.get_parameter('model_path').value  # .engine
        pt_path = self.get_parameter('model_fallback_path').value  # .pt
        
        if os.path.exists(engine_path):
            self.get_logger().info(f'Loading TensorRT engine: {engine_path}')
            self.model = YOLO(engine_path, task='segment')
        elif os.path.exists(pt_path):
            self.get_logger().warn(f'TensorRT engine not found, using PyTorch: {pt_path}')
            self.model = YOLO(pt_path)
            # Auto-export TensorRT engine for next time
            self.get_logger().info('Exporting TensorRT engine (first-time, may take ~5 min)...')
            self.model.export(format='engine', imgsz=640, device='cuda:0', half=True)
            self.model = YOLO(engine_path, task='segment')
        else:
            self.get_logger().error(f'No model found at {engine_path} or {pt_path}')
            raise FileNotFoundError(f'YOLO model not found')

    def set_target_callback(self, msg: String):
        label = msg.data.lower().strip()
        self.target_label = label
        if label in self.class_name_map:
            self.target_class_id = self.class_name_map[label]
            self.get_logger().info(f"Target object set to '{label}' (Class ID: {self.target_class_id})")
        else:
            self.get_logger().warn(f"Target label '{label}' not in standard map, defaulting to bottle (ID: 39)")
            self.target_class_id = 39

    def info_callback(self, msg):
        self.latest_camera_info = msg
        
    def depth_callback(self, msg):
        self.latest_depth = msg

    def color_callback(self, msg):
        if self.latest_camera_info is None or self.latest_depth is None:
            return
            
        # Decode color image
        if msg.encoding == 'yuyv' or msg.encoding == 'yuv422_yuy2':
            raw = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, 2)
            bgr = cv2.cvtColor(raw, cv2.COLOR_YUV2BGR_YUYV)
        elif msg.encoding == 'rgb8':
            bgr = cv2.cvtColor(np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, 3), cv2.COLOR_RGB2BGR)
        elif msg.encoding == 'bgr8':
            bgr = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, 3)
        else:
            bgr = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            
        # Decode depth image
        try:
            depth_img = self.bridge.imgmsg_to_cv2(self.latest_depth, desired_encoding='passthrough')
        except Exception as e:
            self.get_logger().error(f'Depth decode error: {e}')
            return
            
        # Inference
        results = self.model(bgr, verbose=False, conf=self.conf_thresh)
        if len(results) == 0:
            self._publish_empty(msg.header)
            return
            
        result = results[0]
        boxes = result.boxes
        masks = result.masks
        
        if boxes is None or len(boxes) == 0:
            self._publish_empty(msg.header)
            self._publish_debug_image(bgr, result, msg.header)
            return

        det_array = Detection2DArray()
        det_array.header = msg.header
        
        best_target = None
        best_conf = -1.0
        best_point = None
        best_surf = None
        
        # Process detections
        for i in range(len(boxes)):
            box = boxes[i]
            cls_id = int(box.cls[0].item())
            conf = float(box.conf[0].item())
            
            # Create Detection2D
            xywh = box.xywh[0].cpu().numpy() # cx, cy, w, h
            det = Detection2D()
            det.header = msg.header
            det.bbox.center.position.x = float(xywh[0])
            det.bbox.center.position.y = float(xywh[1])
            det.bbox.size_x = float(xywh[2])
            det.bbox.size_y = float(xywh[3])
            
            hyp = ObjectHypothesisWithPose()
            if hasattr(hyp, 'hypothesis'): # ROS 2 Humble
                hyp.hypothesis.class_id = str(cls_id)
                hyp.hypothesis.score = conf
            else: # ROS 2 Foxy fallback
                hyp.id = str(cls_id)
                hyp.score = conf
                
            det.results.append(hyp)
            det_array.detections.append(det)
            
            # Check if this is a better target
            if cls_id == self.target_class_id and conf > best_conf:
                if masks is not None and len(masks) > i:
                    mask_polygon = masks[i].xy[0] # polygon coordinates
                    if len(mask_polygon) > 2:
                        depth_mm = self._get_mask_median_depth(mask_polygon, depth_img)
                        if depth_mm is not None:
                            depth_m = depth_mm / 1000.0
                            pt3d = self._deproject_pixel_to_point(xywh[0], xywh[1], depth_m, self.latest_camera_info)
                            best_target = pt3d
                            best_conf = conf
                            best_surf = self._surface_proximity_depth(int(xywh[0]), int(xywh[1]), depth_img)

        # Publish detections
        self.det_pub.publish(det_array)
        
        # Publish target point and surface depth
        if best_target is not None:
            pt_msg = PointStamped()
            pt_msg.header = msg.header
            pt_msg.header.frame_id = 'd405_color_optical_frame'
            pt_msg.point.x = best_target[0]
            pt_msg.point.y = best_target[1]
            pt_msg.point.z = best_target[2]
            self.target_pub.publish(pt_msg)
            
        if best_surf is not None:
            surf_msg = Float64()
            surf_msg.data = best_surf
            self.surf_pub.publish(surf_msg)
            
        # Publish debug image
        self._publish_debug_image(bgr, result, msg.header)

    def _publish_empty(self, header):
        det_array = Detection2DArray()
        det_array.header = header
        self.det_pub.publish(det_array)

    def _publish_debug_image(self, bgr, result, header):
        if self.img_pub.get_subscription_count() > 0:
            annotated_frame = result.plot()
            img_msg = self.bridge.cv2_to_imgmsg(annotated_frame, encoding="bgr8")
            img_msg.header = header
            self.img_pub.publish(img_msg)

    def _deproject_pixel_to_point(self, u, v, depth_m, camera_info):
        """Deproject pixel (u,v) + depth to 3D point using pinhole model."""
        fx = camera_info.k[0]
        fy = camera_info.k[4]
        cx = camera_info.k[2]
        cy = camera_info.k[5]
        x = (u - cx) * depth_m / fx
        y = (v - cy) * depth_m / fy
        z = depth_m
        return (x, y, z)

    def _get_mask_median_depth(self, mask_polygon, depth_image, depth_scale=0.001):
        """Sample depth ONLY within segmentation mask, filter noise."""
        h, w = depth_image.shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillPoly(mask, [mask_polygon.astype(np.int32)], 255)
        
        # Check mask size (reject >15% of frame)
        n_px = np.sum(mask == 255)
        max_px = int(h * w * 0.15)
        if n_px > max_px:
            return None  # Mask too large, likely floor/background
        
        depths = depth_image[mask == 255].astype(float) * depth_scale * 1000.0  # mm
        valid = depths[(depths > 70.0) & (depths < 600.0)]  # D405 range guards
        if valid.size < 20:
            return None
        return float(np.median(valid))  # mm

    def _surface_proximity_depth(self, obj_px, obj_py, depth_image, 
                                  depth_scale=0.001, inner_r=35, outer_r=80):
        """Annular ring depth around object centroid to detect floor/table surface."""
        h, w = depth_image.shape[:2]
        ys, xs = np.ogrid[-outer_r:outer_r+1, -outer_r:outer_r+1]
        r2 = xs**2 + ys**2
        ring_mask = (r2 >= inner_r**2) & (r2 <= outer_r**2)
        
        row0, col0 = max(0, obj_py - outer_r), max(0, obj_px - outer_r)
        row1, col1 = min(h, obj_py + outer_r + 1), min(w, obj_px + outer_r + 1)
        
        rm_crop = ring_mask[
            (row0 - (obj_py - outer_r)):(row1 - (obj_py - outer_r)),
            (col0 - (obj_px - outer_r)):(col1 - (obj_px - outer_r))
        ]
        patch = depth_image[row0:row1, col0:col1]
        if patch.shape != rm_crop.shape:
            return None
        
        depths = patch[rm_crop].astype(float) * depth_scale * 1000.0
        valid = depths[(depths > 70.0) & (depths < 600.0)]
        if valid.size < 20:
            return None
        return float(np.median(valid))

def main(args=None):
    rclpy.init(args=args)
    node = DetectionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
