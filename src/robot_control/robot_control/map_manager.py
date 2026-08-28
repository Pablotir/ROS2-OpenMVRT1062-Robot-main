#!/usr/bin/env python3
"""
map_manager.py — Lifelong Map Lifecycle & Session Versioning
============================================================
Manages versioned SLAM map sessions, occupancy grid exports (PGM/YAML),
pose graph serialization (.posegraph/.data), and atomic `latest` symlinks.

Directory Structure:
  /root/maps/
    ├── latest -> 20260825_203000/    (atomic symlink to newest session)
    ├── 20260825_203000/
    │   ├── exploration_map.pgm       (P5 binary grayscale occupancy grid)
    │   ├── exploration_map.yaml      (Map server metadata)
    │   ├── exploration_posegraph.posegraph (SLAM Toolbox pose graph)
    │   ├── exploration_posegraph.data      (SLAM Toolbox serialized scan data)
    │   └── metadata.json             (Session details, bounding box, stats)
    └── 20260824_181500/
        └── ...

Features:
  - Atomic symlink creation for zero-downtime map swapping
  - Cross-session map persistence mounted directly into Docker
  - Pure Python library API + standalone ROS 2 Node & CLI
"""

import os
import sys
import json
import time
import shutil
import subprocess
from datetime import datetime
from typing import Optional, List, Dict, Any, Tuple

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy
from nav_msgs.msg import OccupancyGrid
from std_srvs.srv import Trigger


# Default directories inside Docker container & host fallbacks
DEFAULT_MAP_DIRS = [
    '/root/maps',                  # Primary Docker volume mount (./maps)
    '/root/ros2_ws/maps',          # Container workspace fallback
    os.path.expanduser('~/maps'),  # User home fallback
    './maps'                       # Local workspace relative path
]


# ═════════════════════════════════════════════════════════════════════════════
# CORE LIBRARY FUNCTIONS (No ROS 2 Runtime Dependency)
# ═════════════════════════════════════════════════════════════════════════════

def get_maps_base_dir(custom_path: Optional[str] = None) -> str:
    """Resolve and ensure the base maps storage directory exists."""
    if custom_path and str(custom_path).strip():
        base_dir = os.path.abspath(os.path.expanduser(custom_path))
    else:
        env_dir = os.environ.get('MAPS_DIR')
        if env_dir:
            base_dir = os.path.abspath(os.path.expanduser(env_dir))
        else:
            # Pick first available or default to primary
            base_dir = DEFAULT_MAP_DIRS[0]

    os.makedirs(base_dir, exist_ok=True)
    return base_dir


def create_session_directory(base_dir: Optional[str] = None,
                             timestamp_str: Optional[str] = None) -> str:
    """Create a new timestamped map session directory (YYYYMMDD_HHMMSS)."""
    parent = get_maps_base_dir(base_dir)
    if not timestamp_str:
        timestamp_str = datetime.now().strftime('%Y%m%d_%H%M%S')

    session_dir = os.path.join(parent, timestamp_str)
    os.makedirs(session_dir, exist_ok=True)
    return session_dir


def update_latest_symlink(session_dir: str, base_dir: Optional[str] = None) -> bool:
    """Atomically update /root/maps/latest to point to session_dir."""
    parent = get_maps_base_dir(base_dir)
    latest_link = os.path.join(parent, 'latest')
    tmp_link = os.path.join(parent, f'.latest_tmp_{os.getpid()}')

    # Use relative symlink target so the directory remains portable
    rel_target = os.path.relpath(session_dir, parent)

    try:
        if os.path.lexists(tmp_link):
            if os.path.islink(tmp_link) or os.path.isfile(tmp_link):
                os.remove(tmp_link)
            elif os.path.isdir(tmp_link):
                shutil.rmtree(tmp_link)

        # Create temporary symlink
        os.symlink(rel_target, tmp_link)

        # Atomic rename on POSIX (Windows fallbacks to remove+rename)
        if hasattr(os, 'replace'):
            os.replace(tmp_link, latest_link)
        else:
            if os.path.lexists(latest_link):
                if os.path.islink(latest_link) or os.path.isfile(latest_link):
                    os.remove(latest_link)
                elif os.path.isdir(latest_link):
                    shutil.rmtree(latest_link)
            os.rename(tmp_link, latest_link)

        # Write text pointer for non-symlink tools/viewers
        pointer_file = os.path.join(parent, 'latest_session.txt')
        with open(pointer_file, 'w') as f:
            f.write(f"{os.path.basename(session_dir)}\n")

        return True
    except Exception as e:
        # Fallback for Windows environments where symlink privileges might be restricted
        try:
            pointer_file = os.path.join(parent, 'latest_session.txt')
            with open(pointer_file, 'w') as f:
                f.write(f"{os.path.basename(session_dir)}\n")
            return True
        except Exception:
            return False


def save_occupancy_grid(map_msg: OccupancyGrid,
                        target_dir: str,
                        base_name: str = 'exploration_map') -> Tuple[str, str]:
    """Write ROS 2 OccupancyGrid to binary PGM (P5) and YAML map description."""
    os.makedirs(target_dir, exist_ok=True)
    w, h = map_msg.info.width, map_msg.info.height
    res = map_msg.info.resolution
    ox = map_msg.info.origin.position.x
    oy = map_msg.info.origin.position.y

    # 1. Binary Grayscale PGM (P5)
    pgm_filename = f"{base_name}.pgm"
    pgm_path = os.path.join(target_dir, pgm_filename)
    with open(pgm_path, 'wb') as f:
        f.write(f'P5\n{w} {h}\n255\n'.encode('ascii'))
        # OccupancyGrid row 0 is bottom; PGM row 0 is top → flip vertically
        for row in range(h - 1, -1, -1):
            row_data = bytearray(w)
            for col in range(w):
                val = map_msg.data[row * w + col]
                if val == -1:       # Unknown
                    row_data[col] = 205
                elif val == 0:      # Free space
                    row_data[col] = 254
                else:               # Occupied obstacle (100)
                    row_data[col] = 0
            f.write(row_data)

    # 2. YAML Metadata
    yaml_filename = f"{base_name}.yaml"
    yaml_path = os.path.join(target_dir, yaml_filename)
    with open(yaml_path, 'w') as f:
        f.write(f"image: {pgm_filename}\n")
        f.write(f"resolution: {res:.4f}\n")
        f.write(f"origin: [{ox:.4f}, {oy:.4f}, 0.0]\n")
        f.write("negate: 0\n")
        f.write("occupied_thresh: 0.65\n")
        f.write("free_thresh: 0.196\n")

    return pgm_path, yaml_path


def save_session_metadata(target_dir: str,
                          metadata: Dict[str, Any]) -> str:
    """Save metadata.json documenting map session metrics and provenance."""
    os.makedirs(target_dir, exist_ok=True)
    meta_path = os.path.join(target_dir, 'metadata.json')

    default_meta = {
        'created_at': datetime.now().isoformat(),
        'session_dir': target_dir,
        'version': '2.0.0',
    }
    merged = {**default_meta, **metadata}

    with open(meta_path, 'w') as f:
        json.dump(merged, f, indent=2)

    return meta_path


def request_serialize_posegraph(target_dir: str,
                                base_name: str = 'exploration_posegraph',
                                logger=None) -> str:
    """Request SLAM Toolbox to serialize the pose graph into target_dir."""
    os.makedirs(target_dir, exist_ok=True)
    posegraph_prefix = os.path.join(target_dir, base_name)

    # SLAM Toolbox serialize_map service requires forward-slashes even on Windows
    norm_prefix = posegraph_prefix.replace('\\', '/')

    cmd = [
        'ros2', 'service', 'call',
        '/slam_toolbox/serialize_map',
        'slam_toolbox/srv/SerializePoseGraph',
        f"{{filename: '{norm_prefix}'}}"
    ]

    try:
        subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        if logger:
            logger.info(f"Pose graph serialization triggered: {posegraph_prefix}")
    except Exception as e:
        if logger:
            logger.warn(f"Failed to trigger pose graph serialization: {e}")

    return posegraph_prefix


def list_saved_maps(base_dir: Optional[str] = None) -> List[Dict[str, Any]]:
    """List all saved map sessions with status flags and timestamps."""
    parent = get_maps_base_dir(base_dir)
    if not os.path.isdir(parent):
        return []

    sessions = []
    latest_target = None
    latest_link = os.path.join(parent, 'latest')

    if os.path.islink(latest_link):
        try:
            target = os.readlink(latest_link)
            latest_target = os.path.basename(target)
        except Exception:
            pass

    for entry in sorted(os.listdir(parent)):
        full_path = os.path.join(parent, entry)
        if not os.path.isdir(full_path) or entry == 'latest' or entry.startswith('.'):
            continue

        pgm_exists = os.path.exists(os.path.join(full_path, 'exploration_map.pgm'))
        yaml_exists = os.path.exists(os.path.join(full_path, 'exploration_map.yaml'))
        posegraph_exists = os.path.exists(os.path.join(full_path, 'exploration_posegraph.posegraph')) or \
                           os.path.exists(os.path.join(full_path, 'exploration_posegraph.data'))

        meta_file = os.path.join(full_path, 'metadata.json')
        meta_data = {}
        if os.path.exists(meta_file):
            try:
                with open(meta_file, 'r') as f:
                    meta_data = json.load(f)
            except Exception:
                pass

        sessions.append({
            'name': entry,
            'path': full_path,
            'is_latest': (entry == latest_target),
            'has_pgm': pgm_exists,
            'has_yaml': yaml_exists,
            'has_posegraph': posegraph_exists,
            'created_at': meta_data.get('created_at', entry),
            'metadata': meta_data
        })

    return sessions


def get_latest_map_session(base_dir: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """Resolve the active/latest map session directory."""
    parent = get_maps_base_dir(base_dir)
    latest_link = os.path.join(parent, 'latest')

    if os.path.exists(latest_link):
        real_path = os.path.realpath(latest_link)
        if os.path.isdir(real_path):
            return {
                'name': os.path.basename(real_path),
                'path': real_path,
                'posegraph_prefix': os.path.join(real_path, 'exploration_posegraph'),
                'yaml_path': os.path.join(real_path, 'exploration_map.yaml'),
                'pgm_path': os.path.join(real_path, 'exploration_map.pgm'),
            }

    # Fallback to newest session folder
    maps = list_saved_maps(parent)
    if maps:
        newest = maps[-1]
        real_path = newest['path']
        return {
            'name': newest['name'],
            'path': real_path,
            'posegraph_prefix': os.path.join(real_path, 'exploration_posegraph'),
            'yaml_path': os.path.join(real_path, 'exploration_map.yaml'),
            'pgm_path': os.path.join(real_path, 'exploration_map.pgm'),
        }

    return None


# ═════════════════════════════════════════════════════════════════════════════
# ROS 2 NODE INTERFACE
# ═════════════════════════════════════════════════════════════════════════════

class MapManagerNode(Node):
    """ROS 2 Node providing runtime map management and on-demand saving services."""

    def __init__(self):
        super().__init__('map_manager')

        self.declare_parameter('maps_dir', '/root/maps')
        self._maps_dir = get_maps_base_dir(self.get_parameter('maps_dir').value)

        self._latest_map: Optional[OccupancyGrid] = None

        # Subscribe to /map (Transient Local QoS for latching)
        map_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )
        self.create_subscription(OccupancyGrid, '/map', self._on_map, map_qos)

        # Services
        self.create_service(Trigger, '/map_manager/save_map', self._srv_save_map)
        self.create_service(Trigger, '/map_manager/get_latest', self._srv_get_latest)

        self.get_logger().info(f"Map Manager ready | Base directory: {self._maps_dir}")

    def _on_map(self, msg: OccupancyGrid):
        self._latest_map = msg

    def _srv_save_map(self, request, response):
        """Service callback to save current map into a new versioned session."""
        if self._latest_map is None:
            response.success = False
            response.message = "No /map data received yet to save."
            return response

        session_dir = create_session_directory(self._maps_dir)
        pgm_path, yaml_path = save_occupancy_grid(self._latest_map, session_dir)
        posegraph_prefix = request_serialize_posegraph(session_dir, logger=self.get_logger())

        meta = {
            'width': self._latest_map.info.width,
            'height': self._latest_map.info.height,
            'resolution': self._latest_map.info.resolution,
            'origin': [
                self._latest_map.info.origin.position.x,
                self._latest_map.info.origin.position.y,
                0.0
            ],
            'saved_pgm': os.path.basename(pgm_path),
            'saved_yaml': os.path.basename(yaml_path),
            'saved_posegraph': os.path.basename(posegraph_prefix)
        }
        save_session_metadata(session_dir, meta)
        update_latest_symlink(session_dir, self._maps_dir)

        response.success = True
        response.message = f"Saved map session: {os.path.basename(session_dir)}"
        self.get_logger().info(f"Map session saved and symlinked -> {session_dir}")
        return response

    def _srv_get_latest(self, request, response):
        latest = get_latest_map_session(self._maps_dir)
        if latest:
            response.success = True
            response.message = json.dumps(latest)
        else:
            response.success = False
            response.message = "No saved map sessions found."
        return response


def main(args=None):
    if len(sys.argv) > 1 and sys.argv[1] == '--list':
        base = get_maps_base_dir()
        sessions = list_saved_maps(base)
        print(f"\nSaved Maps in {base}:")
        print("─" * 60)
        for s in sessions:
            flag = " [LATEST]" if s['is_latest'] else ""
            pgm_flag = "PGM" if s['has_pgm'] else "---"
            pg_flag = "POSEGRAPH" if s['has_posegraph'] else "---------"
            print(f" • {s['name']}{flag} ({pgm_flag} | {pg_flag}) -> {s['path']}")
        print("─" * 60)
        return

    rclpy.init(args=args)
    node = MapManagerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
