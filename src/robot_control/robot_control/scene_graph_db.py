#!/usr/bin/env python3
"""
scene_graph_db.py — SQLite Semantic Scene Graph Database Engine
===============================================================
Implements persistent storage for Layer 1 of the Modular OVMM framework.
Stores segmented topological room regions, centroids, bounding polygons,
landmark detections, and doorway choke points.

Database Location:
  /root/maps/latest/scene_graph.db (or configured map directory)

Tables:
  - `regions`: Topological room clusters (id, label, centroid, polygon, area, confidence)
  - `landmarks`: Recognized semantic objects linked to regions (id, region_id, class_name, bbox, snapshot_path)
  - `chokepoints`: Detected doorways and traversable choke boundaries (id, region_a, region_b, x, y, width)
"""

import os
import json
import sqlite3
from datetime import datetime
from typing import Optional, List, Dict, Any, Tuple


class SceneGraphDB:
    """Thread-safe SQLite database manager for semantic scene graphs."""

    def __init__(self, db_path: str):
        if db_path == ':memory:':
            self.db_path = ':memory:'
            self._persistent_conn = sqlite3.connect(':memory:')
            self._persistent_conn.row_factory = sqlite3.Row
        else:
            self.db_path = os.path.abspath(os.path.expanduser(db_path))
            os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
            self._persistent_conn = None

        self._init_tables()

    def _get_connection(self) -> sqlite3.Connection:
        if self._persistent_conn:
            return self._persistent_conn
        conn = sqlite3.connect(self.db_path, timeout=10.0)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_tables(self):
        with self._get_connection() as conn:
            cursor = conn.cursor()

            # 1. Regions table (Topological room clusters)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS regions (
                    region_id INTEGER PRIMARY KEY,
                    label TEXT DEFAULT 'unlabeled',
                    centroid_x REAL NOT NULL,
                    centroid_y REAL NOT NULL,
                    area_sqm REAL DEFAULT 0.0,
                    polygon_json TEXT NOT NULL,
                    confidence REAL DEFAULT 0.0,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
            """)

            # 2. Landmarks table (Objects recognized inside rooms)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS landmarks (
                    landmark_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    region_id INTEGER NOT NULL,
                    class_name TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    bbox_json TEXT NOT NULL,
                    snapshot_path TEXT,
                    detected_at TEXT NOT NULL,
                    FOREIGN KEY (region_id) REFERENCES regions (region_id) ON DELETE CASCADE
                )
            """)

            # 3. Chokepoints table (Doorways & bottlenecks connecting regions)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS chokepoints (
                    door_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    region_a INTEGER,
                    region_b INTEGER,
                    x REAL NOT NULL,
                    y REAL NOT NULL,
                    width_m REAL NOT NULL,
                    detected_at TEXT NOT NULL
                )
            """)

            conn.commit()

    # ── Region Operations ───────────────────────────────────────────────────

    def upsert_region(self,
                      region_id: int,
                      centroid_x: float,
                      centroid_y: float,
                      polygon_coords: List[Tuple[float, float]],
                      label: str = 'unlabeled',
                      area_sqm: float = 0.0,
                      confidence: float = 0.0) -> int:
        """Insert or update a topological region."""
        now = datetime.now().isoformat()
        polygon_json = json.dumps([[float(x), float(y)] for x, y in polygon_coords])

        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO regions (region_id, label, centroid_x, centroid_y, area_sqm, polygon_json, confidence, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(region_id) DO UPDATE SET
                    centroid_x = excluded.centroid_x,
                    centroid_y = excluded.centroid_y,
                    area_sqm = excluded.area_sqm,
                    polygon_json = excluded.polygon_json,
                    label = CASE WHEN excluded.confidence > regions.confidence THEN excluded.label ELSE regions.label END,
                    confidence = MAX(regions.confidence, excluded.confidence),
                    updated_at = excluded.updated_at
            """, (region_id, label, float(centroid_x), float(centroid_y), float(area_sqm), polygon_json, float(confidence), now, now))
            conn.commit()
            return region_id

    def update_region_label(self, region_id: int, label: str, confidence: float = 1.0) -> bool:
        """Update the semantic room label for a specific region."""
        now = datetime.now().isoformat()
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                UPDATE regions
                SET label = ?, confidence = ?, updated_at = ?
                WHERE region_id = ?
            """, (label, float(confidence), now, int(region_id)))
            conn.commit()
            return cursor.rowcount > 0

    def get_region(self, region_id: int) -> Optional[Dict[str, Any]]:
        """Fetch region by its ID."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM regions WHERE region_id = ?", (int(region_id),))
            row = cursor.fetchone()
            if row:
                res = dict(row)
                res['polygon'] = json.loads(res['polygon_json'])
                return res
            return None

    def get_all_regions(self) -> List[Dict[str, Any]]:
        """Fetch all topological regions."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM regions ORDER BY region_id ASC")
            results = []
            for row in cursor.fetchall():
                r = dict(row)
                r['polygon'] = json.loads(r['polygon_json'])
                results.append(r)
            return results

    def find_region_by_label(self, label: str) -> Optional[Dict[str, Any]]:
        """Find the most confident region matching a semantic label (case-insensitive substring)."""
        target = f"%{label.lower().strip()}%"
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT * FROM regions 
                WHERE LOWER(label) LIKE ? 
                ORDER BY confidence DESC, area_sqm DESC 
                LIMIT 1
            """, (target,))
            row = cursor.fetchone()
            if row:
                res = dict(row)
                res['polygon'] = json.loads(res['polygon_json'])
                return res
            return None

    # ── Landmark Operations ─────────────────────────────────────────────────

    def add_landmark(self,
                     region_id: int,
                     class_name: str,
                     confidence: float,
                     bbox: List[float],
                     snapshot_path: Optional[str] = None) -> int:
        """Record an observed landmark inside a region."""
        now = datetime.now().isoformat()
        bbox_json = json.dumps([float(v) for v in bbox])

        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO landmarks (region_id, class_name, confidence, bbox_json, snapshot_path, detected_at)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (int(region_id), class_name.lower().strip(), float(confidence), bbox_json, snapshot_path, now))
            conn.commit()
            return cursor.lastrowid

    def get_landmarks_for_region(self, region_id: int) -> List[Dict[str, Any]]:
        """Get all recognized landmarks within a region."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM landmarks WHERE region_id = ? ORDER BY confidence DESC", (int(region_id),))
            results = []
            for row in cursor.fetchall():
                l = dict(row)
                l['bbox'] = json.loads(l['bbox_json'])
                results.append(l)
            return results

    # ── Chokepoint (Doorway) Operations ────────────────────────────────────

    def add_chokepoint(self, x: float, y: float, width_m: float, region_a: int = -1, region_b: int = -1) -> int:
        """Record a doorway / choke point connecting regions."""
        now = datetime.now().isoformat()
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO chokepoints (region_a, region_b, x, y, width_m, detected_at)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (int(region_a), int(region_b), float(x), float(y), float(width_m), now))
            conn.commit()
            return cursor.lastrowid

    def get_all_chokepoints(self) -> List[Dict[str, Any]]:
        """Fetch all detected doorways."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM chokepoints ORDER BY door_id ASC")
            return [dict(row) for row in cursor.fetchall()]

    # ── Spatial Point-In-Polygon Query ──────────────────────────────────────

    def get_region_for_pose(self, x: float, y: float) -> Optional[Dict[str, Any]]:
        """Determine which topological region contains the point (x, y)."""
        regions = self.get_all_regions()
        for r in regions:
            poly = r['polygon']
            if len(poly) >= 3 and _point_in_polygon(x, y, poly):
                return r
        return None


def _point_in_polygon(x: float, y: float, poly: List[List[float]]) -> bool:
    """Ray casting algorithm for 2D point-in-polygon test."""
    n = len(poly)
    inside = False
    p1x, p1y = poly[0]
    for i in range(1, n + 1):
        p2x, p2y = poly[i % n]
        if min(p1y, p2y) < y <= max(p1y, p2y):
            if x <= max(p1x, p2x):
                if p1y != p2y:
                    x_inters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                    if p1x == p2x or x <= x_inters:
                        inside = not inside
        p1x, p1y = p2x, p2y
    return inside
