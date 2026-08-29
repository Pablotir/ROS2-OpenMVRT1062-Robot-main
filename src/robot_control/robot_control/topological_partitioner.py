#!/usr/bin/env python3
"""
topological_partitioner.py — Geometric Voronoi & Morphological Map Segmenter
=============================================================================
Segments 2D OccupancyGrids into discrete topological room regions and detects
doorway choke points using morphological distance transforms and watershed
ridge detection.

Implements Layer 1 of the Modular OVMM framework.
"""

import math
import cv2
import numpy as np
from typing import List, Dict, Tuple, Optional, Any

try:
    from nav_msgs.msg import OccupancyGrid
except ImportError:
    OccupancyGrid = Any


class TopologicalRegionData:
    def __init__(self,
                 region_id: int,
                 centroid_x: float,
                 centroid_y: float,
                 area_sqm: float,
                 polygon_coords: List[Tuple[float, float]],
                 is_hallway: bool = False):
        self.region_id = region_id
        self.centroid_x = centroid_x
        self.centroid_y = centroid_y
        self.area_sqm = area_sqm
        self.polygon_coords = polygon_coords
        self.is_hallway = is_hallway


class TopologicalPartitioner:
    """Partitions a 2D SLAM grid into topological rooms and doorways."""

    def __init__(self,
                 min_room_area_sqm: float = 1.5,
                 max_door_width_m: float = 1.4,
                 choke_threshold_m: float = 0.55,
                 polygon_epsilon_m: float = 0.08):
        self.min_room_area_sqm = min_room_area_sqm
        self.max_door_width_m = max_door_width_m
        self.choke_threshold_m = choke_threshold_m
        self.polygon_epsilon_m = polygon_epsilon_m

    def partition_occupancy_grid(self,
                                 map_msg: OccupancyGrid) -> Tuple[List[TopologicalRegionData], List[Dict[str, Any]], np.ndarray]:
        """
        Segment the occupancy grid into regions.
        
        Returns:
          - regions: List of TopologicalRegionData
          - chokepoints: List of detected doorway dicts (x, y, width_m)
          - labeled_grid: 2D numpy array with integer region labels per pixel
        """
        w, h = map_msg.info.width, map_msg.info.height
        res = map_msg.info.resolution
        ox = map_msg.info.origin.position.x
        oy = map_msg.info.origin.position.y

        if w < 10 or h < 10:
            return [], [], np.zeros((h, w), dtype=np.int32)

        # 1. Reshape raw occupancy data to 2D numpy array (row, col)
        raw_grid = np.array(map_msg.data, dtype=np.int8).reshape((h, w))

        # Binary free space mask (0 = free, everything else = obstacle/unknown)
        free_mask = (raw_grid == 0).astype(np.uint8) * 255

        # Morphological opening to remove small sensor noise specks
        kernel_noise = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        free_clean = cv2.morphologyEx(free_mask, cv2.MORPH_OPEN, kernel_noise)

        # Count total free pixels
        total_free_px = np.count_nonzero(free_clean)
        min_room_px = int(self.min_room_area_sqm / (res * res))
        if total_free_px < min_room_px:
            return [], [], np.zeros((h, w), dtype=np.int32)

        # 2. Euclidean Distance Transform (distance to nearest occupied/unknown cell)
        dist_transform = cv2.distanceTransform(free_clean, cv2.DIST_L2, 5) * res  # in meters

        # 3. Find topological peaks (room centers / open space maxima)
        # Dilate distance transform to find local maxima
        peak_kernel_size = max(5, int(1.2 / res))  # ~1.2m footprint
        if peak_kernel_size % 2 == 0:
            peak_kernel_size += 1

        kernel_peak = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (peak_kernel_size, peak_kernel_size))
        dist_dilated = cv2.dilate(dist_transform, kernel_peak)

        # A pixel is a local maximum if its distance value equals the dilated neighborhood
        local_max = (dist_transform == dist_dilated) & (dist_transform > self.choke_threshold_m)

        # Connect nearby maxima
        num_markers, markers = cv2.connectedComponents(local_max.astype(np.uint8))

        if num_markers <= 1:
            # Single open space without sub-partitions
            markers = (free_clean > 0).astype(np.int32)
            num_markers = 2

        # 4. Watershed Segmentation on the Distance Transform
        # Invert distance transform for watershed (peaks become basins)
        dist_uint8 = np.clip(255 - (dist_transform / np.max(dist_transform) * 255), 0, 255).astype(np.uint8)
        dist_bgr = cv2.cvtColor(dist_uint8, cv2.COLOR_GRAY2BGR)

        # Markers input for watershed must have background = 0, peaks = 1..N
        markers_ws = markers.copy()
        # Mark unknown/walls as distinct boundary (-1)
        markers_ws[free_clean == 0] = 0

        cv2.watershed(dist_bgr, markers_ws)

        # 5. Extract Topological Regions and Polygons
        regions: List[TopologicalRegionData] = []
        unique_labels = np.unique(markers_ws)

        region_idx = 1
        for label in unique_labels:
            if label <= 0:  # Skip background (0) and watershed boundary (-1)
                continue

            region_mask = (markers_ws == label).astype(np.uint8) * 255
            area_px = np.count_nonzero(region_mask)
            area_sqm = area_px * (res * res)

            if area_sqm < self.min_room_area_sqm:
                continue

            # Find outer contours of this region
            contours, _ = cv2.findContours(region_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not contours:
                continue

            # Take the largest contour for this region
            main_contour = max(contours, key=cv2.contourArea)

            # Polygon approximation
            epsilon_px = self.polygon_epsilon_m / res
            approx = cv2.approxPolyDP(main_contour, epsilon_px, True)

            # Convert contour coordinates (col, row) to map frame (x, y)
            polygon_coords: List[Tuple[float, float]] = []
            for pt in approx:
                c, r = pt[0]
                world_x = float(c * res + ox)
                world_y = float(r * res + oy)
                polygon_coords.append((world_x, world_y))

            # Compute Region Centroid
            M = cv2.moments(main_contour)
            if M["m00"] > 0:
                cx_px = M["m10"] / M["m00"]
                cy_px = M["m01"] / M["m00"]
            else:
                cx_px, cy_px = approx[0][0]

            centroid_x = float(cx_px * res + ox)
            centroid_y = float(cy_px * res + oy)

            # Detect if this region has hallway geometry (elongated: length > 3 * width and width < 1.4m)
            rect = cv2.minAreaRect(main_contour)
            rw, rh = rect[1]
            min_dim_m = min(rw, rh) * res
            max_dim_m = max(rw, rh) * res
            is_hallway = (min_dim_m < 1.4) and (max_dim_m > 3.0 * min_dim_m)

            regions.append(TopologicalRegionData(
                region_id=region_idx,
                centroid_x=centroid_x,
                centroid_y=centroid_y,
                area_sqm=float(area_sqm),
                polygon_coords=polygon_coords,
                is_hallway=is_hallway
            ))
            region_idx += 1

        # 6. Detect Chokepoints (Doorways on watershed boundaries between distinct regions)
        chokepoints: List[Dict[str, Any]] = []
        boundary_mask = (markers_ws == -1) & (free_clean > 0)

        # Look for constriction points where distance transform is small (< max_door_width_m / 2)
        door_candidates = boundary_mask & (dist_transform < (self.max_door_width_m / 2.0)) & (dist_transform > 0.15)
        num_doors, door_labels = cv2.connectedComponents(door_candidates.astype(np.uint8))

        for d_id in range(1, num_doors):
            pts = np.where(door_labels == d_id)
            if len(pts[0]) < 3:
                continue

            mean_r = float(np.mean(pts[0]))
            mean_c = float(np.mean(pts[1]))

            door_x = float(mean_c * res + ox)
            door_y = float(mean_r * res + oy)

            # Estimate doorway width from distance transform clearance
            max_clearance = float(np.max(dist_transform[pts]) * 2.0)
            chokepoints.append({
                'x': door_x,
                'y': door_y,
                'width_m': min(self.max_door_width_m, max(0.6, max_clearance))
            })

        return regions, chokepoints, markers_ws
