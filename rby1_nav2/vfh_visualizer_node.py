#!/usr/bin/env python3
"""Sensor-only VFH visualizer node with rotation safety check."""

from __future__ import annotations

import math
import time
from typing import List, Optional, Sequence, Tuple

import numpy as np
import rclpy
from geometry_msgs.msg import Point, PoseStamped, Twist
from nav_msgs.msg import Odometry
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rclpy.time import Time
from sensor_msgs.msg import LaserScan, PointCloud2
from sensor_msgs_py import point_cloud2
from std_msgs.msg import Float32MultiArray, Header, String
from tf2_ros import Buffer, TransformException, TransformListener
from tf_transformations import euler_from_quaternion, quaternion_from_euler, quaternion_matrix
from visualization_msgs.msg import Marker, MarkerArray


class VFHVisualizerNode(Node):
    """Visual-only VFH node with rotation collision check."""

    def __init__(self) -> None:
        super().__init__("vfh_visualizer_node")

        # Core VFH parameters
        self.declare_parameter("sector_angle", 5.0)
        self.declare_parameter("safety_margin", 0.05)
        self.declare_parameter("min_turning_radius", 0.4)
        self.declare_parameter("max_linear_speed", 0.2)
        self.declare_parameter("max_reverse_speed", 0.0)
        self.declare_parameter("min_command_linear_speed", 0.03)
        self.declare_parameter("max_angular_speed", 0.5)
        self.declare_parameter("min_obstacle_dist", 0.3)
        self.declare_parameter("repulse_radius", 0.8)
        self.declare_parameter("rear_lookahead_distance", 1.2)
        self.declare_parameter("binary_threshold_low", 0.08)
        self.declare_parameter("binary_threshold_high", 0.15)
        self.declare_parameter("wide_valley_sectors", 16)
        self.declare_parameter("trajectory_check_step_deg", 5.0)
        self.declare_parameter("histogram_inflation_radius", 0.0)
        self.declare_parameter("show_preferred_arrow", False)

        # Direction preference
        self.declare_parameter("preferred_angle_deg", 0.0)
        self.declare_parameter("mu_pref", 4.0)
        self.declare_parameter("mu_heading", 2.0)
        self.declare_parameter("mu_prev", 2.0)
        self.declare_parameter("mu_density", 0.8)
        self.declare_parameter("density_speed_factor", 0.5)
        self.declare_parameter("density_speed_min_scale", 0.6)
        self.declare_parameter("density_speed_window", 2)
        self.declare_parameter("reverse_fallback_required_count", 3)
        self.declare_parameter("forward_fallback_required_count", 2)
        self.declare_parameter("fallback_commit_duration", 1.0)
        self.declare_parameter("front_recovery_required_count", 3)
        self.declare_parameter("front_recovery_clearance", 0.45)

        # Rotation safety
        self.declare_parameter("rotation_check_step_deg", 3.0)

        # IO parameters
        self.declare_parameter("scan_topic", "/scan_merged")
        self.declare_parameter("odom_topic", "/odom")
        self.declare_parameter("pointcloud_topic", "/livox/lidar")
        self.declare_parameter("goal_pose_topic", "/goal_pose")
        self.declare_parameter("use_goal_pose_topic", True)
        self.declare_parameter("use_pointcloud", True)
        self.declare_parameter("pointcloud_max_points", 3000)
        self.declare_parameter("pointcloud_min_range", 0.05)
        self.declare_parameter("pointcloud_max_range", 5.0)
        self.declare_parameter("pointcloud_filter_floor", True)
        self.declare_parameter("pointcloud_floor_z_max", -0.10)
        self.declare_parameter("pointcloud_filter_body", True)
        self.declare_parameter("noise_filter_enabled", True)
        self.declare_parameter("noise_filter_min_points", 8)
        self.declare_parameter("noise_filter_neighbor_radius", 0.12)
        self.declare_parameter("noise_filter_min_neighbors", 1)
        self.declare_parameter("clearance_lateral_margin", 0.10)
        self.declare_parameter(
            "self_filter_footprint",
            [0.097, -0.30, 0.097, 0.30, -0.260, 0.30, -0.563, 0.15, -0.563, -0.15, -0.260, -0.30],
        )
        self.declare_parameter("pointcloud_use_tf_fallback", False)
        self.declare_parameter("pointcloud_tf_fallback_source_frame", "livox_lidar")
        self.declare_parameter("pointcloud_tf_fallback_x", -0.265442083)
        self.declare_parameter("pointcloud_tf_fallback_y", 0.002229000)
        self.declare_parameter("pointcloud_tf_fallback_z", 1.389722440)
        self.declare_parameter("pointcloud_tf_fallback_roll", -3.136562568)
        self.declare_parameter("pointcloud_tf_fallback_pitch", 0.013624321)
        self.declare_parameter("pointcloud_tf_fallback_yaw", -0.003078160)
        self.declare_parameter("debug_base_frame", "base_nav")
        self.declare_parameter("debug_topic_prefix", "/vfh_viz")
        self.declare_parameter("update_rate", 10.0)

        # Load parameters
        self.sector_angle_deg = max(1.0, float(self.get_parameter("sector_angle").value))
        self.safety_margin = float(self.get_parameter("safety_margin").value)
        self.min_turning_radius = float(self.get_parameter("min_turning_radius").value)
        self.max_linear_speed = float(self.get_parameter("max_linear_speed").value)
        max_reverse_speed = float(self.get_parameter("max_reverse_speed").value)
        self.max_reverse_speed = abs(max_reverse_speed) if max_reverse_speed > 0.0 else self.max_linear_speed
        self.min_command_linear_speed = abs(float(self.get_parameter("min_command_linear_speed").value))
        self.max_angular_speed = float(self.get_parameter("max_angular_speed").value)
        self.min_obstacle_dist = float(self.get_parameter("min_obstacle_dist").value)
        self.repulse_radius = float(self.get_parameter("repulse_radius").value)
        self.rear_lookahead_distance_param = float(self.get_parameter("rear_lookahead_distance").value)
        self.binary_threshold_low = max(0.0, float(self.get_parameter("binary_threshold_low").value))
        self.binary_threshold_high = max(self.binary_threshold_low, float(self.get_parameter("binary_threshold_high").value))
        self.wide_valley_sectors = max(2, int(self.get_parameter("wide_valley_sectors").value))
        self.trajectory_check_step = math.radians(
            max(1.0, float(self.get_parameter("trajectory_check_step_deg").value))
        )
        self.histogram_inflation_radius_param = float(self.get_parameter("histogram_inflation_radius").value)
        self.show_preferred_arrow = bool(self.get_parameter("show_preferred_arrow").value)
        self.preferred_angle = math.radians(float(self.get_parameter("preferred_angle_deg").value))
        self.preferred_angle_default = self.preferred_angle
        self.mu_pref = float(self.get_parameter("mu_pref").value)
        self.mu_heading = float(self.get_parameter("mu_heading").value)
        self.mu_prev = float(self.get_parameter("mu_prev").value)
        self.mu_density = float(self.get_parameter("mu_density").value)
        self.density_speed_factor = max(0.0, float(self.get_parameter("density_speed_factor").value))
        self.density_speed_min_scale = float(np.clip(
            float(self.get_parameter("density_speed_min_scale").value), 0.0, 1.0
        ))
        self.density_speed_window = max(0, int(self.get_parameter("density_speed_window").value))
        self.reverse_fallback_required_count = max(
            1, int(self.get_parameter("reverse_fallback_required_count").value)
        )
        self.forward_fallback_required_count = max(
            1, int(self.get_parameter("forward_fallback_required_count").value)
        )
        self.fallback_commit_duration = max(0.0, float(self.get_parameter("fallback_commit_duration").value))
        self.front_recovery_required_count = max(
            1, int(self.get_parameter("front_recovery_required_count").value)
        )
        self.front_recovery_clearance = max(
            self.min_obstacle_dist,
            float(self.get_parameter("front_recovery_clearance").value),
        )
        self.rotation_check_step = math.radians(
            max(1.0, float(self.get_parameter("rotation_check_step_deg").value))
        )
        self.scan_topic = str(self.get_parameter("scan_topic").value)
        self.odom_topic = str(self.get_parameter("odom_topic").value)
        self.pointcloud_topic = str(self.get_parameter("pointcloud_topic").value)
        self.goal_pose_topic = str(self.get_parameter("goal_pose_topic").value)
        self.use_goal_pose_topic = bool(self.get_parameter("use_goal_pose_topic").value)
        self.use_pointcloud = bool(self.get_parameter("use_pointcloud").value)
        self.pointcloud_max_points = int(self.get_parameter("pointcloud_max_points").value)
        self.pointcloud_min_range = float(self.get_parameter("pointcloud_min_range").value)
        self.pointcloud_max_range = float(self.get_parameter("pointcloud_max_range").value)
        self.pointcloud_filter_floor = bool(self.get_parameter("pointcloud_filter_floor").value)
        self.pointcloud_floor_z_max = float(self.get_parameter("pointcloud_floor_z_max").value)
        self.pointcloud_filter_body = bool(self.get_parameter("pointcloud_filter_body").value)
        self.noise_filter_enabled = bool(self.get_parameter("noise_filter_enabled").value)
        self.noise_filter_min_points = int(self.get_parameter("noise_filter_min_points").value)
        self.noise_filter_neighbor_radius = float(self.get_parameter("noise_filter_neighbor_radius").value)
        self.noise_filter_min_neighbors = int(self.get_parameter("noise_filter_min_neighbors").value)
        self.clearance_lateral_margin = float(self.get_parameter("clearance_lateral_margin").value)

        self.default_self_filter_polygon = np.array(
            [[0.097, -0.30], [0.097, 0.30], [-0.260, 0.30], [-0.563, 0.15], [-0.563, -0.15], [-0.260, -0.30]],
            dtype=np.float32,
        )
        self.self_filter_polygon = self.default_self_filter_polygon.copy()
        self_filter_raw = self.get_parameter("self_filter_footprint").value
        try:
            flat = np.asarray(self_filter_raw, dtype=np.float32).reshape(-1)
            if flat.size >= 6 and (flat.size % 2 == 0):
                self.self_filter_polygon = flat.reshape(-1, 2).astype(np.float32, copy=False)
        except Exception:
            pass
        self._refresh_footprint_metrics()
        min_rear_lookahead = self.footprint_rear + max(self.safety_margin, self.min_obstacle_dist)
        self.rear_lookahead_distance = max(min_rear_lookahead, self.rear_lookahead_distance_param)
        self.self_filter_polygon_expanded = self._expand_polygon(self.self_filter_polygon, self.safety_margin)

        self.pointcloud_use_tf_fallback = bool(self.get_parameter("pointcloud_use_tf_fallback").value)
        self.pointcloud_tf_fallback_source_frame = str(self.get_parameter("pointcloud_tf_fallback_source_frame").value)
        self.pointcloud_tf_fallback_translation = np.array(
            [float(self.get_parameter("pointcloud_tf_fallback_x").value),
             float(self.get_parameter("pointcloud_tf_fallback_y").value),
             float(self.get_parameter("pointcloud_tf_fallback_z").value)],
            dtype=np.float32,
        )
        fallback_quat = quaternion_from_euler(
            float(self.get_parameter("pointcloud_tf_fallback_roll").value),
            float(self.get_parameter("pointcloud_tf_fallback_pitch").value),
            float(self.get_parameter("pointcloud_tf_fallback_yaw").value),
        )
        self.pointcloud_tf_fallback_rotation = quaternion_matrix(fallback_quat)[:3, :3].astype(np.float32, copy=False)

        self.debug_base_frame = str(self.get_parameter("debug_base_frame").value)
        raw_prefix = str(self.get_parameter("debug_topic_prefix").value).strip()
        if not raw_prefix:
            raw_prefix = "/vfh_viz"
        if not raw_prefix.startswith("/"):
            raw_prefix = f"/{raw_prefix}"
        self.debug_topic_prefix = raw_prefix.rstrip("/")
        self.update_rate = max(1.0, float(self.get_parameter("update_rate").value))

        self.num_sectors = max(8, int(round(360.0 / self.sector_angle_deg)))
        self.sector_angle_rad = 2.0 * math.pi / float(self.num_sectors)
        self.sector_centers = np.linspace(-math.pi, math.pi, self.num_sectors, endpoint=False, dtype=np.float32)
        self.prev_binary_hist = np.ones(self.num_sectors, dtype=bool)

        # State
        self.scan_points_xy: np.ndarray = np.empty((0, 2), dtype=np.float32)
        self.cloud_points_xy: np.ndarray = np.empty((0, 2), dtype=np.float32)
        self.cloud_points_xy_raw: np.ndarray = np.empty((0, 2), dtype=np.float32)
        self.cloud_points_xy_rejected: np.ndarray = np.empty((0, 2), dtype=np.float32)
        self.cloud_points_xyz: np.ndarray = np.empty((0, 3), dtype=np.float32)
        self.cloud_points_xyz_raw: np.ndarray = np.empty((0, 3), dtype=np.float32)
        self.cloud_points_xyz_rejected: np.ndarray = np.empty((0, 3), dtype=np.float32)
        self.current_yaw: float = 0.0
        self.robot_xy: Tuple[float, float] = (0.0, 0.0)
        self.prev_selected_local: float = 0.0
        self.reverse_fallback_fail_count = 0
        self.forward_fallback_fail_count = 0
        self.fallback_commit_until = 0.0
        self.fallback_commit_angle: Optional[float] = None
        self.fallback_commit_kind = ""
        self.front_recovery_blocked = False
        self.front_recovery_confirm_count = 0
        self.front_recovery_last_check_time = 0.0
        self.goal_pose: Optional[PoseStamped] = None
        self.goal_local_xy: Optional[Tuple[float, float]] = None
        self.goal_mode = "preferred_angle"
        self.last_info_time = 0.0
        self.last_tf_warn_time = 0.0

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # Subscriptions
        self.scan_sub = self.create_subscription(LaserScan, self.scan_topic, self._scan_callback, qos_profile_sensor_data)
        self.odom_sub = self.create_subscription(Odometry, self.odom_topic, self._odom_callback, qos_profile_sensor_data)
        self.cloud_sub = self.create_subscription(PointCloud2, self.pointcloud_topic, self._pointcloud_callback, qos_profile_sensor_data)
        if self.use_goal_pose_topic:
            self.goal_sub = self.create_subscription(PoseStamped, self.goal_pose_topic, self._goal_pose_callback, 10)
        else:
            self.goal_sub = None

        # Publishers
        self.info_pub = self.create_publisher(String, f"{self.debug_topic_prefix}/info", 10)
        self.sector_min_pub = self.create_publisher(Float32MultiArray, f"{self.debug_topic_prefix}/sector_min_dist", 10)
        self.sector_density_pub = self.create_publisher(Float32MultiArray, f"{self.debug_topic_prefix}/sector_density", 10)
        self.masked_pub = self.create_publisher(Float32MultiArray, f"{self.debug_topic_prefix}/masked_hist", 10)
        self.pred_cmd_pub = self.create_publisher(Twist, f"{self.debug_topic_prefix}/predicted_cmd", 10)
        self.marker_pub = self.create_publisher(MarkerArray, f"{self.debug_topic_prefix}/markers", 10)
        self.cloud_filtered_pub = self.create_publisher(PointCloud2, f"{self.debug_topic_prefix}/cloud_filtered", 10)
        self.cloud_rejected_pub = self.create_publisher(PointCloud2, f"{self.debug_topic_prefix}/cloud_rejected", 10)

        self.timer = self.create_timer(1.0 / self.update_rate, self._on_timer)
        self.get_logger().info("VFH visualizer (rotation-safe) ready. topics=%s/*, frame=%s" % (self.debug_topic_prefix, self.debug_base_frame))

    # ──────────────────── Utility ────────────────────

    @staticmethod
    def _normalize_angle(angle: float) -> float:
        return math.atan2(math.sin(angle), math.cos(angle))

    @staticmethod
    def _normalize_frame_id(frame_id: str) -> str:
        return frame_id.lstrip("/").strip()

    @staticmethod
    def _expand_polygon(polygon: np.ndarray, margin: float) -> np.ndarray:
        if polygon.ndim != 2 or polygon.shape[1] != 2 or polygon.shape[0] < 3:
            return polygon.astype(np.float32, copy=False)
        center = np.mean(polygon, axis=0)
        directions = polygon - center
        norms = np.linalg.norm(directions, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-6)
        return (polygon + (directions / norms) * float(max(0.0, margin))).astype(np.float32, copy=False)

    @staticmethod
    def _points_in_polygon(points_xy: np.ndarray, polygon: np.ndarray) -> np.ndarray:
        if points_xy.ndim != 2 or points_xy.shape[1] != 2:
            return np.zeros(0, dtype=bool)
        if polygon.ndim != 2 or polygon.shape[1] != 2 or polygon.shape[0] < 3:
            return np.zeros(points_xy.shape[0], dtype=bool)
        inside = np.zeros(points_xy.shape[0], dtype=bool)
        px, py = points_xy[:, 0], points_xy[:, 1]
        j = polygon.shape[0] - 1
        for i in range(polygon.shape[0]):
            xi, yi = polygon[i]
            xj, yj = polygon[j]
            crosses = (yi > py) != (yj > py)
            if np.any(crosses):
                x_int = (xj - xi) * (py[crosses] - yi) / (yj - yi + 1e-12) + xi
                inside[crosses] ^= px[crosses] < x_int
            j = i
        return inside

    @staticmethod
    def _cross_2d(a: np.ndarray, b: np.ndarray) -> float:
        return float(a[0] * b[1] - a[1] * b[0])

    def _angle_diff(self, a: float, b: float) -> float:
        return self._normalize_angle(a - b)

    def _refresh_footprint_metrics(self) -> None:
        if self.self_filter_polygon.ndim != 2 or self.self_filter_polygon.shape[0] < 3:
            self.self_filter_polygon = self.default_self_filter_polygon.copy()
        self.footprint_front = max(0.0, float(np.max(self.self_filter_polygon[:, 0])))
        self.footprint_rear = max(0.0, float(-np.min(self.self_filter_polygon[:, 0])))
        self.footprint_left = max(0.0, float(np.max(self.self_filter_polygon[:, 1])))
        self.footprint_right = max(0.0, float(-np.min(self.self_filter_polygon[:, 1])))
        self.footprint_radius = float(np.max(np.linalg.norm(self.self_filter_polygon, axis=1)))
        if self.histogram_inflation_radius_param > 0.0:
            self.histogram_inflation_radius = self.histogram_inflation_radius_param
        else:
            self.histogram_inflation_radius = (
                max(self.footprint_left, self.footprint_right) + self.safety_margin
            )

    def _rear_corridor_half_width(self) -> float:
        return max(self.footprint_left, self.footprint_right) + self.clearance_lateral_margin

    def _rear_pointcloud_auto_range(self) -> float:
        return math.hypot(max(0.0, self.rear_lookahead_distance), self._rear_corridor_half_width())

    def _angle_to_sector_index(self, angle: float) -> int:
        wrapped = self._normalize_angle(angle)
        return int(np.mod(int(round((wrapped + math.pi) / self.sector_angle_rad)), self.num_sectors))

    def _is_index_in_opening(self, index: int, start: int, width: int) -> bool:
        return int(np.mod(index - start, self.num_sectors)) < width

    def _find_openings(self, hist: np.ndarray) -> List[Tuple[int, int, int]]:
        if hist.size == 0 or not np.any(hist):
            return []
        if np.all(hist):
            return [(0, self.num_sectors - 1, self.num_sectors)]
        blocked = np.where(~hist)[0]
        scan_start = int((blocked[-1] + 1) % self.num_sectors)
        openings: List[Tuple[int, int, int]] = []
        offset = 0
        while offset < self.num_sectors:
            idx = int((scan_start + offset) % self.num_sectors)
            if not hist[idx]:
                offset += 1
                continue
            start = idx
            width = 0
            while offset < self.num_sectors:
                cur = int((scan_start + offset) % self.num_sectors)
                if not hist[cur]:
                    break
                width += 1
                offset += 1
            end = int((start + width - 1) % self.num_sectors)
            openings.append((start, end, width))
        return openings

    def _candidate_indices_from_openings(self, masked_hist: np.ndarray, target_angle: float) -> List[int]:
        openings = self._find_openings(masked_hist)
        if not openings:
            return []
        target_idx = self._angle_to_sector_index(target_angle)
        heading_idx = self._angle_to_sector_index(0.0)
        prev_idx = self._angle_to_sector_index(self.prev_selected_local)
        candidates: List[int] = []

        def add_idx(index: int) -> None:
            idx = int(index % self.num_sectors)
            if idx not in candidates:
                candidates.append(idx)

        for start, _end, width in openings:
            if width <= self.wide_valley_sectors:
                add_idx(start + (width - 1) // 2)
                continue

            side_offset = max(1, self.wide_valley_sectors // 2)
            add_idx(start + side_offset)
            add_idx(start + width - 1 - side_offset)
            for idx in (target_idx, heading_idx, prev_idx):
                if self._is_index_in_opening(idx, start, width):
                    add_idx(idx)
        return candidates

    def _noise_keep_mask(self, points_xy: np.ndarray) -> np.ndarray:
        if (
            not self.noise_filter_enabled
            or points_xy.shape[0] < self.noise_filter_min_points
            or self.noise_filter_neighbor_radius <= 0.0
            or self.noise_filter_min_neighbors <= 0
        ):
            return np.ones(points_xy.shape[0], dtype=bool)

        cell_size = self.noise_filter_neighbor_radius
        cell_ids = np.floor(points_xy / cell_size).astype(np.int32)
        cells = {}
        for idx, cell in enumerate(cell_ids):
            cells.setdefault((int(cell[0]), int(cell[1])), []).append(idx)

        keep = np.zeros(points_xy.shape[0], dtype=bool)
        radius_sq = self.noise_filter_neighbor_radius * self.noise_filter_neighbor_radius
        for idx, cell in enumerate(cell_ids):
            cx, cy = int(cell[0]), int(cell[1])
            neighbor_count = 0
            for gx in range(cx - 1, cx + 2):
                for gy in range(cy - 1, cy + 2):
                    for other in cells.get((gx, gy), []):
                        if other == idx:
                            continue
                        delta = points_xy[other] - points_xy[idx]
                        if float(np.dot(delta, delta)) <= radius_sq:
                            neighbor_count += 1
                            if neighbor_count >= self.noise_filter_min_neighbors:
                                keep[idx] = True
                                break
                    if keep[idx]:
                        break
                if keep[idx]:
                    break
        return keep

    # ──────────────────── Sensor Callbacks ────────────────────

    def _scan_callback(self, msg: LaserScan) -> None:
        ranges = np.asarray(msg.ranges, dtype=np.float32)
        if ranges.size == 0:
            self.scan_points_xy = np.empty((0, 2), dtype=np.float32)
            return
        angles = msg.angle_min + np.arange(ranges.size, dtype=np.float32) * msg.angle_increment
        valid = np.isfinite(ranges) & (ranges >= max(msg.range_min, 0.01))
        if msg.range_max > 0.0:
            valid &= ranges <= msg.range_max
        vr, va = ranges[valid], angles[valid]
        if vr.size == 0:
            self.scan_points_xy = np.empty((0, 2), dtype=np.float32)
            return
        scan_points = np.column_stack((vr * np.cos(va), vr * np.sin(va), np.zeros_like(vr)))
        base_points = self._transform_points_to_debug_frame(scan_points, msg.header.frame_id, msg.header.stamp)
        if base_points.size == 0:
            self.scan_points_xy = np.empty((0, 2), dtype=np.float32)
            return
        xy = base_points[:, :2]
        if self.pointcloud_filter_body:
            xy = xy[~self._points_in_polygon(xy, self.self_filter_polygon_expanded)]
        self.scan_points_xy = xy[self._noise_keep_mask(xy)].astype(np.float32, copy=False)

    def _odom_callback(self, msg: Odometry) -> None:
        q = msg.pose.pose.orientation
        _, _, yaw = euler_from_quaternion([q.x, q.y, q.z, q.w])
        self.current_yaw = float(yaw)
        self.robot_xy = (float(msg.pose.pose.position.x), float(msg.pose.pose.position.y))

    def _goal_pose_callback(self, msg: PoseStamped) -> None:
        self.goal_pose = msg

    def _pointcloud_callback(self, msg: PointCloud2) -> None:
        if not self.use_pointcloud:
            self._clear_cloud_points()
            return
        try:
            points = self._extract_xyz_points(msg)
        except Exception:
            self._clear_cloud_points()
            return
        if points.size == 0:
            self._clear_cloud_points()
            return
        if points.ndim == 1:
            points = points.reshape(1, -1)
        if points.shape[0] > self.pointcloud_max_points > 0:
            step = int(math.ceil(points.shape[0] / float(self.pointcloud_max_points)))
            points = points[::step]
        transformed_xyz = self._transform_points_to_debug_frame(points, msg.header.frame_id, msg.header.stamp)
        if transformed_xyz.size == 0:
            self._clear_cloud_points()
            return
        xy = transformed_xyz[:, :2]
        distances = np.hypot(xy[:, 0], xy[:, 1])
        max_range = self.pointcloud_max_range if self.pointcloud_max_range > 0.0 else max(
            self.repulse_radius * 1.5,
            self._rear_pointcloud_auto_range(),
        )
        valid = np.isfinite(distances) & (distances >= self.pointcloud_min_range) & (distances <= max_range)
        base_xyz = transformed_xyz[valid]
        if base_xyz.size == 0:
            self._clear_cloud_points()
            return
        keep = np.ones(base_xyz.shape[0], dtype=bool)
        if self.pointcloud_filter_floor:
            keep &= base_xyz[:, 2] > self.pointcloud_floor_z_max
        if self.pointcloud_filter_body:
            keep &= ~self._points_in_polygon(base_xyz[:, :2], self.self_filter_polygon_expanded)
        self.cloud_points_xyz_raw = base_xyz.astype(np.float32, copy=False)
        filtered_xyz = base_xyz[keep]
        noise_keep = self._noise_keep_mask(filtered_xyz[:, :2]) if filtered_xyz.size > 0 else np.zeros(0, dtype=bool)
        rejected_parts = [base_xyz[~keep]]
        if filtered_xyz.size > 0:
            rejected_parts.append(filtered_xyz[~noise_keep])
        self.cloud_points_xyz = filtered_xyz[noise_keep].astype(np.float32, copy=False)
        self.cloud_points_xyz_rejected = np.vstack(rejected_parts).astype(np.float32, copy=False)
        self.cloud_points_xy_raw = self.cloud_points_xyz_raw[:, :2]
        self.cloud_points_xy = self.cloud_points_xyz[:, :2]
        self.cloud_points_xy_rejected = self.cloud_points_xyz_rejected[:, :2]

    def _extract_xyz_points(self, msg: PointCloud2) -> np.ndarray:
        raw = point_cloud2.read_points(msg, field_names=("x", "y", "z"), skip_nans=True)
        arr = raw if isinstance(raw, np.ndarray) else np.asarray(list(raw))
        if arr.size == 0:
            return np.empty((0, 3), dtype=np.float32)
        if hasattr(arr.dtype, "names") and arr.dtype.names is not None:
            return np.column_stack((np.asarray(arr["x"], dtype=np.float32),
                                    np.asarray(arr["y"], dtype=np.float32),
                                    np.asarray(arr["z"], dtype=np.float32)))
        arr = np.asarray(arr, dtype=np.float32)
        if arr.ndim == 1:
            return arr.reshape(-1, 3)
        return arr[:, :3].astype(np.float32, copy=False)

    def _clear_cloud_points(self) -> None:
        for attr in ("cloud_points_xy", "cloud_points_xy_raw", "cloud_points_xy_rejected",
                      "cloud_points_xyz", "cloud_points_xyz_raw", "cloud_points_xyz_rejected"):
            dim = 3 if "xyz" in attr else 2
            setattr(self, attr, np.empty((0, dim), dtype=np.float32))

    def _transform_points_to_debug_frame(self, points_xyz, source_frame, source_stamp):
        source = self._normalize_frame_id(source_frame or "")
        target = self._normalize_frame_id(self.debug_base_frame)
        if not source or source == target:
            return points_xyz.astype(np.float32, copy=False)
        try:
            tf_msg = self.tf_buffer.lookup_transform(target, source, Time.from_msg(source_stamp), timeout=Duration(seconds=0.05))
        except TransformException:
            try:
                tf_msg = self.tf_buffer.lookup_transform(target, source, Time(), timeout=Duration(seconds=0.05))
            except TransformException:
                fallback_source = self._normalize_frame_id(self.pointcloud_tf_fallback_source_frame)
                if self.pointcloud_use_tf_fallback and (not fallback_source or source == fallback_source):
                    return (points_xyz.astype(np.float32) @ self.pointcloud_tf_fallback_rotation.T + self.pointcloud_tf_fallback_translation).astype(np.float32)
                return np.empty((0, 3), dtype=np.float32)
        t = np.array([tf_msg.transform.translation.x, tf_msg.transform.translation.y, tf_msg.transform.translation.z], dtype=np.float32)
        r = quaternion_matrix([tf_msg.transform.rotation.x, tf_msg.transform.rotation.y, tf_msg.transform.rotation.z, tf_msg.transform.rotation.w])[:3, :3].astype(np.float32)
        return (points_xyz.astype(np.float32) @ r.T + t).astype(np.float32)

    def _goal_position_in_debug_frame(self) -> Optional[np.ndarray]:
        if self.goal_pose is None:
            return None
        pose = self.goal_pose
        source = self._normalize_frame_id(pose.header.frame_id or self.debug_base_frame)
        target = self._normalize_frame_id(self.debug_base_frame)
        point = np.array(
            [pose.pose.position.x, pose.pose.position.y, pose.pose.position.z],
            dtype=np.float32,
        )
        if source == target:
            return point
        try:
            tf_msg = self.tf_buffer.lookup_transform(target, source, Time(), timeout=Duration(seconds=0.05))
        except TransformException:
            return None
        translation = np.array(
            [tf_msg.transform.translation.x, tf_msg.transform.translation.y, tf_msg.transform.translation.z],
            dtype=np.float32,
        )
        rotation = quaternion_matrix(
            [tf_msg.transform.rotation.x, tf_msg.transform.rotation.y,
             tf_msg.transform.rotation.z, tf_msg.transform.rotation.w]
        )[:3, :3].astype(np.float32)
        return point @ rotation.T + translation

    def _update_preferred_angle_from_goal(self) -> None:
        goal_point = self._goal_position_in_debug_frame()
        if goal_point is None:
            self.preferred_angle = self.preferred_angle_default
            self.goal_local_xy = None
            self.goal_mode = "preferred_angle"
            return
        x, y = float(goal_point[0]), float(goal_point[1])
        if math.hypot(x, y) < 1e-3:
            self.preferred_angle = self.preferred_angle_default
            self.goal_local_xy = (x, y)
            self.goal_mode = "goal_reached"
            return
        self.preferred_angle = self._normalize_angle(math.atan2(y, x))
        self.goal_local_xy = (x, y)
        self.goal_mode = "rviz_goal"

    # ──────────────────── VFH Pipeline ────────────────────

    def _compose_points(self) -> np.ndarray:
        if self.scan_points_xy.size == 0 and self.cloud_points_xy.size == 0:
            return np.empty((0, 2), dtype=np.float32)
        if self.scan_points_xy.size == 0:
            return self.cloud_points_xy
        if self.cloud_points_xy.size == 0:
            return self.scan_points_xy
        return np.vstack((self.scan_points_xy, self.cloud_points_xy))

    def _build_polar_histogram(self, points: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        densities = np.zeros(self.num_sectors, dtype=np.float32)
        min_distances = np.full(self.num_sectors, np.inf, dtype=np.float32)
        if points.size == 0:
            return densities, min_distances
        distances = np.hypot(points[:, 0], points[:, 1])
        active_radius = max(self.repulse_radius, self.histogram_inflation_radius + 0.05)
        valid = np.isfinite(distances) & (distances > 0.01) & (distances <= active_radius)
        if not np.any(valid):
            return densities, min_distances
        pts, dists = points[valid], distances[valid]
        angles = np.arctan2(pts[:, 1], pts[:, 0])
        weights = np.clip(1.0 - np.square(dists / max(active_radius, 1e-6)), 0.0, 1.0)
        inflation = max(0.01, self.histogram_inflation_radius)
        for angle, dist, weight in zip(angles, dists, weights):
            if weight <= 0.0:
                continue
            half_angle = math.pi if dist <= inflation else math.asin(min(1.0, inflation / max(float(dist), 1e-6)))
            span = min(self.num_sectors // 2, int(math.ceil(half_angle / self.sector_angle_rad)))
            center = self._angle_to_sector_index(float(angle))
            indices = np.mod(np.arange(center - span, center + span + 1, dtype=np.int32), self.num_sectors)
            densities[indices] += float(weight)
            np.minimum.at(min_distances, indices, float(dist))
        return densities, min_distances

    def _get_footprint_distance(self, angle: float) -> float:
        direction = np.array([math.cos(angle), math.sin(angle)], dtype=np.float32)
        polygon = self.self_filter_polygon
        best_t = float("inf")
        for i in range(polygon.shape[0]):
            p0 = polygon[i]
            p1 = polygon[(i + 1) % polygon.shape[0]]
            edge = p1 - p0
            denom = self._cross_2d(direction, edge)
            if abs(denom) < 1e-9:
                continue
            t = self._cross_2d(p0, edge) / denom
            u = self._cross_2d(p0, direction) / denom
            if t >= 0.0 and (-1e-6 <= u <= 1.0 + 1e-6):
                best_t = min(best_t, t)
        if math.isfinite(best_t):
            return float(best_t)
        return float(getattr(self, "footprint_radius", 0.0))

    def _build_binary_histogram(self, densities: np.ndarray, min_distances: np.ndarray, obstacle_points: np.ndarray) -> np.ndarray:
        hist = self.prev_binary_hist.copy()
        hist[densities > self.binary_threshold_high] = False
        hist[densities < self.binary_threshold_low] = True
        clearance_threshold = max(self.safety_margin, self.min_obstacle_dist)
        for i, angle in enumerate(self.sector_centers):
            la = float(angle)
            fp = self._get_footprint_distance(la)
            if float(min_distances[i]) - fp <= clearance_threshold:
                hist[i] = False
                continue
            if not self._check_width_clearance(la, fp, obstacle_points, clearance_threshold):
                hist[i] = False
        self.prev_binary_hist = hist.copy()
        return hist

    def _check_width_clearance(self, direction, footprint_forward, obstacle_points, clearance_threshold):
        if obstacle_points.size == 0:
            return True
        cos_d, sin_d = math.cos(direction), math.sin(direction)
        along = obstacle_points[:, 0] * cos_d + obstacle_points[:, 1] * sin_d
        lateral = -obstacle_points[:, 0] * sin_d + obstacle_points[:, 1] * cos_d
        left_lim = self._get_footprint_distance(self._normalize_angle(direction + math.pi * 0.5)) + self.safety_margin
        right_lim = self._get_footprint_distance(self._normalize_angle(direction - math.pi * 0.5)) + self.safety_margin
        in_path = (lateral <= left_lim) & (lateral >= -right_lim) & (along > 0.0)
        if not np.any(in_path):
            return True
        return (float(np.min(along[in_path])) - footprint_forward) > clearance_threshold

    def _is_reverse_corridor_safe(self, obstacle_points: np.ndarray) -> bool:
        if obstacle_points.size == 0:
            return True
        rear_limit = self._get_footprint_distance(math.pi) + self.min_obstacle_dist
        return self._get_rear_clearance(obstacle_points) > rear_limit

    def _build_masked_histogram(self, points: np.ndarray) -> np.ndarray:
        masked = np.ones(self.num_sectors, dtype=bool)
        if points.size == 0:
            return masked
        reverse_safe = self._is_reverse_corridor_safe(points)
        rear_indices = np.abs(np.asarray([self._normalize_angle(float(a)) for a in self.sector_centers])) > (
            math.pi * 0.5
        )
        masked[rear_indices] = reverse_safe
        for idx in np.where(~rear_indices)[0]:
            if not self._is_trajectory_safe(float(self.sector_centers[idx]), points):
                masked[idx] = False
        return masked

    def _is_trajectory_safe(self, target_angle: float, obstacle_points: np.ndarray) -> bool:
        if obstacle_points.size == 0:
            return True
        if abs(target_angle) > math.pi * 0.5:
            return self._is_reverse_corridor_safe(obstacle_points)

        if abs(target_angle) < max(0.15, self.sector_angle_rad):
            return self._is_forward_after_rotation_safe(0.0, obstacle_points)

        radius = max(0.05, self.min_turning_radius)
        steps = max(3, int(abs(target_angle) / self.trajectory_check_step))
        polygon = self.self_filter_polygon_expanded
        turn_sign = 1.0 if target_angle >= 0.0 else -1.0

        for step in range(1, steps + 1):
            theta = target_angle * step / steps
            abs_theta = abs(theta)
            center_x = radius * math.sin(abs_theta)
            center_y = turn_sign * radius * (1.0 - math.cos(abs_theta))
            cos_t = math.cos(-theta)
            sin_t = math.sin(-theta)
            shifted_x = obstacle_points[:, 0] - center_x
            shifted_y = obstacle_points[:, 1] - center_y
            rx = shifted_x * cos_t - shifted_y * sin_t
            ry = shifted_x * sin_t + shifted_y * cos_t
            if np.any(self._points_in_polygon(np.column_stack((rx, ry)), polygon)):
                return False
        return self._is_forward_after_rotation_safe(target_angle, obstacle_points)

    # ──────────────────── Rotation Safety (METHOD 1) ────────────────────

    def _is_rotation_safe(self, target_angle: float, obstacle_points: np.ndarray) -> bool:
        """
        Check if rotating from heading 0 to target_angle is collision-free.

        Simulates the rotation in small steps and checks if the footprint
        collides with any obstacle point at each intermediate angle.
        """
        if obstacle_points.size == 0:
            return True

        num_steps = max(3, int(abs(target_angle) / self.rotation_check_step))
        polygon = self.self_filter_polygon_expanded

        for step in range(1, num_steps + 1):
            angle = target_angle * step / num_steps
            cos_a = math.cos(-angle)
            sin_a = math.sin(-angle)

            # Rotate obstacle points by -angle = equivalent to robot rotating by +angle
            rx = obstacle_points[:, 0] * cos_a - obstacle_points[:, 1] * sin_a
            ry = obstacle_points[:, 0] * sin_a + obstacle_points[:, 1] * cos_a
            rotated = np.column_stack((rx, ry))

            if np.any(self._points_in_polygon(rotated, polygon)):
                return False

        return True

    def _is_forward_after_rotation_safe(self, target_angle: float, obstacle_points: np.ndarray) -> bool:
        return self._get_forward_after_rotation_clearance(target_angle, obstacle_points) > self.min_obstacle_dist

    def _get_forward_after_rotation_clearance(self, target_angle: float, obstacle_points: np.ndarray) -> float:
        if obstacle_points.size == 0:
            return float("inf")
        cos_a = math.cos(-target_angle)
        sin_a = math.sin(-target_angle)
        rx = obstacle_points[:, 0] * cos_a - obstacle_points[:, 1] * sin_a
        ry = obstacle_points[:, 0] * sin_a + obstacle_points[:, 1] * cos_a
        front = self._get_footprint_distance(0.0)
        left = self._get_footprint_distance(math.pi * 0.5) + self.safety_margin
        right = self._get_footprint_distance(-math.pi * 0.5) + self.safety_margin
        in_path = (rx > 0.0) & (ry <= left) & (ry >= -right)
        if not np.any(in_path):
            return float("inf")
        return float(np.min(rx[in_path])) - front

    def _is_selected_motion_safe(self, local_angle: float, obstacle_points: np.ndarray) -> bool:
        if abs(local_angle) > math.pi * 0.5:
            return self._is_reverse_corridor_safe(obstacle_points)
        if abs(local_angle) >= 0.15 and not self._is_rotation_safe(local_angle, obstacle_points):
            return False
        return self._is_forward_after_rotation_safe(local_angle, obstacle_points)

    @staticmethod
    def _is_forward_hemisphere(local_angle: float) -> bool:
        return abs(local_angle) < (math.pi * 0.5)

    def _reset_fallback_waits(self) -> None:
        self.reverse_fallback_fail_count = 0
        self.forward_fallback_fail_count = 0

    def _clear_fallback_commit(self) -> None:
        self.fallback_commit_until = 0.0
        self.fallback_commit_angle = None
        self.fallback_commit_kind = ""

    def _reset_front_recovery_block(self) -> None:
        self.front_recovery_blocked = False
        self.front_recovery_confirm_count = 0
        self.front_recovery_last_check_time = 0.0

    def _front_recovery_allows(self, local_angle: float, obstacle_points: np.ndarray) -> bool:
        if not self.front_recovery_blocked or not self._is_forward_hemisphere(local_angle):
            return True
        clearance = self._get_forward_after_rotation_clearance(local_angle, obstacle_points)
        if clearance >= self.front_recovery_clearance:
            now_sec = time.monotonic()
            if now_sec - self.front_recovery_last_check_time >= 0.05:
                self.front_recovery_confirm_count += 1
                self.front_recovery_last_check_time = now_sec
        else:
            self.front_recovery_confirm_count = 0
        if self.front_recovery_confirm_count >= self.front_recovery_required_count:
            self._reset_front_recovery_block()
            return True
        return False

    def _commit_selected_direction(
        self,
        local_angle: float,
        reset_fallback_wait: bool = True,
        clear_fallback_commit: bool = True,
    ) -> float:
        if reset_fallback_wait:
            self._reset_fallback_waits()
        if clear_fallback_commit:
            self._clear_fallback_commit()
        self.prev_selected_local = local_angle
        return local_angle

    def _fallback_kind(self, preferred_local: float, candidate_local: float) -> Optional[str]:
        preferred_forward = self._is_forward_hemisphere(preferred_local)
        candidate_forward = self._is_forward_hemisphere(candidate_local)
        if preferred_forward and not candidate_forward:
            return "reverse"
        if (not preferred_forward) and candidate_forward:
            return "forward"
        return None

    def _fallback_required_count(self, kind: str) -> int:
        if kind == "forward":
            return self.forward_fallback_required_count
        return self.reverse_fallback_required_count

    def _fallback_wait_count(self, kind: str) -> int:
        if kind == "forward":
            return self.forward_fallback_fail_count
        return self.reverse_fallback_fail_count

    def _set_fallback_wait_count(self, kind: str, value: int) -> None:
        if kind == "forward":
            self.forward_fallback_fail_count = value
            self.reverse_fallback_fail_count = 0
        else:
            self.reverse_fallback_fail_count = value
            self.forward_fallback_fail_count = 0

    def _select_fallback_after_delay(self, fallback_angle: float, kind: str) -> Optional[float]:
        count = self._fallback_wait_count(kind) + 1
        self._set_fallback_wait_count(kind, count)
        if count < self._fallback_required_count(kind):
            return None
        if kind == "reverse":
            self.front_recovery_blocked = True
            self.front_recovery_confirm_count = 0
            self.front_recovery_last_check_time = 0.0
        else:
            self._reset_front_recovery_block()
        self.fallback_commit_angle = fallback_angle
        self.fallback_commit_kind = kind
        self.fallback_commit_until = time.monotonic() + self.fallback_commit_duration
        return self._commit_selected_direction(
            fallback_angle,
            reset_fallback_wait=False,
            clear_fallback_commit=False,
        )

    def _try_committed_fallback(self, preferred_local: float, obstacle_points: np.ndarray) -> Optional[float]:
        if self.fallback_commit_angle is None or time.monotonic() >= self.fallback_commit_until:
            self._clear_fallback_commit()
            return None
        kind = self._fallback_kind(preferred_local, self.fallback_commit_angle)
        if kind != self.fallback_commit_kind:
            self._clear_fallback_commit()
            return None
        if not self._is_selected_motion_safe(self.fallback_commit_angle, obstacle_points):
            self._clear_fallback_commit()
            return None
        return self._commit_selected_direction(
            self.fallback_commit_angle,
            reset_fallback_wait=False,
            clear_fallback_commit=False,
        )

    def _select_direction_safe(
        self,
        masked_hist: np.ndarray,
        densities: np.ndarray,
        min_distances: np.ndarray,
        obstacle_points: np.ndarray,
    ) -> Optional[float]:
        """
        Select direction with VFH+ cost function + rotation safety check.

        If the best direction requires unsafe rotation, try alternatives.
        If no safe rotation exists, fallback to reverse.
        """
        committed = self._try_committed_fallback(self.preferred_angle, obstacle_points)
        if committed is not None:
            return committed

        open_indices = np.where(masked_hist)[0]
        if open_indices.size == 0:
            rear_clearance = self._get_rear_clearance(obstacle_points)
            if self._is_forward_hemisphere(self.preferred_angle):
                if rear_clearance > (self._get_footprint_distance(math.pi) + self.min_obstacle_dist):
                    return self._select_fallback_after_delay(math.pi, "reverse")
            elif self._is_selected_motion_safe(0.0, obstacle_points):
                return self._select_fallback_after_delay(0.0, "forward")
            self._reset_fallback_waits()
            return None

        # Determine effective preferred direction
        forward_passable = any(
            abs(float(self.sector_centers[idx])) < (math.pi * 0.5) for idx in open_indices
        )
        if forward_passable:
            effective_preferred = self.preferred_angle
            effective_mu_pref = self.mu_pref
        else:
            effective_preferred = self._normalize_angle(self.preferred_angle + math.pi)
            effective_mu_pref = self.mu_pref * 0.5

        candidate_indices = self._candidate_indices_from_openings(masked_hist, effective_preferred)
        if not candidate_indices:
            if self._is_forward_hemisphere(self.preferred_angle):
                rear_clearance = self._get_rear_clearance(obstacle_points)
                if rear_clearance > (self._get_footprint_distance(math.pi) + self.min_obstacle_dist):
                    return self._select_fallback_after_delay(math.pi, "reverse")
            elif self._is_selected_motion_safe(0.0, obstacle_points):
                return self._select_fallback_after_delay(0.0, "forward")
            self._reset_fallback_waits()
            return None

        # Score VFH+ opening candidates, not every free sector.
        scored = []
        for idx in candidate_indices:
            local_angle = float(self.sector_centers[idx])
            cost = (
                effective_mu_pref * abs(self._angle_diff(local_angle, effective_preferred))
                + self.mu_heading * abs(self._angle_diff(local_angle, 0.0))
                + self.mu_prev * abs(self._angle_diff(local_angle, self.prev_selected_local))
                + self.mu_density * float(densities[idx])
            )
            scored.append((local_angle, cost))
        scored.sort(key=lambda x: x[1])

        # Try same-hemisphere candidates first. Opposite-hemisphere candidates are fallbacks.
        pending_fallback: Optional[Tuple[float, str]] = None
        for local_angle, cost in scored:
            if self._is_selected_motion_safe(local_angle, obstacle_points):
                fallback_kind = self._fallback_kind(self.preferred_angle, local_angle)
                if fallback_kind is not None:
                    if pending_fallback is None:
                        pending_fallback = (local_angle, fallback_kind)
                    continue
                if not self._front_recovery_allows(local_angle, obstacle_points):
                    continue
                return self._commit_selected_direction(local_angle)

        if pending_fallback is not None:
            return self._select_fallback_after_delay(pending_fallback[0], pending_fallback[1])

        # No candidate has safe rotation → try the opposite hemisphere fallback.
        if self._is_forward_hemisphere(self.preferred_angle):
            rear_clearance = self._get_rear_clearance(obstacle_points)
            if rear_clearance > (self._get_footprint_distance(math.pi) + self.min_obstacle_dist):
                best_angle = scored[0][0] if scored else self.preferred_angle
                reverse_angle = math.copysign(math.pi, best_angle)
                return self._select_fallback_after_delay(reverse_angle, "reverse")
        elif self._is_selected_motion_safe(0.0, obstacle_points):
            return self._select_fallback_after_delay(0.0, "forward")

        # Nothing works → return None (stop)
        self._reset_fallback_waits()
        return None

    # ──────────────────── Command Generation ────────────────────

    def _reverse_yaw_error(self, local_angle: float) -> float:
        rear_axis = math.copysign(math.pi, local_angle if abs(local_angle) > 1e-6 else 1.0)
        return self._normalize_angle(local_angle - rear_axis)

    def _motion_heading_error(self, local_angle: float) -> Tuple[bool, float]:
        if abs(local_angle) <= (math.pi * 0.5):
            return False, self._normalize_angle(local_angle)
        return True, self._reverse_yaw_error(local_angle)

    def _get_rear_clearance(self, points: np.ndarray) -> float:
        if points.size == 0:
            return float("inf")
        lat = self._rear_corridor_half_width()
        rear_depth = -points[:, 0]
        rear = (
            (rear_depth > 0.0)
            & (rear_depth <= self.rear_lookahead_distance)
            & (np.abs(points[:, 1]) <= lat)
        )
        if not np.any(rear):
            return float("inf")
        return float(np.min(rear_depth[rear]))

    def _selected_density(self, selected_local: Optional[float], densities: np.ndarray) -> float:
        if selected_local is None or densities.size == 0:
            return 0.0
        center = self._angle_to_sector_index(selected_local)
        window = min(self.density_speed_window, max(0, self.num_sectors // 2))
        indices = np.mod(
            np.arange(center - window, center + window + 1, dtype=np.int32),
            self.num_sectors,
        )
        values = densities[indices]
        values = values[np.isfinite(values)]
        if values.size == 0:
            return 0.0
        return float(np.mean(values))

    def _density_speed_scale(self, selected_density: float) -> float:
        if self.density_speed_factor <= 0.0:
            return 1.0
        density = max(0.0, float(selected_density))
        scale = 1.0 / (1.0 + self.density_speed_factor * density)
        return float(np.clip(scale, self.density_speed_min_scale, 1.0))

    def _predict_cmd(
        self,
        selected_local: Optional[float],
        nearest_obstacle: float,
        points: np.ndarray,
        density_speed_scale: float = 1.0,
    ) -> Twist:
        cmd = Twist()
        if selected_local is None:
            return cmd

        rear_clearance = self._get_rear_clearance(points)
        is_reverse, heading_err = self._motion_heading_error(selected_local)
        abs_err = abs(heading_err)

        angular_gain = 1.2
        angular = float(np.clip(heading_err * angular_gain, -self.max_angular_speed, self.max_angular_speed))

        base_linear = -self.max_reverse_speed if is_reverse else self.max_linear_speed
        if abs_err < 0.3:
            linear = base_linear
        elif abs_err < 1.0:
            linear = base_linear * 0.4
        else:
            linear = 0.0

        if is_reverse:
            clearance_for_scale = rear_clearance - self._get_footprint_distance(math.pi)
            speed_scale_distance = self.rear_lookahead_distance - self._get_footprint_distance(math.pi)
        else:
            clearance_for_scale = self._get_forward_after_rotation_clearance(selected_local, points)
            speed_scale_distance = self.repulse_radius
        if clearance_for_scale <= self.min_obstacle_dist:
            return cmd

        denom = max(1e-6, speed_scale_distance - self.min_obstacle_dist)
        scale = float(np.clip((clearance_for_scale - self.min_obstacle_dist) / denom, 0.0, 1.0))
        density_scale = float(np.clip(density_speed_scale, 0.0, 1.0))
        linear_cmd = float(linear * scale * density_scale)
        min_speed = self.min_command_linear_speed * density_scale
        if linear_cmd != 0.0 and min_speed > 0.0 and abs(linear_cmd) < min_speed:
            linear_cmd = math.copysign(min_speed, linear_cmd)
        cmd.linear.x = float(np.clip(linear_cmd, -self.max_reverse_speed, self.max_linear_speed))
        cmd.angular.z = float(np.clip(angular, -self.max_angular_speed, self.max_angular_speed))
        return cmd

    # ──────────────────── Visualization ────────────────────

    def _make_arrow(self, mid, ns, angle, length, color, shaft=0.02, head=0.04, alpha=0.95):
        m = Marker()
        m.header.frame_id = self.debug_base_frame
        m.header.stamp = self.get_clock().now().to_msg()
        m.ns, m.id, m.type, m.action = ns, mid, Marker.ARROW, Marker.ADD
        m.scale.x, m.scale.y, m.scale.z = float(shaft), float(head), float(head * 1.5)
        m.color.r, m.color.g, m.color.b, m.color.a = float(color[0]), float(color[1]), float(color[2]), float(alpha)
        p0, p1 = Point(), Point()
        p0.z = p1.z = 0.06
        p1.x, p1.y = float(length * math.cos(angle)), float(length * math.sin(angle))
        m.points = [p0, p1]
        return m

    def _make_sector_arrow(self, idx, angle, is_passable):
        color = (0.05, 0.85, 0.18) if is_passable else (0.95, 0.08, 0.04)
        m = self._make_arrow(
            int(idx),
            "vfh_sector_arrows",
            float(angle),
            self.repulse_radius * 0.72,
            color,
            shaft=0.012,
            head=0.035,
            alpha=0.82,
        )
        m.header.stamp = self.get_clock().now().to_msg()
        return m

    def _publish_cloud_debug(self) -> None:
        now_stamp = self.get_clock().now().to_msg()
        for pub, data in [(self.cloud_filtered_pub, self.cloud_points_xyz),
                          (self.cloud_rejected_pub, self.cloud_points_xyz_rejected)]:
            h = Header()
            h.frame_id = self.debug_base_frame
            h.stamp = now_stamp
            pub.publish(point_cloud2.create_cloud_xyz32(h, data.tolist() if data.size > 0 else []))

    def _publish_markers(self, points, min_distances, masked_hist, selected_local, nearest_obstacle):
        msg = MarkerArray()
        stamp = self.get_clock().now().to_msg()

        for old_id in (0, 1):
            dm = Marker()
            dm.header.frame_id, dm.header.stamp = self.debug_base_frame, stamp
            dm.ns, dm.id, dm.action = "vfh_sectors", old_id, Marker.DELETE
            msg.markers.append(dm)

        for i, angle in enumerate(self.sector_centers):
            marker = self._make_sector_arrow(i, angle, bool(masked_hist[i]))
            marker.header.stamp = stamp
            msg.markers.append(marker)

        if self.show_preferred_arrow:
            msg.markers.append(
                self._make_arrow(2, "vfh_pref", self.preferred_angle, 0.9, (1.0, 0.8, 0.1))
            )
        else:
            dm = Marker()
            dm.header.frame_id, dm.header.stamp = self.debug_base_frame, stamp
            dm.ns, dm.id, dm.action = "vfh_pref", 2, Marker.DELETE
            msg.markers.append(dm)

        if self.goal_local_xy is not None:
            gm = Marker()
            gm.header.frame_id, gm.header.stamp = self.debug_base_frame, stamp
            gm.ns, gm.id, gm.type, gm.action = "vfh_goal_point", 8, Marker.SPHERE, Marker.ADD
            gm.pose.position.x = float(self.goal_local_xy[0])
            gm.pose.position.y = float(self.goal_local_xy[1])
            gm.pose.position.z = 0.08
            gm.pose.orientation.w = 1.0
            gm.scale.x = gm.scale.y = gm.scale.z = 0.12
            gm.color.r, gm.color.g, gm.color.b, gm.color.a = 1.0, 0.85, 0.05, 0.95
            msg.markers.append(gm)
        else:
            dm = Marker()
            dm.header.frame_id, dm.header.stamp = self.debug_base_frame, stamp
            dm.ns, dm.id, dm.action = "vfh_goal_point", 8, Marker.DELETE
            msg.markers.append(dm)

        if selected_local is not None:
            msg.markers.append(
                self._make_arrow(
                    3,
                    "vfh_selected",
                    selected_local,
                    self.repulse_radius * 1.55,
                    (0.1, 0.7, 1.0),
                    shaft=0.035,
                    head=0.085,
                    alpha=1.0,
                )
            )
        else:
            dm = Marker()
            dm.header.frame_id, dm.header.stamp, dm.ns, dm.id, dm.action = self.debug_base_frame, stamp, "vfh_selected", 3, Marker.DELETE
            msg.markers.append(dm)

        # Obstacle points
        pts_m = Marker()
        pts_m.header.frame_id, pts_m.header.stamp = self.debug_base_frame, stamp
        pts_m.ns, pts_m.id, pts_m.type, pts_m.action = "vfh_points", 4, Marker.POINTS, Marker.ADD
        pts_m.scale.x = pts_m.scale.y = 0.015
        pts_m.color.r, pts_m.color.g, pts_m.color.b, pts_m.color.a = 0.9, 0.9, 0.9, 0.8
        if points.size > 0:
            for p in points[::max(1, points.shape[0] // 600)]:
                pt = Point()
                pt.x, pt.y, pt.z = float(p[0]), float(p[1]), 0.01
                pts_m.points.append(pt)
        msg.markers.append(pts_m)

        # Cloud filtered / rejected
        for mid, data, color, alpha in [(6, self.cloud_points_xyz, (0.1, 0.9, 1.0), 0.95),
                                         (7, self.cloud_points_xyz_rejected, (1.0, 0.2, 1.0), 0.35)]:
            cm = Marker()
            cm.header.frame_id, cm.header.stamp = self.debug_base_frame, stamp
            cm.ns, cm.id, cm.type, cm.action = f"vfh_cloud_{mid}", mid, Marker.POINTS, Marker.ADD
            cm.scale.x = cm.scale.y = 0.02 if mid == 6 else 0.016
            cm.color.r, cm.color.g, cm.color.b, cm.color.a = color[0], color[1], color[2], alpha
            if data.size > 0:
                for p in data[::max(1, data.shape[0] // 800)]:
                    pt = Point()
                    pt.x, pt.y, pt.z = float(p[0]), float(p[1]), float(p[2])
                    cm.points.append(pt)
            msg.markers.append(cm)

        # Status text
        st = Marker()
        st.header.frame_id, st.header.stamp = self.debug_base_frame, stamp
        st.ns, st.id, st.type, st.action = "vfh_status", 5, Marker.TEXT_VIEW_FACING, Marker.ADD
        st.pose.orientation.w, st.pose.position.z, st.scale.z = 1.0, 0.25, 0.08
        st.color.r, st.color.g, st.color.b, st.color.a = 1.0, 1.0, 1.0, 0.9
        sel_deg = math.degrees(selected_local) if selected_local is not None else float("nan")
        pref_deg = math.degrees(self.preferred_angle)
        passable = int(np.count_nonzero(masked_hist))
        st.text = (
            f"near={nearest_obstacle:.2f}m sel={sel_deg:.1f}deg "
            f"pref={pref_deg:.1f}deg mode={self.goal_mode} "
            f"pass={passable}/{self.num_sectors} "
            f"cloud={self.cloud_points_xy.shape[0]}/{self.cloud_points_xy_raw.shape[0]}"
        )
        msg.markers.append(st)

        self.marker_pub.publish(msg)

    # ──────────────────── Main Timer ────────────────────

    def _on_timer(self) -> None:
        self._update_preferred_angle_from_goal()
        points = self._compose_points()

        densities, min_distances = self._build_polar_histogram(points)
        binary_hist = self._build_binary_histogram(densities, min_distances, points)
        masked_hist = self._build_masked_histogram(points)

        # KEY CHANGE: use rotation-safe direction selection
        selected_local = self._select_direction_safe(masked_hist, densities, min_distances, points)

        nearest = float(np.min(np.hypot(points[:, 0], points[:, 1]))) if points.size > 0 else float("inf")
        selected_density = self._selected_density(selected_local, densities)
        density_speed_scale = self._density_speed_scale(selected_density)
        pred_cmd = self._predict_cmd(selected_local, nearest, points, density_speed_scale)

        # Publish debug data
        min_msg = Float32MultiArray()
        min_msg.data = np.where(np.isfinite(min_distances), min_distances, -1.0).astype(np.float32).tolist()
        self.sector_min_pub.publish(min_msg)

        den_msg = Float32MultiArray()
        den_msg.data = densities.astype(np.float32).tolist()
        self.sector_density_pub.publish(den_msg)

        mask_msg = Float32MultiArray()
        mask_msg.data = masked_hist.astype(np.float32).tolist()
        self.masked_pub.publish(mask_msg)

        self.pred_cmd_pub.publish(pred_cmd)
        self._publish_cloud_debug()
        self._publish_markers(points, min_distances, masked_hist, selected_local, nearest)

        now = time.monotonic()
        if now - self.last_info_time >= 0.5:
            self.last_info_time = now
            info = String()
            sel_deg = math.degrees(selected_local) if selected_local is not None else float("nan")
            near_txt = f"{nearest:.3f}" if math.isfinite(nearest) else "inf"
            pref_deg = math.degrees(self.preferred_angle)
            fallback_hold_remaining = max(0.0, self.fallback_commit_until - now)
            info.data = (
                f"points={points.shape[0]} nearest={near_txt}m "
                f"goal_mode={self.goal_mode} pref={pref_deg:.1f}deg "
                f"selected={sel_deg:.1f}deg pred_cmd=({pred_cmd.linear.x:.3f},{pred_cmd.angular.z:.3f}) "
                f"sel_density={selected_density:.3f} density_scale={density_speed_scale:.2f} "
                f"fb_wait=R{self.reverse_fallback_fail_count}/{self.reverse_fallback_required_count},"
                f"F{self.forward_fallback_fail_count}/{self.forward_fallback_required_count} "
                f"fb_hold={self.fallback_commit_kind}:{fallback_hold_remaining:.1f}s "
                f"front_recover={int(self.front_recovery_blocked)}:"
                f"{self.front_recovery_confirm_count}/{self.front_recovery_required_count} "
                f"cloud_filtered={self.cloud_points_xy.shape[0]} "
                f"cloud_raw={self.cloud_points_xy_raw.shape[0]} "
                f"cloud_rejected={self.cloud_points_xy_rejected.shape[0]}"
            )
            self.info_pub.publish(info)


def main(args: Optional[Sequence[str]] = None) -> None:
    rclpy.init(args=args)
    node = VFHVisualizerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
