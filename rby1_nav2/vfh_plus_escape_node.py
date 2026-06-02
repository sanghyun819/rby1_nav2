#!/usr/bin/env python3
"""VFH+ based reactive escape node with rotation safety check."""

from __future__ import annotations

import math
import time
from enum import Enum
from typing import List, Optional, Sequence, Tuple

import numpy as np
import rclpy
from geometry_msgs.msg import Point, Point32, PolygonStamped, PoseStamped, Twist
from inha_interfaces.action import VfhPlusEscape
from nav2_simple_commander.robot_navigator import BasicNavigator
from nav_msgs.msg import Odometry, Path
from rclpy.action import ActionServer, CancelResponse, GoalResponse
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.duration import Duration
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rclpy.time import Time
from sensor_msgs.msg import LaserScan, PointCloud2
from sensor_msgs_py import point_cloud2
from std_msgs.msg import Float32MultiArray, Header, String
from tf2_ros import Buffer, TransformException, TransformListener
from tf_transformations import euler_from_quaternion, quaternion_from_euler, quaternion_matrix
from visualization_msgs.msg import Marker, MarkerArray


class EscapeReason(Enum):
    PLANNER_FAILED = "planner_failed"
    CONTROLLER_FAILED = "controller_failed"


class EscapeMode(Enum):
    GOAL_DIRECTION = "goal_direction"
    WIDEST_OPENING = "widest_opening"
    PATH_DIRECTION = "path_direction"
    PREFERRED_DIRECTION = "preferred_direction"


class VFHPlusEscapeNode(Node):
    """Reactive VFH+ escape node with rotation safety and Nav2 handoff."""

    def __init__(self, node_name: str = "vfh_plus_escape_node") -> None:
        super().__init__(node_name)

        # ── Parameters ──
        self.declare_parameter("sector_angle", 5.0)
        self.declare_parameter("safety_margin", 0.05)
        self.declare_parameter("min_turning_radius", 0.4)
        self.declare_parameter("max_linear_speed", 0.1)
        self.declare_parameter("max_reverse_speed", 0.0)
        self.declare_parameter("min_command_linear_speed", 0.03)
        self.declare_parameter("max_angular_speed", 0.2)
        self.declare_parameter("min_obstacle_dist", 0.3)
        self.declare_parameter("repulse_radius", 0.8)
        self.declare_parameter("rear_lookahead_distance", 1.2)
        self.declare_parameter("binary_threshold_low", 0.08)
        self.declare_parameter("binary_threshold_high", 0.15)
        self.declare_parameter("wide_valley_sectors", 16)
        self.declare_parameter("mu1", 5.0)   # goal direction preference
        self.declare_parameter("mu2", 2.0)   # current heading maintenance
        self.declare_parameter("mu3", 2.0)   # previous selection maintenance
        self.declare_parameter("mu_density", 0.8)
        self.declare_parameter("density_speed_factor", 0.5)
        self.declare_parameter("density_speed_min_scale", 0.6)
        self.declare_parameter("density_speed_window", 2)
        self.declare_parameter("reverse_fallback_required_count", 3)
        self.declare_parameter("forward_fallback_required_count", 2)
        self.declare_parameter("fallback_commit_duration", 1.0)
        self.declare_parameter("front_recovery_required_count", 3)
        self.declare_parameter("front_recovery_clearance", 0.45)
        self.declare_parameter("widest_width_weight", 4.0)
        self.declare_parameter("widest_clearance_weight", 1.5)
        self.declare_parameter("widest_preferred_clearance", 2.0)
        self.declare_parameter("widest_rotation_weight", 1.0)
        self.declare_parameter("widest_prev_weight", 0.5)
        self.declare_parameter("widest_reverse_penalty", 0.8)
        self.declare_parameter("escape_timeout", 30.0)
        self.declare_parameter("planner_escape_distance", 0.4)
        self.declare_parameter("controller_escape_distance", 1.0)
        self.declare_parameter("nav2_check_interval", 1.0)
        self.declare_parameter("rotation_check_step_deg", 6.0)
        self.declare_parameter("trajectory_check_step_deg", 5.0)
        self.declare_parameter("histogram_inflation_radius", 0.0)

        self.declare_parameter("scan_topic", "/scan_merged")
        self.declare_parameter("odom_topic", "/odom")
        self.declare_parameter("pointcloud_topic", "/livox/lidar")
        self.declare_parameter("goal_pose_topic", "/goal_pose")
        self.declare_parameter("path_topic", "/plan")
        self.declare_parameter("use_path_topic", True)
        self.declare_parameter("enable_goal_pose_start", True)
        self.declare_parameter("enable_action_server", True)
        self.declare_parameter("action_name", "vfh_escape")
        self.declare_parameter("goal_escape_reason", EscapeReason.PLANNER_FAILED.value)
        self.declare_parameter("escape_mode", EscapeMode.GOAL_DIRECTION.value)
        self.declare_parameter("planner_escape_mode", EscapeMode.WIDEST_OPENING.value)
        self.declare_parameter("controller_escape_mode", EscapeMode.PATH_DIRECTION.value)
        self.declare_parameter("path_lookahead_distance", 0.7)
        self.declare_parameter("path_min_target_distance", 0.15)
        self.declare_parameter("nav2_handoff_enabled", False)
        self.declare_parameter("pointcloud_max_points", 4000)
        self.declare_parameter("pointcloud_min_range", 0.05)
        self.declare_parameter("pointcloud_max_range", 0.0)
        self.declare_parameter("pointcloud_filter_floor", True)
        self.declare_parameter("pointcloud_floor_z_max", -0.10)
        self.declare_parameter("pointcloud_filter_body", True)
        self.declare_parameter("noise_filter_enabled", True)
        self.declare_parameter("noise_filter_min_points", 8)
        self.declare_parameter("noise_filter_neighbor_radius", 0.12)
        self.declare_parameter("noise_filter_min_neighbors", 1)
        self.declare_parameter(
            "self_filter_footprint",
            [0.097, -0.30, 0.097, 0.30, -0.260, 0.30, -0.563, 0.15, -0.563, -0.15, -0.260, -0.30],
        )
        self.declare_parameter("clearance_lateral_margin", 0.10)
        self.declare_parameter("cmd_topic", "/cmd_vel_raw")
        self.declare_parameter("dry_run_only", True)
        self.declare_parameter("debug_base_frame", "base_nav")
        self.declare_parameter("debug_topic_prefix", "/vfh_debug")
        self.declare_parameter("pointcloud_use_tf_fallback", False)
        self.declare_parameter("pointcloud_tf_fallback_source_frame", "livox_lidar")
        self.declare_parameter("pointcloud_tf_fallback_x", -0.265442083)
        self.declare_parameter("pointcloud_tf_fallback_y", 0.002229000)
        self.declare_parameter("pointcloud_tf_fallback_z", 1.389722440)
        self.declare_parameter("pointcloud_tf_fallback_roll", -3.136562568)
        self.declare_parameter("pointcloud_tf_fallback_pitch", 0.013624321)
        self.declare_parameter("pointcloud_tf_fallback_yaw", -0.003078160)

        # ── Load parameters ──
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
        self.mu1 = float(self.get_parameter("mu1").value)
        self.mu2 = float(self.get_parameter("mu2").value)
        self.mu3 = float(self.get_parameter("mu3").value)
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
        self.widest_width_weight = float(self.get_parameter("widest_width_weight").value)
        self.widest_clearance_weight = float(self.get_parameter("widest_clearance_weight").value)
        self.widest_preferred_clearance = max(
            self.min_obstacle_dist,
            float(self.get_parameter("widest_preferred_clearance").value),
        )
        self.widest_rotation_weight = float(self.get_parameter("widest_rotation_weight").value)
        self.widest_prev_weight = float(self.get_parameter("widest_prev_weight").value)
        self.widest_reverse_penalty = float(self.get_parameter("widest_reverse_penalty").value)
        self.escape_timeout = float(self.get_parameter("escape_timeout").value)
        self.planner_escape_distance = float(self.get_parameter("planner_escape_distance").value)
        self.controller_escape_distance = float(self.get_parameter("controller_escape_distance").value)
        self.nav2_check_interval = float(self.get_parameter("nav2_check_interval").value)
        self.rotation_check_step = math.radians(
            max(1.0, float(self.get_parameter("rotation_check_step_deg").value))
        )
        self.trajectory_check_step = math.radians(
            max(1.0, float(self.get_parameter("trajectory_check_step_deg").value))
        )
        self.histogram_inflation_radius_param = float(self.get_parameter("histogram_inflation_radius").value)

        self.scan_topic = str(self.get_parameter("scan_topic").value)
        self.odom_topic = str(self.get_parameter("odom_topic").value)
        self.pointcloud_topic = str(self.get_parameter("pointcloud_topic").value)
        self.goal_pose_topic = str(self.get_parameter("goal_pose_topic").value)
        self.path_topic = str(self.get_parameter("path_topic").value)
        self.use_path_topic = bool(self.get_parameter("use_path_topic").value)
        self.enable_goal_pose_start = bool(self.get_parameter("enable_goal_pose_start").value)
        self.enable_action_server = bool(self.get_parameter("enable_action_server").value)
        self.action_name = str(self.get_parameter("action_name").value)
        self.goal_escape_reason = self._parse_escape_reason(
            str(self.get_parameter("goal_escape_reason").value)
        )
        self.default_escape_mode = self._parse_escape_mode(str(self.get_parameter("escape_mode").value))
        self.planner_escape_mode = self._parse_escape_mode(str(self.get_parameter("planner_escape_mode").value))
        self.controller_escape_mode = self._parse_escape_mode(str(self.get_parameter("controller_escape_mode").value))
        self.path_lookahead_distance = float(self.get_parameter("path_lookahead_distance").value)
        self.path_min_target_distance = float(self.get_parameter("path_min_target_distance").value)
        self.nav2_handoff_enabled = bool(self.get_parameter("nav2_handoff_enabled").value)
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
        self.cmd_topic = str(self.get_parameter("cmd_topic").value)
        self.dry_run_only = bool(self.get_parameter("dry_run_only").value)
        self.debug_base_frame = str(self.get_parameter("debug_base_frame").value)

        raw_prefix = str(self.get_parameter("debug_topic_prefix").value).strip()
        if not raw_prefix:
            raw_prefix = "/vfh_debug"
        if not raw_prefix.startswith("/"):
            raw_prefix = f"/{raw_prefix}"
        self.debug_topic_prefix = raw_prefix.rstrip("/")

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

        # Footprint polygon
        self.default_self_filter_polygon = np.array(
            [[0.097, -0.30], [0.097, 0.30], [-0.260, 0.30], [-0.563, 0.15], [-0.563, -0.15], [-0.260, -0.30]],
            dtype=np.float32,
        )
        self.self_filter_polygon = self.default_self_filter_polygon.copy()
        try:
            flat = np.asarray(self.get_parameter("self_filter_footprint").value, dtype=np.float32).reshape(-1)
            if flat.size >= 6 and (flat.size % 2 == 0):
                self.self_filter_polygon = flat.reshape(-1, 2).astype(np.float32, copy=False)
        except Exception:
            pass
        self._refresh_footprint_metrics()
        min_rear_lookahead = self.footprint_rear + max(self.safety_margin, self.min_obstacle_dist)
        self.rear_lookahead_distance = max(min_rear_lookahead, self.rear_lookahead_distance_param)
        self.self_filter_polygon_expanded = self._expand_polygon(self.self_filter_polygon, self.safety_margin)

        # Sectors
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
        self.current_pose: Optional[Tuple[float, float, float]] = None
        self.last_path: Optional[Path] = None

        self.goal_pose: Optional[PoseStamped] = None
        self.escape_reason: Optional[EscapeReason] = None
        self.current_escape_mode = self.default_escape_mode
        self.escape_active = False
        self.escape_result = False
        self.escape_result_message = ""
        self.escape_start_time = 0.0
        self.active_escape_timeout = self.escape_timeout
        self.active_escape_distance = 0.0
        self.action_speed_limit: Optional[float] = None
        self.action_preferred_local_angle: Optional[float] = None
        self.action_preferred_global_angle: Optional[float] = None
        self.escape_start_xy = (0.0, 0.0)
        self.escape_start_pose_pending = False

        self.prev_direction_global = 0.0
        self.prev_selected_local = 0.0
        self.last_nav2_check_time = 0.0
        self.last_no_sensor_warn = 0.0

        self.controller_anchor_xy = (0.0, 0.0)
        self.controller_trial_active = False
        self.controller_trial_start_time = 0.0
        self.controller_trial_start_xy = (0.0, 0.0)
        self.controller_fail_count = 0
        self.reverse_fallback_fail_count = 0
        self.forward_fallback_fail_count = 0
        self.fallback_commit_until = 0.0
        self.fallback_commit_angle: Optional[float] = None
        self.fallback_commit_kind = ""
        self.front_recovery_blocked = False
        self.front_recovery_confirm_count = 0
        self.front_recovery_last_check_time = 0.0

        self.last_debug_info_time = 0.0
        self.debug_info_period = 0.5

        self.navigator = BasicNavigator() if self.nav2_handoff_enabled else None
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.action_callback_group = ReentrantCallbackGroup()

        # Subscriptions
        self.scan_sub = self.create_subscription(LaserScan, self.scan_topic, self._scan_callback, qos_profile_sensor_data)
        self.cloud_sub = self.create_subscription(PointCloud2, self.pointcloud_topic, self._pointcloud_callback, qos_profile_sensor_data)
        self.odom_sub = self.create_subscription(Odometry, self.odom_topic, self._odom_callback, qos_profile_sensor_data)
        self.goal_sub = (
            self.create_subscription(PoseStamped, self.goal_pose_topic, self._goal_pose_callback, 10)
            if self.enable_goal_pose_start
            else None
        )
        self.path_sub = (
            self.create_subscription(Path, self.path_topic, self._path_callback, 10)
            if self.use_path_topic
            else None
        )

        # Publishers
        self.cmd_pub = self.create_publisher(Twist, self.cmd_topic, 10)
        self.debug_info_pub = self.create_publisher(String, f"{self.debug_topic_prefix}/info", 10)
        self.debug_sector_dist_pub = self.create_publisher(Float32MultiArray, f"{self.debug_topic_prefix}/sector_min_dist", 10)
        self.debug_sector_density_pub = self.create_publisher(Float32MultiArray, f"{self.debug_topic_prefix}/sector_density", 10)
        self.debug_footprint_pub = self.create_publisher(PolygonStamped, f"{self.debug_topic_prefix}/footprint", 10)
        self.debug_target_pose_pub = self.create_publisher(PoseStamped, f"{self.debug_topic_prefix}/target_pose", 10)
        self.debug_marker_pub = self.create_publisher(MarkerArray, f"{self.debug_topic_prefix}/markers", 10)
        self.debug_cloud_filtered_pub = self.create_publisher(PointCloud2, f"{self.debug_topic_prefix}/cloud_filtered", 10)
        self.debug_cloud_rejected_pub = self.create_publisher(PointCloud2, f"{self.debug_topic_prefix}/cloud_rejected", 10)

        self.action_server = None
        if self.enable_action_server:
            self.action_server = ActionServer(
                self,
                VfhPlusEscape,
                self.action_name,
                execute_callback=self._execute_action,
                goal_callback=self._action_goal_callback,
                cancel_callback=self._action_cancel_callback,
                callback_group=self.action_callback_group,
            )

        self.control_timer = self.create_timer(0.1, self._control_loop)

        self.get_logger().info(
            "VFHPlusEscapeNode ready: sectors=%d, dry_run=%s, action=%s, goal_start=%s"
            % (self.num_sectors, self.dry_run_only, self.action_name if self.enable_action_server else "disabled", self.enable_goal_pose_start)
        )

    # ──────────────────── Utility ────────────────────

    @staticmethod
    def _normalize_angle(angle: float) -> float:
        return math.atan2(math.sin(angle), math.cos(angle))

    @staticmethod
    def _normalize_frame_id(frame_id: str) -> str:
        return frame_id.lstrip("/").strip()

    @staticmethod
    def _parse_escape_reason(value: str) -> EscapeReason:
        normalized = value.strip().lower()
        for reason in EscapeReason:
            if normalized in (reason.value, reason.name.lower()):
                return reason
        return EscapeReason.PLANNER_FAILED

    @staticmethod
    def _parse_escape_mode(value: str) -> EscapeMode:
        normalized = value.strip().lower().replace("-", "_")
        for mode in EscapeMode:
            if normalized in (mode.value, mode.name.lower()):
                return mode
        return EscapeMode.GOAL_DIRECTION

    @staticmethod
    def _reason_from_action_goal(value: int) -> EscapeReason:
        if int(value) == VfhPlusEscape.Goal.REASON_CONTROLLER_FAILED:
            return EscapeReason.CONTROLLER_FAILED
        return EscapeReason.PLANNER_FAILED

    def _mode_from_action_goal(self, value: int, reason: EscapeReason) -> EscapeMode:
        mapping = {
            VfhPlusEscape.Goal.MODE_WIDEST_OPENING: EscapeMode.WIDEST_OPENING,
            VfhPlusEscape.Goal.MODE_GOAL_DIRECTION: EscapeMode.GOAL_DIRECTION,
            VfhPlusEscape.Goal.MODE_PATH_DIRECTION: EscapeMode.PATH_DIRECTION,
            VfhPlusEscape.Goal.MODE_PREFERRED_DIRECTION: EscapeMode.PREFERRED_DIRECTION,
        }
        return mapping.get(int(value), self._mode_for_reason(reason))

    def _mode_for_reason(self, reason: EscapeReason) -> EscapeMode:
        if reason == EscapeReason.PLANNER_FAILED:
            return self.planner_escape_mode
        if reason == EscapeReason.CONTROLLER_FAILED:
            return self.controller_escape_mode
        return self.default_escape_mode

    def _distance_for_reason(self, reason: EscapeReason) -> float:
        if reason == EscapeReason.PLANNER_FAILED:
            return max(0.0, self.planner_escape_distance)
        if reason == EscapeReason.CONTROLLER_FAILED:
            return max(0.0, self.controller_escape_distance)
        return 0.0

    def _angle_diff(self, a: float, b: float) -> float:
        return self._normalize_angle(a - b)

    @staticmethod
    def _distance_xy(p0: Tuple[float, float], p1: Tuple[float, float]) -> float:
        return math.hypot(p0[0] - p1[0], p0[1] - p1[1])

    def _current_escape_distance(self) -> float:
        if self.current_pose is None:
            return 0.0
        return float(self._distance_xy((self.current_pose[0], self.current_pose[1]), self.escape_start_xy))

    def _preferred_direction_angle(self, direction) -> Optional[float]:
        x = float(direction.x)
        y = float(direction.y)
        if not (math.isfinite(x) and math.isfinite(y)):
            return None
        if math.hypot(x, y) < 1e-3:
            return None
        return self._normalize_angle(math.atan2(y, x))

    @staticmethod
    def _expand_polygon(polygon: np.ndarray, margin: float) -> np.ndarray:
        if polygon.ndim != 2 or polygon.shape[1] != 2 or polygon.shape[0] < 3:
            return polygon.astype(np.float32, copy=False)
        center = np.mean(polygon, axis=0)
        directions = polygon - center
        norms = np.maximum(np.linalg.norm(directions, axis=1, keepdims=True), 1e-6)
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
            lateral_extent = max(self.footprint_left, self.footprint_right)
            self.histogram_inflation_radius = lateral_extent + self.safety_margin

    def _rear_corridor_half_width(self) -> float:
        return max(self.footprint_left, self.footprint_right) + self.clearance_lateral_margin

    def _rear_pointcloud_auto_range(self) -> float:
        return math.hypot(max(0.0, self.rear_lookahead_distance), self._rear_corridor_half_width())

    def _widest_pointcloud_auto_range(self) -> float:
        return math.hypot(
            self.footprint_radius + max(0.0, self.widest_preferred_clearance),
            max(self.footprint_left, self.footprint_right) + self.safety_margin,
        )

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

    def _pointcloud_callback(self, msg: PointCloud2) -> None:
        try:
            raw = point_cloud2.read_points(msg, field_names=("x", "y", "z"), skip_nans=True)
            arr = raw if isinstance(raw, np.ndarray) else np.asarray(list(raw))
        except Exception:
            self._clear_cloud_points()
            return
        if arr.size == 0:
            self._clear_cloud_points()
            return
        if hasattr(arr.dtype, "names") and arr.dtype.names is not None:
            points = np.column_stack((np.asarray(arr["x"], dtype=np.float32),
                                      np.asarray(arr["y"], dtype=np.float32),
                                      np.asarray(arr["z"], dtype=np.float32)))
        else:
            points = np.asarray(arr, dtype=np.float32)
            if points.ndim == 1:
                points = points.reshape(-1, 3)
            points = points[:, :3]

        if points.shape[0] > self.pointcloud_max_points > 0:
            step = int(math.ceil(points.shape[0] / float(self.pointcloud_max_points)))
            points = points[::step]

        points = self._transform_points_to_debug_frame(points, msg.header.frame_id, msg.header.stamp)
        if points.size == 0:
            self._clear_cloud_points()
            return

        xy = points[:, :2]
        dists = np.hypot(xy[:, 0], xy[:, 1])
        max_range = self.pointcloud_max_range if self.pointcloud_max_range > 0.0 else max(
            self.repulse_radius * 1.5,
            self.min_turning_radius + self.footprint_radius + self.safety_margin + 0.3,
            self._rear_pointcloud_auto_range(),
            self._widest_pointcloud_auto_range(),
        )
        range_valid = np.isfinite(dists) & (dists >= self.pointcloud_min_range) & (dists <= max_range)
        base_xyz = points[range_valid]
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

    def _clear_cloud_points(self) -> None:
        for attr in ("cloud_points_xy", "cloud_points_xy_raw", "cloud_points_xy_rejected",
                     "cloud_points_xyz", "cloud_points_xyz_raw", "cloud_points_xyz_rejected"):
            dim = 3 if "xyz" in attr else 2
            setattr(self, attr, np.empty((0, dim), dtype=np.float32))

    def _transform_points_to_debug_frame(self, points_xyz: np.ndarray, source_frame: str, source_stamp) -> np.ndarray:
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
                    return (
                        points_xyz.astype(np.float32) @ self.pointcloud_tf_fallback_rotation.T
                        + self.pointcloud_tf_fallback_translation
                    ).astype(np.float32)
                return np.empty((0, 3), dtype=np.float32)
        translation = np.array(
            [tf_msg.transform.translation.x, tf_msg.transform.translation.y, tf_msg.transform.translation.z],
            dtype=np.float32,
        )
        rotation = quaternion_matrix(
            [tf_msg.transform.rotation.x, tf_msg.transform.rotation.y, tf_msg.transform.rotation.z, tf_msg.transform.rotation.w]
        )[:3, :3].astype(np.float32)
        return (points_xyz.astype(np.float32) @ rotation.T + translation).astype(np.float32)

    def _transform_point_to_debug_frame(self, point_xyz: np.ndarray, source_frame: str, source_stamp=None) -> Optional[np.ndarray]:
        source = self._normalize_frame_id(source_frame or "")
        target = self._normalize_frame_id(self.debug_base_frame)
        point = np.asarray(point_xyz, dtype=np.float32).reshape(3)
        if not source or source == target:
            return point

        lookup_time = Time()
        if source_stamp is not None:
            try:
                lookup_time = Time.from_msg(source_stamp)
            except Exception:
                lookup_time = Time()
        try:
            tf_msg = self.tf_buffer.lookup_transform(target, source, lookup_time, timeout=Duration(seconds=0.05))
        except TransformException:
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
        return (point @ rotation.T + translation).astype(np.float32)

    def _goal_position_in_debug_frame(self) -> Optional[np.ndarray]:
        if self.goal_pose is None:
            return None
        pose = self.goal_pose
        point = np.array(
            [pose.pose.position.x, pose.pose.position.y, pose.pose.position.z],
            dtype=np.float32,
        )
        return self._transform_point_to_debug_frame(point, pose.header.frame_id or self.debug_base_frame, pose.header.stamp)

    def _odom_callback(self, msg: Odometry) -> None:
        q = msg.pose.pose.orientation
        _, _, yaw = euler_from_quaternion([q.x, q.y, q.z, q.w])
        self.current_pose = (float(msg.pose.pose.position.x), float(msg.pose.pose.position.y), float(yaw))

    def _path_callback(self, msg: Path) -> None:
        self.last_path = msg

    def _goal_pose_callback(self, msg: PoseStamped) -> None:
        self._start_escape(msg, self.goal_escape_reason, block=False)

    # ──────────────────── VFH Pipeline ────────────────────

    def _compose_obstacle_points(self) -> np.ndarray:
        if self.scan_points_xy.size == 0 and self.cloud_points_xy.size == 0:
            return np.empty((0, 2), dtype=np.float32)
        if self.scan_points_xy.size == 0:
            return self.cloud_points_xy
        if self.cloud_points_xy.size == 0:
            return self.scan_points_xy
        return np.vstack((self.scan_points_xy, self.cloud_points_xy))

    def _build_polar_histogram(self, points: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Step 1: VFH+ primary polar histogram with footprint-width expansion."""
        densities = np.zeros(self.num_sectors, dtype=np.float32)
        min_distances = np.full(self.num_sectors, np.inf, dtype=np.float32)
        if points.size == 0:
            return densities, min_distances
        dists = np.hypot(points[:, 0], points[:, 1])
        active_radius = max(self.repulse_radius, self.histogram_inflation_radius + 0.05)
        valid = np.isfinite(dists) & (dists > 0.01) & (dists <= active_radius)
        if not np.any(valid):
            return densities, min_distances
        pts, ds = points[valid], dists[valid]
        angles = np.arctan2(pts[:, 1], pts[:, 0])
        weights = np.clip(1.0 - np.square(ds / max(active_radius, 1e-6)), 0.0, 1.0)
        inflation = max(0.01, self.histogram_inflation_radius)

        for angle, dist, weight in zip(angles, ds, weights):
            if weight <= 0.0:
                continue
            half_angle = math.pi if dist <= inflation else math.asin(min(1.0, inflation / max(float(dist), 1e-6)))
            span = min(self.num_sectors // 2, int(math.ceil(half_angle / self.sector_angle_rad)))
            center = self._angle_to_sector_index(float(angle))
            indices = np.mod(np.arange(center - span, center + span + 1, dtype=np.int32), self.num_sectors)
            densities[indices] += float(weight)
            np.minimum.at(min_distances, indices, float(dist))
        return densities, min_distances

    def get_footprint_distance(self, angle: float) -> float:
        """Ray-polygon intersection for directional footprint radius."""
        direction = np.array([math.cos(angle), math.sin(angle)], dtype=np.float32)
        polygon = self.self_filter_polygon
        best_t = float("inf")
        for i in range(polygon.shape[0]):
            p0, p1 = polygon[i], polygon[(i + 1) % polygon.shape[0]]
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

    def _check_width_clearance(self, direction: float, fp_fwd: float, points: np.ndarray, threshold: float) -> bool:
        """Check robot-width corridor for obstacles along direction."""
        if points.size == 0:
            return True
        cos_d, sin_d = math.cos(direction), math.sin(direction)
        along = points[:, 0] * cos_d + points[:, 1] * sin_d
        lateral = -points[:, 0] * sin_d + points[:, 1] * cos_d
        left = self.get_footprint_distance(self._normalize_angle(direction + math.pi * 0.5)) + self.safety_margin
        right = self.get_footprint_distance(self._normalize_angle(direction - math.pi * 0.5)) + self.safety_margin
        in_path = (lateral <= left) & (lateral >= -right) & (along > 0.0)
        if not np.any(in_path):
            return True
        return (float(np.min(along[in_path])) - fp_fwd) > threshold

    def _directional_corridor_clearance(
        self,
        direction: float,
        points: np.ndarray,
        max_clearance: float,
    ) -> float:
        if points.size == 0:
            return float("inf")
        cos_d, sin_d = math.cos(direction), math.sin(direction)
        along = points[:, 0] * cos_d + points[:, 1] * sin_d
        lateral = -points[:, 0] * sin_d + points[:, 1] * cos_d
        fp_fwd = self.get_footprint_distance(direction)
        left = self.get_footprint_distance(self._normalize_angle(direction + math.pi * 0.5)) + self.safety_margin
        right = self.get_footprint_distance(self._normalize_angle(direction - math.pi * 0.5)) + self.safety_margin
        max_along = fp_fwd + max(0.0, max_clearance)
        in_path = (
            (along > 0.0)
            & (along <= max_along)
            & (lateral <= left)
            & (lateral >= -right)
        )
        if not np.any(in_path):
            return float("inf")
        return float(np.min(along[in_path])) - fp_fwd

    def _build_binary_histogram(self, densities: np.ndarray, min_distances: np.ndarray, points: np.ndarray) -> np.ndarray:
        """Step 2: VFH+ binary histogram with threshold hysteresis."""
        hist = self.prev_binary_hist.copy()
        hist[densities > self.binary_threshold_high] = False
        hist[densities < self.binary_threshold_low] = True
        threshold = max(self.safety_margin, self.min_obstacle_dist)
        for i, angle in enumerate(self.sector_centers):
            la = float(angle)
            fp = self.get_footprint_distance(la)
            if float(min_distances[i]) - fp <= threshold:
                hist[i] = False
                continue
            if not self._check_width_clearance(la, fp, points, threshold):
                hist[i] = False
        self.prev_binary_hist = hist.copy()
        return hist

    def _is_reverse_corridor_safe(self, obstacle_points: np.ndarray) -> bool:
        if obstacle_points.size == 0:
            return True
        rear_limit = self.get_footprint_distance(math.pi) + self.min_obstacle_dist
        return self._get_rear_clearance(obstacle_points) > rear_limit

    def _build_masked_histogram(self, points: np.ndarray) -> np.ndarray:
        """Step 3: Geometry pass/fail using our footprint swept along each sector."""
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

    # ──────────────────── Rotation Safety ────────────────────

    def _is_rotation_safe(self, target_angle: float, obstacle_points: np.ndarray) -> bool:
        """
        Check if rotating from heading 0 to target_angle is collision-free.

        Rotates obstacle points by intermediate angles and checks footprint overlap.
        """
        if obstacle_points.size == 0:
            return True

        num_steps = max(3, int(abs(target_angle) / self.rotation_check_step))
        polygon = self.self_filter_polygon_expanded

        for step in range(1, num_steps + 1):
            angle = target_angle * step / num_steps
            cos_a = math.cos(-angle)
            sin_a = math.sin(-angle)
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
        front = self.get_footprint_distance(0.0)
        left = self.get_footprint_distance(math.pi * 0.5) + self.safety_margin
        right = self.get_footprint_distance(-math.pi * 0.5) + self.safety_margin
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
        current_yaw: float,
        reset_fallback_wait: bool = True,
        clear_fallback_commit: bool = True,
    ) -> float:
        if reset_fallback_wait:
            self._reset_fallback_waits()
        if clear_fallback_commit:
            self._clear_fallback_commit()
        self.prev_selected_local = local_angle
        self.prev_direction_global = self._normalize_angle(current_yaw + local_angle)
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

    def _select_fallback_after_delay(
        self,
        fallback_angle: float,
        kind: str,
        current_yaw: float,
    ) -> Optional[float]:
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
            current_yaw,
            reset_fallback_wait=False,
            clear_fallback_commit=False,
        )

    def _try_committed_fallback(
        self,
        preferred_local: float,
        obstacle_points: np.ndarray,
        current_yaw: float,
    ) -> Optional[float]:
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
            current_yaw,
            reset_fallback_wait=False,
            clear_fallback_commit=False,
        )

    # ──────────────────── Direction Selection ────────────────────

    def _determine_goal_angle(self) -> Tuple[float, str]:
        """Compute a global-equivalent goal angle from goal pose transformed into base frame."""
        if self.current_pose is None or self.goal_pose is None:
            if self.current_pose is not None:
                return self.current_pose[2], "FORWARD"
            return 0.0, "FORWARD"

        goal_point = self._goal_position_in_debug_frame()
        if goal_point is None:
            return self.current_pose[2], "GOAL_TF_MISSING"
        x, y = float(goal_point[0]), float(goal_point[1])
        if math.hypot(x, y) < 1e-3:
            return self.current_pose[2], "GOAL_REACHED"
        local_goal = self._normalize_angle(math.atan2(y, x))
        return self._normalize_angle(self.current_pose[2] + local_goal), "GOAL_DIRECT_TF"

    def _path_preferred_local_angle(self) -> Optional[float]:
        if self.last_path is None or not self.last_path.poses:
            return None

        local_points: List[np.ndarray] = []
        default_frame = self.last_path.header.frame_id or "map"
        default_stamp = self.last_path.header.stamp
        for pose_stamped in self.last_path.poses:
            source_frame = pose_stamped.header.frame_id or default_frame
            source_stamp = pose_stamped.header.stamp if pose_stamped.header.frame_id else default_stamp
            point = np.array(
                [
                    pose_stamped.pose.position.x,
                    pose_stamped.pose.position.y,
                    pose_stamped.pose.position.z,
                ],
                dtype=np.float32,
            )
            local = self._transform_point_to_debug_frame(point, source_frame, source_stamp)
            if local is not None and np.all(np.isfinite(local[:2])):
                local_points.append(local[:2])

        if not local_points:
            return None

        points = np.asarray(local_points, dtype=np.float32)
        distances = np.hypot(points[:, 0], points[:, 1])
        start_idx = int(np.argmin(distances))
        target = points[start_idx]
        accumulated = 0.0
        prev = points[start_idx]

        for point in points[start_idx + 1:]:
            accumulated += float(np.linalg.norm(point - prev))
            prev = point
            if accumulated >= self.path_lookahead_distance:
                target = point
                break
        else:
            for point in points[start_idx + 1:]:
                if float(np.linalg.norm(point)) >= self.path_min_target_distance:
                    target = point

        if float(np.linalg.norm(target)) < self.path_min_target_distance:
            return None
        return self._normalize_angle(math.atan2(float(target[1]), float(target[0])))

    def _determine_preferred_goal_angle(self, current_yaw: float) -> Tuple[float, str]:
        if self.current_escape_mode == EscapeMode.PREFERRED_DIRECTION:
            if self.action_preferred_local_angle is not None:
                if self.action_preferred_global_angle is None:
                    self.action_preferred_global_angle = self._normalize_angle(
                        current_yaw + self.action_preferred_local_angle
                    )
                return self.action_preferred_global_angle, "PREFERRED_DIRECTION_ACTION_GLOBAL"
            return current_yaw, "PREFERRED_DIRECTION_MISSING"

        if self.current_escape_mode == EscapeMode.PATH_DIRECTION:
            if self.action_preferred_local_angle is not None:
                if self.action_preferred_global_angle is None:
                    self.action_preferred_global_angle = self._normalize_angle(
                        current_yaw + self.action_preferred_local_angle
                    )
                return self.action_preferred_global_angle, "PATH_DIRECTION_ACTION_GLOBAL"
            path_angle = self._path_preferred_local_angle()
            if path_angle is not None:
                return self._normalize_angle(current_yaw + path_angle), "PATH_DIRECTION_PATH"
            goal_angle, _ = self._determine_goal_angle()
            return goal_angle, "PATH_DIRECTION_FALLBACK_GOAL"

        if self.current_escape_mode == EscapeMode.WIDEST_OPENING:
            return current_yaw, "WIDEST_OPENING"

        return self._determine_goal_angle()

    def _select_widest_opening_direction(
        self,
        masked_hist: np.ndarray,
        densities: np.ndarray,
        min_distances: np.ndarray,
        obstacle_points: np.ndarray,
        current_yaw: float,
    ) -> Optional[float]:
        committed = self._try_committed_fallback(0.0, obstacle_points, current_yaw)
        if committed is not None:
            return committed

        openings = self._find_openings(masked_hist)
        if not openings:
            rear_clearance = self._get_rear_clearance(obstacle_points)
            if rear_clearance > (self.get_footprint_distance(math.pi) + self.min_obstacle_dist):
                return self._select_fallback_after_delay(math.pi, "reverse", current_yaw)
            self._reset_fallback_waits()
            return None

        scored: List[Tuple[float, float]] = []

        def add_candidate(index: int, width: int) -> None:
            idx = int(index % self.num_sectors)
            local_angle = float(self.sector_centers[idx])
            clearance = self._directional_corridor_clearance(
                local_angle,
                obstacle_points,
                self.widest_preferred_clearance,
            )
            if not math.isfinite(clearance):
                clearance = self.widest_preferred_clearance
            width_norm = float(width) / float(max(1, self.num_sectors))
            clearance_norm = float(np.clip(clearance / max(self.widest_preferred_clearance, 1e-6), 0.0, 1.0))
            rotation_norm = abs(self._angle_diff(local_angle, 0.0)) / math.pi
            prev_norm = abs(self._angle_diff(local_angle, self.prev_selected_local)) / math.pi
            reverse_penalty = self.widest_reverse_penalty if abs(local_angle) > (math.pi * 0.5) else 0.0
            cost = (
                -self.widest_width_weight * width_norm
                - self.widest_clearance_weight * clearance_norm
                + self.widest_rotation_weight * rotation_norm
                + self.widest_prev_weight * prev_norm
                + self.mu_density * float(densities[idx])
                + reverse_penalty
            )
            scored.append((local_angle, cost))

        heading_idx = self._angle_to_sector_index(0.0)
        prev_idx = self._angle_to_sector_index(self.prev_selected_local)
        for start, _end, width in openings:
            center_idx = start + (width - 1) // 2
            add_candidate(center_idx, width)
            if self._is_index_in_opening(heading_idx, start, width):
                add_candidate(heading_idx, width)
            if self._is_index_in_opening(prev_idx, start, width):
                add_candidate(prev_idx, width)
            if width > self.wide_valley_sectors:
                side_offset = max(1, self.wide_valley_sectors // 2)
                add_candidate(start + side_offset, width)
                add_candidate(start + width - 1 - side_offset, width)

        # Keep the best score for duplicate angles.
        unique = {}
        for local_angle, cost in scored:
            key = round(local_angle, 6)
            if key not in unique or cost < unique[key][1]:
                unique[key] = (local_angle, cost)
        ordered = sorted(unique.values(), key=lambda item: item[1])

        pending_reverse: Optional[float] = None
        for local_angle, _cost in ordered:
            if self._is_selected_motion_safe(local_angle, obstacle_points):
                if abs(local_angle) > (math.pi * 0.5):
                    if pending_reverse is None:
                        pending_reverse = local_angle
                    continue
                if not self._front_recovery_allows(local_angle, obstacle_points):
                    continue
                return self._commit_selected_direction(local_angle, current_yaw)

        if pending_reverse is not None:
            return self._select_fallback_after_delay(pending_reverse, "reverse", current_yaw)

        rear_clearance = self._get_rear_clearance(obstacle_points)
        if rear_clearance > (self.get_footprint_distance(math.pi) + self.min_obstacle_dist):
            return self._select_fallback_after_delay(math.pi, "reverse", current_yaw)
        self._reset_fallback_waits()
        return None

    def _select_direction_safe(
        self,
        masked_hist: np.ndarray,
        densities: np.ndarray,
        min_distances: np.ndarray,
        obstacle_points: np.ndarray,
        goal_angle_global: float,
        current_yaw: float,
    ) -> Optional[float]:
        """
        Select direction with VFH+ cost function + rotation safety check.

        Falls back to reverse if no safe rotation exists.
        """
        # Convert goal to local angle
        goal_local = self._normalize_angle(goal_angle_global - current_yaw)
        committed = self._try_committed_fallback(goal_local, obstacle_points, current_yaw)
        if committed is not None:
            return committed

        open_indices = np.where(masked_hist)[0]
        if open_indices.size == 0:
            rear_clearance = self._get_rear_clearance(obstacle_points)
            if self._is_forward_hemisphere(goal_local):
                if rear_clearance > (self.get_footprint_distance(math.pi) + self.min_obstacle_dist):
                    return self._select_fallback_after_delay(math.pi, "reverse", current_yaw)
            elif self._is_selected_motion_safe(0.0, obstacle_points):
                return self._select_fallback_after_delay(0.0, "forward", current_yaw)
            self._reset_fallback_waits()
            return None

        # Check if forward hemisphere has any passable sector
        forward_passable = any(
            abs(float(self.sector_centers[idx])) < (math.pi * 0.5) for idx in open_indices
        )
        if forward_passable:
            effective_goal = goal_local
            effective_mu1 = self.mu1
        else:
            # Forward blocked → prefer reverse direction
            effective_goal = self._normalize_angle(goal_local + math.pi)
            effective_mu1 = self.mu1 * 0.5

        candidate_indices = self._candidate_indices_from_openings(masked_hist, effective_goal)
        if not candidate_indices:
            if self._is_forward_hemisphere(goal_local):
                rear_clearance = self._get_rear_clearance(obstacle_points)
                if rear_clearance > (self.get_footprint_distance(math.pi) + self.min_obstacle_dist):
                    return self._select_fallback_after_delay(math.pi, "reverse", current_yaw)
            elif self._is_selected_motion_safe(0.0, obstacle_points):
                return self._select_fallback_after_delay(0.0, "forward", current_yaw)
            self._reset_fallback_waits()
            return None

        # Score VFH+ opening candidates, not every free sector.
        scored: List[Tuple[float, float]] = []
        for idx in candidate_indices:
            local_angle = float(self.sector_centers[idx])
            cost = (
                effective_mu1 * abs(self._angle_diff(local_angle, effective_goal))
                + self.mu2 * abs(self._angle_diff(local_angle, 0.0))  # heading maintenance
                + self.mu3 * abs(self._angle_diff(local_angle, self.prev_selected_local))
                + self.mu_density * float(densities[idx])
            )
            scored.append((local_angle, cost))
        scored.sort(key=lambda x: x[1])

        # Try same-hemisphere candidates first. Opposite-hemisphere candidates are fallbacks.
        pending_fallback: Optional[Tuple[float, str]] = None
        for local_angle, cost in scored:
            if self._is_selected_motion_safe(local_angle, obstacle_points):
                fallback_kind = self._fallback_kind(goal_local, local_angle)
                if fallback_kind is not None:
                    if pending_fallback is None:
                        pending_fallback = (local_angle, fallback_kind)
                    continue
                if not self._front_recovery_allows(local_angle, obstacle_points):
                    continue
                return self._commit_selected_direction(local_angle, current_yaw)

        if pending_fallback is not None:
            return self._select_fallback_after_delay(pending_fallback[0], pending_fallback[1], current_yaw)

        # No candidate has safe rotation → try the opposite hemisphere fallback.
        if self._is_forward_hemisphere(goal_local):
            rear_clearance = self._get_rear_clearance(obstacle_points)
            if rear_clearance > (self.get_footprint_distance(math.pi) + self.min_obstacle_dist):
                best_angle = scored[0][0] if scored else goal_local
                reverse_angle = math.copysign(math.pi, best_angle)
                return self._select_fallback_after_delay(reverse_angle, "reverse", current_yaw)
        elif self._is_selected_motion_safe(0.0, obstacle_points):
            return self._select_fallback_after_delay(0.0, "forward", current_yaw)

        # Nothing works → stop
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

    def _generate_cmd_vel(
        self,
        angle_diff: Optional[float],
        nearest_obstacle: float,
        obstacle_points: np.ndarray,
        density_speed_scale: float = 1.0,
    ) -> Twist:
        cmd = Twist()
        if angle_diff is None:
            return cmd

        rear_clearance = self._get_rear_clearance(obstacle_points)
        is_reverse, heading_err = self._motion_heading_error(angle_diff)
        abs_err = abs(heading_err)

        angular_gain = 1.2
        angular = float(np.clip(heading_err * angular_gain, -self.max_angular_speed, self.max_angular_speed))

        forward_limit = self.max_linear_speed
        reverse_limit = self.max_reverse_speed
        if self.action_speed_limit is not None and self.action_speed_limit > 0.0:
            forward_limit = min(forward_limit, self.action_speed_limit)
            reverse_limit = min(reverse_limit, self.action_speed_limit)

        base_linear = -reverse_limit if is_reverse else forward_limit
        if abs_err < 0.3:
            linear = base_linear
        elif abs_err < 1.0:
            linear = base_linear * 0.4
        else:
            linear = 0.0

        if is_reverse:
            speed_ref = rear_clearance - self.get_footprint_distance(math.pi)
            speed_scale_distance = self.rear_lookahead_distance - self.get_footprint_distance(math.pi)
        else:
            speed_ref = self._get_forward_after_rotation_clearance(angle_diff, obstacle_points)
            speed_scale_distance = self.repulse_radius
        if speed_ref <= self.min_obstacle_dist:
            return cmd

        denom = max(1e-6, speed_scale_distance - self.min_obstacle_dist)
        scale = float(np.clip((speed_ref - self.min_obstacle_dist) / denom, 0.0, 1.0))
        density_scale = float(np.clip(density_speed_scale, 0.0, 1.0))
        linear_cmd = float(linear * scale * density_scale)
        min_speed = self.min_command_linear_speed * density_scale
        if linear_cmd != 0.0 and min_speed > 0.0 and abs(linear_cmd) < min_speed:
            linear_cmd = math.copysign(min_speed, linear_cmd)
        cmd.linear.x = float(np.clip(linear_cmd, -reverse_limit, forward_limit))
        cmd.angular.z = float(np.clip(angular, -self.max_angular_speed, self.max_angular_speed))
        return cmd

    def _publish_stop(self) -> None:
        self.cmd_pub.publish(Twist())

    # ──────────────────── Debug Visualization ────────────────────

    def _publish_footprint_debug(self) -> None:
        poly = PolygonStamped()
        poly.header.frame_id = self.debug_base_frame
        poly.header.stamp = self.get_clock().now().to_msg()
        for x, y in self.self_filter_polygon:
            pt = Point32()
            pt.x, pt.y, pt.z = float(x), float(y), 0.0
            poly.polygon.points.append(pt)
        if self.self_filter_polygon.shape[0] > 0:
            pt = Point32()
            pt.x, pt.y, pt.z = float(self.self_filter_polygon[0, 0]), float(self.self_filter_polygon[0, 1]), 0.0
            poly.polygon.points.append(pt)
        self.debug_footprint_pub.publish(poly)

    def _make_arrow(self, mid, ns, angle, length, color, shaft=0.02, head=0.04, alpha=0.95):
        m = Marker()
        m.header.frame_id, m.header.stamp = self.debug_base_frame, self.get_clock().now().to_msg()
        m.ns, m.id, m.type, m.action = ns, mid, Marker.ARROW, Marker.ADD
        m.scale.x, m.scale.y, m.scale.z = float(shaft), float(head), float(head * 1.5)
        m.color.r, m.color.g, m.color.b, m.color.a = float(color[0]), float(color[1]), float(color[2]), float(alpha)
        p0, p1 = Point(), Point()
        p0.z = p1.z = 0.06
        p1.x, p1.y = float(length * math.cos(angle)), float(length * math.sin(angle))
        m.points = [p0, p1]
        return m

    def _make_sector_arrow(self, idx: int, angle: float, is_passable: bool) -> Marker:
        color = (0.05, 0.85, 0.18) if is_passable else (0.95, 0.08, 0.04)
        return self._make_arrow(
            int(idx),
            "vfh_sector_arrows",
            float(angle),
            self.repulse_radius * 0.72,
            color,
            shaft=0.012,
            head=0.035,
            alpha=0.82,
        )

    def _publish_cloud_debug(self) -> None:
        stamp = self.get_clock().now().to_msg()
        for pub, data in (
            (self.debug_cloud_filtered_pub, self.cloud_points_xyz),
            (self.debug_cloud_rejected_pub, self.cloud_points_xyz_rejected),
        ):
            header = Header()
            header.frame_id = self.debug_base_frame
            header.stamp = stamp
            pub.publish(point_cloud2.create_cloud_xyz32(header, data.tolist() if data.size > 0 else []))

    def _publish_marker_debug(self, points, masked_hist, goal_angle, yaw, selected_local, nearest):
        markers = MarkerArray()
        stamp = self.get_clock().now().to_msg()

        for old_id in (0, 1):
            dm = Marker()
            dm.header.frame_id, dm.header.stamp = self.debug_base_frame, stamp
            dm.ns, dm.id, dm.action = "vfh_sectors", old_id, Marker.DELETE
            markers.markers.append(dm)

        for i, angle in enumerate(self.sector_centers):
            marker = self._make_sector_arrow(i, float(angle), bool(masked_hist[i]))
            marker.header.stamp = stamp
            markers.markers.append(marker)

        old_status = Marker()
        old_status.header.frame_id, old_status.header.stamp = self.debug_base_frame, stamp
        old_status.ns, old_status.id, old_status.action = "vfh_status", 4, Marker.DELETE
        markers.markers.append(old_status)

        local_goal = self._normalize_angle(goal_angle - yaw)
        markers.markers.append(self._make_arrow(2, "vfh_goal", local_goal, 0.9, (1.0, 0.8, 0.1)))

        if selected_local is not None:
            markers.markers.append(
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
            markers.markers.append(dm)

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
        markers.markers.append(pts_m)

        for mid, data, color, alpha in (
            (6, self.cloud_points_xyz, (0.1, 0.9, 1.0), 0.95),
            (7, self.cloud_points_xyz_rejected, (1.0, 0.2, 1.0), 0.35),
        ):
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
            markers.markers.append(cm)

        st = Marker()
        st.header.frame_id, st.header.stamp = self.debug_base_frame, stamp
        st.ns, st.id, st.type, st.action = "vfh_status", 5, Marker.TEXT_VIEW_FACING, Marker.ADD
        st.pose.orientation.w, st.pose.position.z, st.scale.z = 1.0, 0.25, 0.08
        st.color.r, st.color.g, st.color.b, st.color.a = 1.0, 1.0, 1.0, 0.9
        sel_deg = math.degrees(selected_local) if selected_local is not None else float("nan")
        passable = int(np.count_nonzero(masked_hist))
        st.text = (
            f"near={nearest:.2f}m sel={sel_deg:.1f}deg "
            f"pass={passable}/{self.num_sectors} "
            f"cloud={self.cloud_points_xy.shape[0]}/{self.cloud_points_xy_raw.shape[0]}"
        )
        markers.markers.append(st)

        self.debug_marker_pub.publish(markers)

    def _publish_debug_state(
        self,
        obstacle_points,
        densities,
        min_distances,
        masked_hist,
        goal_mode,
        goal_angle,
        yaw,
        selected_local,
        cmd_pred,
        nearest,
        selected_density: float = 0.0,
        density_speed_scale: float = 1.0,
    ):
        self._publish_footprint_debug()
        self._publish_cloud_debug()
        self._publish_marker_debug(obstacle_points, masked_hist, goal_angle, yaw, selected_local, nearest)

        sm = Float32MultiArray()
        sm.data = np.where(np.isfinite(min_distances), min_distances, -1.0).astype(np.float32).tolist()
        self.debug_sector_dist_pub.publish(sm)

        sd = Float32MultiArray()
        sd.data = densities.astype(np.float32).tolist()
        self.debug_sector_density_pub.publish(sd)

        now_sec = time.monotonic()
        if now_sec - self.last_debug_info_time < self.debug_info_period:
            return
        self.last_debug_info_time = now_sec
        info = String()
        sel_deg = math.degrees(selected_local) if selected_local is not None else float("nan")
        goal_local_deg = math.degrees(self._normalize_angle(goal_angle - yaw))
        passable = int(np.count_nonzero(masked_hist))
        traveled = 0.0
        if self.current_pose is not None:
            traveled = self._distance_xy((self.current_pose[0], self.current_pose[1]), self.escape_start_xy)
        fallback_hold_remaining = max(0.0, self.fallback_commit_until - now_sec)
        info.data = (
            f"dry_run={self.dry_run_only} mode={goal_mode} escape_mode={self.current_escape_mode.value} "
            f"goal_local={goal_local_deg:.1f}deg "
            f"selected={sel_deg:.1f}deg passable={passable}/{self.num_sectors} "
            f"traveled={traveled:.2f}/{self.active_escape_distance:.2f}m "
            f"nearest={nearest:.3f}m cmd=({cmd_pred.linear.x:.3f},{cmd_pred.angular.z:.3f}) "
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
        self.debug_info_pub.publish(info)

    def _publish_idle_debug(self, mode: str) -> None:
        obstacle_points = self._compose_obstacle_points()
        densities, min_distances = self._build_polar_histogram(obstacle_points)
        _binary_hist = self._build_binary_histogram(densities, min_distances, obstacle_points)
        masked_hist = self._build_masked_histogram(obstacle_points)

        yaw = self.current_pose[2] if self.current_pose is not None else 0.0
        goal_angle = yaw
        selected_local: Optional[float] = None
        if masked_hist.size > 0:
            front_idx = self._angle_to_sector_index(0.0)
            if bool(masked_hist[front_idx]):
                selected_local = 0.0
        nearest = float(np.min(np.hypot(obstacle_points[:, 0], obstacle_points[:, 1]))) if obstacle_points.size > 0 else float("inf")
        self._publish_debug_state(
            obstacle_points,
            densities,
            min_distances,
            masked_hist,
            mode,
            goal_angle,
            yaw,
            selected_local,
            Twist(),
            nearest,
        )

    # ──────────────────── Nav2 Handoff ────────────────────

    def _finish_escape(self, success: bool, message: str) -> None:
        if not self.escape_active:
            return
        self.escape_active = False
        self.escape_result = success
        self.escape_result_message = message
        self.controller_trial_active = False
        self.action_speed_limit = None
        self.action_preferred_local_angle = None
        self.action_preferred_global_angle = None
        self._reset_fallback_waits()
        self._clear_fallback_commit()
        self._reset_front_recovery_block()
        self._publish_stop()
        (self.get_logger().info if success else self.get_logger().warn)(message)

    def _current_pose_stamped(self, frame_id: str = "map") -> Optional[PoseStamped]:
        if self.current_pose is None:
            return None
        x, y, yaw = self.current_pose
        p = PoseStamped()
        p.header.frame_id, p.header.stamp = frame_id, self.get_clock().now().to_msg()
        p.pose.position.x, p.pose.position.y = x, y
        p.pose.orientation.w, p.pose.orientation.z = math.cos(0.5 * yaw), math.sin(0.5 * yaw)
        return p

    def _maybe_handoff_planner(self, now_sec: float, nearest: float) -> None:
        if not self.nav2_handoff_enabled or self.navigator is None:
            return
        if self.current_pose is None or self.goal_pose is None:
            return
        if (now_sec - self.last_nav2_check_time) < self.nav2_check_interval:
            return
        self.last_nav2_check_time = now_sec
        x, y, _ = self.current_pose
        if self._distance_xy((x, y), self.escape_start_xy) < 0.3:
            return
        robot_r = self.footprint_radius
        if nearest < (robot_r + 0.3):
            return
        start = self._current_pose_stamped(frame_id=self.goal_pose.header.frame_id or "map")
        if start is None:
            return
        try:
            path = self.navigator.getPath(start, self.goal_pose)
        except Exception:
            return
        if path is None or len(path.poses) == 0:
            return
        self._publish_stop()
        try:
            self.navigator.goToPose(self.goal_pose)
        except Exception:
            return
        self._finish_escape(True, "Planner recovered. Handed control to Nav2.")

    def _start_controller_trial(self, now_sec: float) -> None:
        if not self.nav2_handoff_enabled or self.navigator is None:
            return
        if self.goal_pose is None or self.current_pose is None:
            return
        x, y, _ = self.current_pose
        self._publish_stop()
        try:
            self.navigator.goToPose(self.goal_pose)
        except Exception:
            self.controller_fail_count += 1
            if self.controller_fail_count >= 3:
                self._finish_escape(False, "Controller recovery failed 3 times.")
            return
        self.controller_trial_active = True
        self.controller_trial_start_time = now_sec
        self.controller_trial_start_xy = (x, y)
        self.controller_anchor_xy = (x, y)

    def _monitor_controller_trial(self, now_sec: float) -> None:
        if not self.controller_trial_active or self.current_pose is None:
            return
        x, y, _ = self.current_pose
        moved = self._distance_xy((x, y), self.controller_trial_start_xy)
        elapsed = now_sec - self.controller_trial_start_time
        if moved >= 0.3:
            self._finish_escape(True, "Controller recovered.")
            return
        done = False
        try:
            done = bool(self.navigator.isTaskComplete())
        except Exception:
            pass
        if elapsed < 5.0 and not done:
            return
        try:
            self.navigator.cancelTask()
        except Exception:
            pass
        self.controller_trial_active = False
        self.controller_fail_count += 1
        if self.controller_fail_count >= 3:
            self._finish_escape(False, "Controller recovery failed 3 times.")

    def _maybe_handoff_controller(self, now_sec: float) -> None:
        if self.current_pose is None:
            return
        x, y, _ = self.current_pose
        if self._distance_xy((x, y), self.controller_anchor_xy) >= 0.5:
            self._start_controller_trial(now_sec)

    # ──────────────────── Action Interface ────────────────────

    def _action_goal_callback(self, _goal_request) -> GoalResponse:
        if self.escape_active:
            self.get_logger().warn("Rejecting VFH escape action: escape already active.")
            return GoalResponse.REJECT
        return GoalResponse.ACCEPT

    def _action_cancel_callback(self, _goal_handle) -> CancelResponse:
        return CancelResponse.ACCEPT

    def _execute_action(self, goal_handle):
        goal = goal_handle.request
        reason = self._reason_from_action_goal(goal.reason)
        mode = self._mode_from_action_goal(goal.mode, reason)
        preferred_local_angle = None
        if mode == EscapeMode.PREFERRED_DIRECTION:
            preferred_local_angle = self._preferred_direction_angle(goal.preferred_direction)
            if preferred_local_angle is None:
                result = VfhPlusEscape.Result()
                result.success = False
                result.message = "preferred_direction must have a non-zero x/y vector."
                goal_handle.abort()
                return result

        max_distance = float(goal.max_distance)
        timeout = float(goal.timeout_sec)
        speed_limit = abs(float(goal.speed_limit)) if float(goal.speed_limit) > 0.0 else None

        self._start_escape(
            goal_pose=None,
            reason=reason,
            block=False,
            mode=mode,
            max_distance=max_distance if max_distance > 0.0 else None,
            preferred_local_angle=preferred_local_angle,
            timeout=timeout if timeout > 0.0 else None,
            speed_limit=speed_limit,
        )

        start_time = time.monotonic()
        result = VfhPlusEscape.Result()

        while rclpy.ok():
            elapsed = time.monotonic() - start_time
            if goal_handle.is_cancel_requested:
                self._finish_escape(False, "VFH escape action canceled.")
                goal_handle.canceled()
                result.success = False
                result.message = self.escape_result_message
                result.distance_traveled = self._current_escape_distance()
                result.elapsed_sec = float(elapsed)
                return result

            feedback = VfhPlusEscape.Feedback()
            feedback.state = self.current_escape_mode.value if self.escape_active else "finished"
            feedback.distance_traveled = self._current_escape_distance()
            feedback.elapsed_sec = float(elapsed)
            goal_handle.publish_feedback(feedback)

            if not self.escape_active:
                break
            time.sleep(0.1)

        elapsed = time.monotonic() - start_time
        result.success = bool(self.escape_result)
        result.message = self.escape_result_message
        result.distance_traveled = self._current_escape_distance()
        result.elapsed_sec = float(elapsed)
        if self.escape_result:
            goal_handle.succeed()
        else:
            goal_handle.abort()
        return result

    # ──────────────────── External Interface ────────────────────

    def _start_escape(
        self,
        goal_pose: Optional[PoseStamped],
        reason: EscapeReason,
        block: bool,
        mode: Optional[EscapeMode] = None,
        max_distance: Optional[float] = None,
        preferred_local_angle: Optional[float] = None,
        timeout: Optional[float] = None,
        speed_limit: Optional[float] = None,
    ) -> Optional[bool]:
        was_active = self.escape_active
        if was_active and self.controller_trial_active and self.navigator is not None:
            try:
                self.navigator.cancelTask()
            except Exception:
                pass
        self.goal_pose = goal_pose
        self.escape_reason = reason
        self.current_escape_mode = mode if mode is not None else self._mode_for_reason(reason)
        self.escape_active = True
        self.escape_result = False
        self.escape_result_message = ""
        self.escape_start_time = time.monotonic()
        self.active_escape_timeout = max(0.1, float(timeout)) if timeout is not None and timeout > 0.0 else self.escape_timeout
        if max_distance is None:
            max_distance = self._distance_for_reason(reason)
        self.active_escape_distance = max(0.0, float(max_distance))
        self.action_preferred_local_angle = preferred_local_angle
        self.action_preferred_global_angle = None
        self.action_speed_limit = speed_limit if speed_limit is not None and speed_limit > 0.0 else None
        self.last_nav2_check_time = self.escape_start_time
        self.controller_fail_count = 0
        self.controller_trial_active = False
        self._reset_fallback_waits()
        self._clear_fallback_commit()
        self._reset_front_recovery_block()
        self.escape_start_pose_pending = self.current_pose is None
        if self.current_pose is not None:
            x, y, yaw = self.current_pose
            self.escape_start_xy = (x, y)
            self.controller_anchor_xy = (x, y)
            self.prev_direction_global = yaw
            self.prev_selected_local = 0.0
            if self.action_preferred_local_angle is not None:
                self.action_preferred_global_angle = self._normalize_angle(
                    yaw + self.action_preferred_local_angle
                )
        self.prev_binary_hist = np.ones(self.num_sectors, dtype=bool)
        action = "restarted" if was_active else "started"
        frame = goal_pose.header.frame_id if goal_pose is not None and goal_pose.header.frame_id else self.debug_base_frame
        self.get_logger().info(
            "Escape %s. reason=%s mode=%s distance=%.2fm timeout=%.1fs frame=%s"
            % (action, reason.name, self.current_escape_mode.value, self.active_escape_distance, self.active_escape_timeout, frame)
        )
        if not block:
            return None
        while rclpy.ok() and self.escape_active:
            rclpy.spin_once(self, timeout_sec=0.1)
        self._publish_stop()
        self.get_logger().info("Escape finished. result=%s" % self.escape_result)
        return self.escape_result

    def escape(self, goal_pose: PoseStamped, reason: EscapeReason) -> bool:
        result = self._start_escape(goal_pose, reason, block=True)
        return bool(result)

    # ──────────────────── Main Loop ────────────────────

    def _control_loop(self) -> None:
        if not self.escape_active:
            self._publish_idle_debug("WAIT_GOAL")
            return

        now_sec = time.monotonic()

        if self.current_pose is None:
            if now_sec - self.last_no_sensor_warn > 1.0:
                self.get_logger().warn("Waiting for /odom.")
                self.last_no_sensor_warn = now_sec
            self._publish_stop()
            self._publish_footprint_debug()
            self._publish_idle_debug("WAIT_ODOM")
            return

        if self.escape_start_pose_pending:
            x, y, yaw = self.current_pose
            self.escape_start_xy = (x, y)
            self.controller_anchor_xy = (x, y)
            self.prev_direction_global = yaw
            self.prev_selected_local = 0.0
            if self.action_preferred_local_angle is not None and self.action_preferred_global_angle is None:
                self.action_preferred_global_angle = self._normalize_angle(
                    yaw + self.action_preferred_local_angle
                )
            self._reset_fallback_waits()
            self._clear_fallback_commit()
            self._reset_front_recovery_block()
            self.escape_start_pose_pending = False

        if (now_sec - self.escape_start_time) > self.active_escape_timeout:
            self._finish_escape(False, "Escape timeout.")
            return

        if self.controller_trial_active:
            self._monitor_controller_trial(now_sec)
            return

        obstacle_points = self._compose_obstacle_points()
        if obstacle_points.size == 0:
            if now_sec - self.last_no_sensor_warn > 1.0:
                self.get_logger().warn("No sensor data. Holding stop.")
                self.last_no_sensor_warn = now_sec
            self._publish_stop()
            self._publish_footprint_debug()
            self._publish_idle_debug("NO_SENSOR")
            return

        # VFH+ pipeline
        densities, min_distances = self._build_polar_histogram(obstacle_points)
        binary_hist = self._build_binary_histogram(densities, min_distances, obstacle_points)
        masked_hist = self._build_masked_histogram(obstacle_points)

        yaw = self.current_pose[2]
        goal_angle, goal_mode = self._determine_preferred_goal_angle(yaw)

        if self.current_escape_mode == EscapeMode.WIDEST_OPENING:
            selected_local = self._select_widest_opening_direction(
                masked_hist, densities, min_distances, obstacle_points, yaw
            )
        else:
            selected_local = self._select_direction_safe(
                masked_hist, densities, min_distances, obstacle_points, goal_angle, yaw
            )

        nearest = float(np.min(np.hypot(obstacle_points[:, 0], obstacle_points[:, 1])))
        selected_density = self._selected_density(selected_local, densities)
        density_speed_scale = self._density_speed_scale(selected_density)
        cmd_pred = self._generate_cmd_vel(selected_local, nearest, obstacle_points, density_speed_scale)

        self._publish_debug_state(
            obstacle_points,
            densities,
            min_distances,
            masked_hist,
            goal_mode,
            goal_angle,
            yaw,
            selected_local,
            cmd_pred,
            nearest,
            selected_density,
            density_speed_scale,
        )

        if self.dry_run_only:
            self._publish_stop()
            return

        traveled = self._distance_xy((self.current_pose[0], self.current_pose[1]), self.escape_start_xy)
        if self.active_escape_distance > 0.0 and traveled >= self.active_escape_distance:
            self._finish_escape(True, "Escape distance reached %.2fm." % traveled)
            return

        self.cmd_pub.publish(cmd_pred)

        if not self.nav2_handoff_enabled:
            return

        if self.escape_reason == EscapeReason.PLANNER_FAILED:
            self._maybe_handoff_planner(now_sec, nearest)
        elif self.escape_reason == EscapeReason.CONTROLLER_FAILED:
            self._maybe_handoff_controller(now_sec)


def _spin_node(node: VFHPlusEscapeNode) -> None:
    executor = MultiThreadedExecutor(num_threads=2)
    executor.add_node(node)
    try:
        executor.spin()
    finally:
        executor.shutdown()


def main(args: Optional[Sequence[str]] = None) -> None:
    rclpy.init(args=args)
    node = VFHPlusEscapeNode()
    try:
        _spin_node(node)
    except KeyboardInterrupt:
        pass
    finally:
        node._publish_stop()
        node.destroy_node()
        rclpy.shutdown()


def main_escape(args: Optional[Sequence[str]] = None) -> None:
    rclpy.init(args=args)
    node = VFHPlusEscapeNode(node_name="vfh_escape_node")
    try:
        _spin_node(node)
    except KeyboardInterrupt:
        pass
    finally:
        node._publish_stop()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
