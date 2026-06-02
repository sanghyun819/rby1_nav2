#!/usr/bin/env python3
"""
Map ↔ Odom TF Broadcaster
============================
AMCL 없이 map→odom TF를 발행합니다.

동작:
  1. 시작 시 map→odom을 항등 변환(identity)으로 발행
  2. RViz2의 "2D Pose Estimate" (/initialpose) 수신 시:
     - 사용자가 지정한 pose = map→base 변환
     - 현재 odom→base TF를 조회
     - map→odom = map→base * (odom→base)^(-1) 계산
     - 이후 계속 이 map→odom을 발행

토픽:
  구독: /initialpose (geometry_msgs/PoseWithCovarianceStamped)
  발행: /tf (map→odom)
"""

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseWithCovarianceStamped, TransformStamped
import tf2_ros
import numpy as np
import math


def quat_to_mat(q):
    """Quaternion (x,y,z,w) → 4×4 homogeneous matrix"""
    x, y, z, w = q
    return np.array([
        [1 - 2*(y*y + z*z), 2*(x*y - z*w),     2*(x*z + y*w),     0],
        [2*(x*y + z*w),     1 - 2*(x*x + z*z), 2*(y*z - x*w),     0],
        [2*(x*z - y*w),     2*(y*z + x*w),     1 - 2*(x*x + y*y), 0],
        [0,                 0,                 0,                   1],
    ])


def mat_to_quat(m):
    """3×3 rotation → Quaternion (x,y,z,w)"""
    tr = m[0, 0] + m[1, 1] + m[2, 2]
    if tr > 0:
        s = 0.5 / math.sqrt(tr + 1.0)
        w = 0.25 / s
        x = (m[2, 1] - m[1, 2]) * s
        y = (m[0, 2] - m[2, 0]) * s
        z = (m[1, 0] - m[0, 1]) * s
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = 2.0 * math.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2])
        w = (m[2, 1] - m[1, 2]) / s
        x = 0.25 * s
        y = (m[0, 1] + m[1, 0]) / s
        z = (m[0, 2] + m[2, 0]) / s
    elif m[1, 1] > m[2, 2]:
        s = 2.0 * math.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2])
        w = (m[0, 2] - m[2, 0]) / s
        x = (m[0, 1] + m[1, 0]) / s
        y = 0.25 * s
        z = (m[1, 2] + m[2, 1]) / s
    else:
        s = 2.0 * math.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1])
        w = (m[1, 0] - m[0, 1]) / s
        x = (m[0, 2] + m[2, 0]) / s
        y = (m[1, 2] + m[2, 1]) / s
        z = 0.25 * s
    return (x, y, z, w)


def pose_to_mat(pos, ori):
    """Position + Quaternion → 4×4 homogeneous matrix"""
    m = quat_to_mat((ori.x, ori.y, ori.z, ori.w))
    m[0, 3] = pos.x
    m[1, 3] = pos.y
    m[2, 3] = pos.z
    return m


class MapPoseBroadcaster(Node):
    def __init__(self):
        super().__init__('map_pose_broadcaster')

        # Parameters
        self.declare_parameter('map_frame', 'map')
        self.declare_parameter('odom_frame', 'odom')
        self.declare_parameter('base_frame', 'base')
        self.declare_parameter('broadcast_rate', 20.0)

        self.map_frame = self.get_parameter('map_frame').value
        self.odom_frame = self.get_parameter('odom_frame').value
        self.base_frame = self.get_parameter('base_frame').value
        rate = self.get_parameter('broadcast_rate').value

        # TF
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        self.tf_broadcaster = tf2_ros.TransformBroadcaster(self)

        # map→odom transform (starts as identity)
        self.map_to_odom = np.eye(4)

        # Subscribe to /initialpose (RViz2's "2D Pose Estimate")
        self.sub = self.create_subscription(
            PoseWithCovarianceStamped,
            '/initialpose',
            self._initialpose_cb,
            10
        )

        # Periodic broadcast
        self.timer = self.create_timer(1.0 / rate, self._broadcast_tf)

        self.get_logger().info(
            f'Map Pose Broadcaster started: {self.map_frame}→{self.odom_frame}'
        )
        self.get_logger().info(
            '  Use RViz2 "2D Pose Estimate" to set robot position on map'
        )

    def _initialpose_cb(self, msg: PoseWithCovarianceStamped):
        """
        RViz2 "2D Pose Estimate" 수신 → map→odom 계산

        사용자가 지정한 pose는 map 프레임 기준 로봇 위치 (map→base).
        현재 odom→base를 조회해서:
          map→odom = map→base * (odom→base)^(-1)
        """
        # map→base (user specified)
        pos = msg.pose.pose.position
        ori = msg.pose.pose.orientation
        map_to_base = pose_to_mat(pos, ori)

        self.get_logger().info(
            f'Received 2D Pose Estimate: '
            f'x={pos.x:.3f}, y={pos.y:.3f}, '
            f'yaw={math.degrees(math.atan2(2*(ori.w*ori.z + ori.x*ori.y), 1 - 2*(ori.y**2 + ori.z**2))):.1f}°'
        )

        # odom→base 조회
        try:
            tf = self.tf_buffer.lookup_transform(
                self.odom_frame, self.base_frame,
                rclpy.time.Time(),
                timeout=rclpy.duration.Duration(seconds=1.0)
            )
            t = tf.transform.translation
            r = tf.transform.rotation
            odom_to_base = pose_to_mat(t, r)
        except Exception as e:
            self.get_logger().warn(
                f'Cannot lookup {self.odom_frame}→{self.base_frame}: {e}'
            )
            self.get_logger().warn(
                'Using identity for odom→base (map→odom = map→base)'
            )
            odom_to_base = np.eye(4)

        # map→odom = map→base * (odom→base)^(-1)
        base_to_odom = np.linalg.inv(odom_to_base)
        self.map_to_odom = map_to_base @ base_to_odom

        self.get_logger().info(
            f'Updated {self.map_frame}→{self.odom_frame}: '
            f'tx={self.map_to_odom[0,3]:.3f}, ty={self.map_to_odom[1,3]:.3f}, '
            f'yaw={math.degrees(math.atan2(self.map_to_odom[1,0], self.map_to_odom[0,0])):.1f}°'
        )

    def _broadcast_tf(self):
        """map→odom TF 주기적 발행"""
        t = TransformStamped()
        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id = self.map_frame
        t.child_frame_id = self.odom_frame

        t.transform.translation.x = float(self.map_to_odom[0, 3])
        t.transform.translation.y = float(self.map_to_odom[1, 3])
        t.transform.translation.z = float(self.map_to_odom[2, 3])

        q = mat_to_quat(self.map_to_odom[:3, :3])
        t.transform.rotation.x = q[0]
        t.transform.rotation.y = q[1]
        t.transform.rotation.z = q[2]
        t.transform.rotation.w = q[3]

        self.tf_broadcaster.sendTransform(t)


def main(args=None):
    rclpy.init(args=args)
    node = MapPoseBroadcaster()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
