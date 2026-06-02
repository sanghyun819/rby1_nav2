#!/usr/bin/env python3
"""
Height Costmap Node (Inflation style)
=======================================
PCD 높이 데이터에서 장애물을 추출하고,
Nav2 inflation_layer처럼 장애물 주변으로 cost가 퍼져나가는
히트맵 스타일 costmap을 생성합니다.

토픽:
  /map_2d                (OccupancyGrid) - 기존 2D 맵 (pgm)
  /height_costmap        (OccupancyGrid) - height 기반 inflated costmap
  /height_costmap_visual (PointCloud2)   - RViz2 컬러 히트맵 시각화
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSDurabilityPolicy, QoSReliabilityPolicy, QoSHistoryPolicy
from nav_msgs.msg import OccupancyGrid
from sensor_msgs.msg import PointCloud2, PointField
import numpy as np
import cv2
import yaml
import os
import struct


class HeightCostmapNode(Node):
    def __init__(self):
        super().__init__('height_costmap_node')

        # ── 파라미터 ──
        self.declare_parameter('pcd_file', '')
        self.declare_parameter('map_yaml', '')
        self.declare_parameter('publish_rate', 1.0)
        self.declare_parameter('obstacle_z_min', 0.15)      # 이 높이 이상이면 장애물
        self.declare_parameter('obstacle_z_max', 2.5)       # 천장 제외
        self.declare_parameter('inflation_radius', 1.0)     # inflation 반경 (m)
        self.declare_parameter('cost_scaling_factor', 3.0)   # 지수 감쇠 계수
        self.declare_parameter('frame_id', 'map')

        pcd_file = self.get_parameter('pcd_file').value
        map_yaml = self.get_parameter('map_yaml').value
        publish_rate = self.get_parameter('publish_rate').value
        self.frame_id = self.get_parameter('frame_id').value
        self.obs_z_min = self.get_parameter('obstacle_z_min').value
        self.obs_z_max = self.get_parameter('obstacle_z_max').value
        self.inflation_radius = self.get_parameter('inflation_radius').value
        self.cost_scaling = self.get_parameter('cost_scaling_factor').value

        if not pcd_file or not map_yaml:
            self.get_logger().error('pcd_file and map_yaml parameters required!')
            return

        # ── map yaml/pgm 로드 ──
        self.get_logger().info(f'Loading map: {map_yaml}')
        with open(map_yaml, 'r') as f:
            map_info = yaml.safe_load(f)

        map_dir = os.path.dirname(os.path.abspath(map_yaml))
        pgm_path = os.path.join(map_dir, map_info['image'])
        self.map_img = cv2.imread(pgm_path, cv2.IMREAD_GRAYSCALE)

        self.resolution = float(map_info['resolution'])
        self.origin = map_info['origin']
        self.map_h, self.map_w = self.map_img.shape

        self.get_logger().info(
            f'  Size: {self.map_w}x{self.map_h}, res={self.resolution}m, '
            f'origin=({self.origin[0]:.2f}, {self.origin[1]:.2f})'
        )

        # ── 2D map → OccupancyGrid 데이터 ──
        self.map_occ_data = self._pgm_to_occupancy(self.map_img)

        # ── PCD → inflated height costmap ──
        self.get_logger().info(f'Loading PCD: {pcd_file}')
        self.costmap_data, self._pc2_n_pts, self._pc2_bytes = self._build_inflated_costmap(pcd_file)

        # ── Publishers (transient_local = latched) ──
        latched_qos = QoSProfile(
            reliability=QoSReliabilityPolicy.RELIABLE,
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1
        )

        self.map_pub = self.create_publisher(OccupancyGrid, '/map_2d', latched_qos)
        self.cost_pub = self.create_publisher(OccupancyGrid, '/height_costmap', latched_qos)
        self.pc2_pub = self.create_publisher(PointCloud2, '/height_costmap_visual', latched_qos)

        self.timer = self.create_timer(1.0 / publish_rate, self._publish)

        self.get_logger().info('=' * 55)
        self.get_logger().info('Height Costmap Node (inflation style) ready!')
        self.get_logger().info('')
        self.get_logger().info('Topics:')
        self.get_logger().info('  /map_2d                → 기존 2D 맵')
        self.get_logger().info('  /height_costmap        → inflated costmap')
        self.get_logger().info('  /height_costmap_visual → 컬러 히트맵 (PointCloud2)')
        self.get_logger().info('')
        self.get_logger().info('RViz2:')
        self.get_logger().info('  Fixed Frame: map')
        self.get_logger().info('  Add /map_2d        → Map (Color: map)')
        self.get_logger().info('  Add /height_costmap_visual → PointCloud2')
        self.get_logger().info('    Style: Flat Squares, Size: 0.05')
        self.get_logger().info('    Color Transformer: RGB8, Alpha: 0.7')
        self.get_logger().info('=' * 55)

    def _pgm_to_occupancy(self, img):
        """PGM 이미지 → OccupancyGrid data (0=free, 100=occ, -1=unknown)"""
        h, w = img.shape
        occ = np.full(h * w, -1, dtype=np.int8)
        flat = img.flatten()
        occ[flat > 230] = 0      # free (white)
        occ[flat < 30] = 100     # occupied (black)
        # pgm은 위→아래, OccupancyGrid는 아래→위
        occ_2d = np.flipud(occ.reshape(h, w))
        return occ_2d.flatten().tolist()

    def _read_pcd(self, filepath):
        header_lines = 0
        with open(filepath, 'r') as f:
            for line in f:
                header_lines += 1
                if line.startswith('DATA'):
                    break
        data = np.loadtxt(filepath, skiprows=header_lines)
        return data[:, 0], data[:, 1], data[:, 2]

    def _cost_to_rgb(self, cost_norm):
        """
        cost 비율 (0~1) → 히트맵 RGB
        파랑 → 초록 → 노랑 → 빨강 색상 스킴
        """
        n = len(cost_norm)
        r = np.zeros(n, dtype=np.uint8)
        g = np.zeros(n, dtype=np.uint8)
        b = np.zeros(n, dtype=np.uint8)

        # 0.0~0.33: 파랑 → 초록
        m = cost_norm < 0.33
        t = cost_norm[m] / 0.33
        r[m] = 0
        g[m] = (255 * t).astype(np.uint8)
        b[m] = (255 * (1 - t)).astype(np.uint8)

        # 0.33~0.66: 초록 → 노랑
        m = (cost_norm >= 0.33) & (cost_norm < 0.66)
        t = (cost_norm[m] - 0.33) / 0.33
        r[m] = (255 * t).astype(np.uint8)
        g[m] = 255
        b[m] = 0

        # 0.66~1.0: 노랑 → 빨강
        m = cost_norm >= 0.66
        t = (cost_norm[m] - 0.66) / 0.34
        r[m] = 255
        g[m] = (255 * (1 - t)).astype(np.uint8)
        b[m] = 0

        return r, g, b

    def _build_inflated_costmap(self, pcd_file):
        """
        PCD에서 높이 장애물을 추출하고,
        inflation_layer처럼 장애물 주변으로 cost가 퍼져나가는 costmap 생성
        """
        x, y, z = self._read_pcd(pcd_file)
        self.get_logger().info(f'  Points: {len(x):,}, Z=[{z.min():.2f}, {z.max():.2f}]')

        W, H = self.map_w, self.map_h
        res = self.resolution
        ox, oy = self.origin[0], self.origin[1]

        # 포인트 → 그리드 인덱스
        col = ((x - ox) / res).astype(int)
        row = ((y - oy) / res).astype(int)

        valid = (col >= 0) & (col < W) & (row >= 0) & (row < H)
        col, row, z_v = col[valid], row[valid], z[valid]
        self.get_logger().info(f'  Points in map: {len(col):,}')

        # 각 셀의 최대 Z
        height_map = np.full((H, W), np.nan, dtype=np.float32)
        idx = np.argsort(z_v)
        height_map[row[idx], col[idx]] = z_v[idx]

        # ── Step 1: 장애물 마스크 (높이가 threshold 이상인 셀) ──
        obstacle_mask = (~np.isnan(height_map)) & \
                        (height_map >= self.obs_z_min) & \
                        (height_map <= self.obs_z_max)

        n_obs = np.count_nonzero(obstacle_mask)
        self.get_logger().info(f'  Obstacle cells (z>={self.obs_z_min}m): {n_obs:,}')

        # ── Step 2: Distance Transform (장애물로부터의 거리) ──
        # obstacle=0(검정), free=255(흰색) 이미지로 만들어서 distanceTransform
        obs_img = np.ones((H, W), dtype=np.uint8) * 255
        obs_img[obstacle_mask] = 0

        # 유클리드 거리 (픽셀 단위)
        dist_px = cv2.distanceTransform(obs_img, cv2.DIST_L2, 5)
        dist_m = dist_px * res  # 미터 단위

        # ── Step 3: 거리 → cost (Nav2 inflation 스타일 지수 감쇠) ──
        # cost = 100 * exp(-cost_scaling * (dist - inscribed_radius))
        # inscribed_radius = 0 (셀 자체)
        inflation_r = self.inflation_radius
        scaling = self.cost_scaling

        costmap = np.full((H, W), -1, dtype=np.int8)  # -1 = unknown

        # 데이터가 있는 영역만 처리
        has_data = ~np.isnan(height_map)

        # 장애물 셀 = 100 (lethal)
        costmap[obstacle_mask] = 100

        # inflation 영역: 장애물이 아니고, 데이터가 있고, inflation 반경 이내
        inflate_mask = has_data & (~obstacle_mask) & (dist_m <= inflation_r) & (dist_m > 0)
        if np.any(inflate_mask):
            d = dist_m[inflate_mask]
            # Nav2 스타일: cost = 252 * exp(-scaling * (d - 0))
            cost_float = 99.0 * np.exp(-scaling * d)
            cost_int = np.clip(cost_float, 1, 99).astype(np.int8)
            costmap[inflate_mask] = cost_int

        # 데이터 있지만 inflation 밖 = free (0)
        free_mask = has_data & (~obstacle_mask) & (dist_m > inflation_r)
        costmap[free_mask] = 0

        # 통계
        lethal_cnt = np.count_nonzero(costmap == 100)
        inflated_cnt = np.count_nonzero((costmap > 0) & (costmap < 100))
        free_cnt = np.count_nonzero(costmap == 0)
        unk_cnt = np.count_nonzero(costmap == -1)
        self.get_logger().info(
            f'  Inflated costmap: lethal={lethal_cnt:,}, '
            f'inflated={inflated_cnt:,}, free={free_cnt:,}, unknown={unk_cnt:,}'
        )
        self.get_logger().info(
            f'  Inflation radius={inflation_r}m, scaling={scaling}'
        )

        occ_data = costmap.flatten().tolist()

        # ── PointCloud2 히트맵 시각화 (cost > 0 인 셀만) ──
        # inflation 영역 + obstacle 모두 시각화
        vis_mask = has_data & (costmap > 0)
        vis_rows, vis_cols = np.where(vis_mask)
        vis_cost = costmap[vis_rows, vis_cols].astype(np.float64) / 100.0

        px = (ox + (vis_cols + 0.5) * res).astype(np.float32)
        py = (oy + (vis_rows + 0.5) * res).astype(np.float32)
        pz = np.zeros_like(px)

        r, g, b = self._cost_to_rgb(vis_cost)

        n_pts = len(px)
        rgb_uint32 = (r.astype(np.uint32) << 16) | (g.astype(np.uint32) << 8) | b.astype(np.uint32)
        rgb_float = np.empty(n_pts, dtype=np.float32)
        for i in range(n_pts):
            rgb_float[i] = struct.unpack('f', struct.pack('I', int(rgb_uint32[i])))[0]

        pc_buf = np.empty(n_pts, dtype=[
            ('x', np.float32), ('y', np.float32), ('z', np.float32), ('rgb', np.float32)
        ])
        pc_buf['x'] = px
        pc_buf['y'] = py
        pc_buf['z'] = pz
        pc_buf['rgb'] = rgb_float

        pc2_bytes = pc_buf.tobytes()
        self.get_logger().info(f'  PointCloud2 visual: {n_pts:,} points')

        return occ_data, n_pts, pc2_bytes

    def _publish(self):
        now = self.get_clock().now().to_msg()

        # ── 2D Map ──
        map_msg = OccupancyGrid()
        map_msg.header.stamp = now
        map_msg.header.frame_id = self.frame_id
        map_msg.info.resolution = self.resolution
        map_msg.info.width = self.map_w
        map_msg.info.height = self.map_h
        map_msg.info.origin.position.x = float(self.origin[0])
        map_msg.info.origin.position.y = float(self.origin[1])
        map_msg.info.origin.position.z = 0.0
        map_msg.info.origin.orientation.w = 1.0
        map_msg.data = self.map_occ_data
        self.map_pub.publish(map_msg)

        # ── Inflated Height Costmap ──
        cost_msg = OccupancyGrid()
        cost_msg.header.stamp = now
        cost_msg.header.frame_id = self.frame_id
        cost_msg.info.resolution = self.resolution
        cost_msg.info.width = self.map_w
        cost_msg.info.height = self.map_h
        cost_msg.info.origin.position.x = float(self.origin[0])
        cost_msg.info.origin.position.y = float(self.origin[1])
        cost_msg.info.origin.position.z = 0.0
        cost_msg.info.origin.orientation.w = 1.0
        cost_msg.data = self.costmap_data
        self.cost_pub.publish(cost_msg)

        # ── PointCloud2 히트맵 ──
        pc_msg = PointCloud2()
        pc_msg.header.stamp = now
        pc_msg.header.frame_id = self.frame_id
        pc_msg.height = 1
        pc_msg.width = self._pc2_n_pts
        pc_msg.fields = [
            PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
            PointField(name='rgb', offset=12, datatype=PointField.FLOAT32, count=1),
        ]
        pc_msg.is_bigendian = False
        pc_msg.point_step = 16
        pc_msg.row_step = 16 * self._pc2_n_pts
        pc_msg.data = self._pc2_bytes
        pc_msg.is_dense = True
        self.pc2_pub.publish(pc_msg)


def main(args=None):
    rclpy.init(args=args)
    node = HeightCostmapNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
