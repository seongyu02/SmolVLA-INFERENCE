#!/usr/bin/env python3
import math
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray, Float32, String
from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped, Point
from visualization_msgs.msg import Marker, MarkerArray


class E6VisualizationNode(Node):
    def __init__(self):
        super().__init__('e6_visualization_node')

        self._frame_id = 'world'
        self._tcp_z = None          # mm
        self._state = None          # [j1..j6 deg, gripper]
        self._action_chunk = None   # (16, 7)
        self._prompt = ''
        self._status = ''
        self._gripper = 0.0

        # TCP 실제 이동 궤적 (최대 500포인트)
        self._tcp_path_msg = Path()
        self._tcp_path_msg.header.frame_id = self._frame_id
        self._max_path_len = 500

        # Subscribers
        self.create_subscription(Float32MultiArray, '/e6/robot/state',         self._cb_state,   10)
        self.create_subscription(Float32,           '/e6/robot/tcp_z',         self._cb_tcpz,    10)
        self.create_subscription(Float32MultiArray, '/e6/policy/action_chunk', self._cb_chunk,   10)
        self.create_subscription(String,            '/e6/task/prompt',         self._cb_prompt,  10)
        self.create_subscription(String,            '/e6/supervisor/status',   self._cb_status,  10)
        self.create_subscription(Float32,           '/e6/gripper/commanded',   self._cb_gripper, 10)

        # Publishers
        self._pub_tcp_path    = self.create_publisher(Path,        '/e6/tcp_actual_path',   10)
        self._pub_pred_path   = self.create_publisher(MarkerArray, '/e6/policy_pred_path',  10)
        self._pub_text        = self.create_publisher(MarkerArray, '/e6/viz/phase_text',    10)
        self._pub_tcp_marker  = self.create_publisher(MarkerArray, '/e6/viz/tcp_marker',    10)

        # 10Hz publish timer
        self.create_timer(0.1, self._publish_all)
        self.get_logger().info('e6_visualization_node started')

    # ── callbacks ──────────────────────────────────────────────────────────

    def _cb_state(self, msg: Float32MultiArray):
        if len(msg.data) >= 7:
            self._state = list(msg.data)
            self._gripper = float(msg.data[6])

    def _cb_tcpz(self, msg: Float32):
        self._tcp_z = float(msg.data)

    def _cb_chunk(self, msg: Float32MultiArray):
        n = len(msg.data)
        if n >= 7:
            steps = n // 7
            self._action_chunk = [[float(msg.data[i * 7 + j]) for j in range(7)]
                                   for i in range(steps)]

    def _cb_prompt(self, msg: String):
        self._prompt = msg.data

    def _cb_status(self, msg: String):
        self._status = msg.data

    def _cb_gripper(self, msg: Float32):
        self._gripper = float(msg.data)

    # ── publish ────────────────────────────────────────────────────────────

    def _publish_all(self):
        now = self.get_clock().now().to_msg()
        self._publish_tcp_path(now)
        self._publish_pred_path(now)
        self._publish_text_markers(now)
        self._publish_tcp_sphere(now)

    def _publish_tcp_path(self, now):
        """TCP 실제 이동 궤적 — tcp_z 기반, xy는 joint 기반 근사"""
        if self._tcp_z is None or self._state is None:
            return

        # 간단한 TCP 위치 근사 (joint angles → 상대적 x, y 근사)
        # 정밀 FK 대신 j1 회전 + tcp_z 조합으로 2D 평면 표현
        j1_rad = math.radians(self._state[0])
        r = 0.3  # 암 reach 근사 (m) — 실제 값 아님, 시각화용
        tcp_x = r * math.cos(j1_rad)
        tcp_y = r * math.sin(j1_rad)
        tcp_z_m = self._tcp_z / 1000.0  # mm → m

        pose = PoseStamped()
        pose.header.stamp = now
        pose.header.frame_id = self._frame_id
        pose.pose.position.x = tcp_x
        pose.pose.position.y = tcp_y
        pose.pose.position.z = tcp_z_m
        pose.pose.orientation.w = 1.0

        self._tcp_path_msg.header.stamp = now
        self._tcp_path_msg.poses.append(pose)
        if len(self._tcp_path_msg.poses) > self._max_path_len:
            self._tcp_path_msg.poses.pop(0)

        self._pub_tcp_path.publish(self._tcp_path_msg)

    def _publish_pred_path(self, now):
        """Policy action chunk 예측 궤적 — delta 누산으로 LINE_STRIP"""
        if self._action_chunk is None or self._state is None or self._tcp_z is None:
            return

        ma = MarkerArray()

        # 현재 TCP 위치
        j1_rad = math.radians(self._state[0])
        r = 0.3
        cx = r * math.cos(j1_rad)
        cy = r * math.sin(j1_rad)
        cz = self._tcp_z / 1000.0

        line = Marker()
        line.header.frame_id = self._frame_id
        line.header.stamp = now
        line.ns = 'policy_pred'
        line.id = 0
        line.type = Marker.LINE_STRIP
        line.action = Marker.ADD
        line.scale.x = 0.005
        line.color.r = 0.2
        line.color.g = 0.8
        line.color.b = 0.2
        line.color.a = 0.9
        line.pose.orientation.w = 1.0

        # 현재 위치부터 시작
        p = Point()
        p.x, p.y, p.z = cx, cy, cz
        line.points.append(p)

        # delta 누산 (j1 delta로 xy 근사, j3 delta로 z 근사)
        ax, ay, az = cx, cy, cz
        j1_cur = self._state[0]
        j3_cur = self._state[2]

        for step in self._action_chunk[:8]:  # 앞 8스텝만
            dj1 = step[0]
            dj3 = step[2]
            j1_cur += dj1
            j3_cur += dj3
            j1r = math.radians(j1_cur)
            ax = r * math.cos(j1r)
            ay = r * math.sin(j1r)
            az -= dj3 * 0.003  # j3 delta → z 근사 (스케일 임의)
            pt = Point()
            pt.x, pt.y, pt.z = ax, ay, az
            line.points.append(pt)

        ma.markers.append(line)

        # 예측 끝점 sphere
        end_sphere = Marker()
        end_sphere.header.frame_id = self._frame_id
        end_sphere.header.stamp = now
        end_sphere.ns = 'policy_pred_end'
        end_sphere.id = 1
        end_sphere.type = Marker.SPHERE
        end_sphere.action = Marker.ADD
        end_sphere.pose.position.x = ax
        end_sphere.pose.position.y = ay
        end_sphere.pose.position.z = az
        end_sphere.pose.orientation.w = 1.0
        end_sphere.scale.x = end_sphere.scale.y = end_sphere.scale.z = 0.02
        end_sphere.color.r = 0.2
        end_sphere.color.g = 1.0
        end_sphere.color.b = 0.2
        end_sphere.color.a = 0.9
        ma.markers.append(end_sphere)

        self._pub_pred_path.publish(ma)

    def _publish_text_markers(self, now):
        """현재 phase prompt + status 텍스트"""
        if self._tcp_z is None:
            return

        ma = MarkerArray()
        j1_rad = math.radians(self._state[0]) if self._state else 0.0
        r = 0.35
        tx = r * math.cos(j1_rad)
        ty = r * math.sin(j1_rad)
        tz = self._tcp_z / 1000.0 + 0.08

        text = Marker()
        text.header.frame_id = self._frame_id
        text.header.stamp = now
        text.ns = 'phase_text'
        text.id = 0
        text.type = Marker.TEXT_VIEW_FACING
        text.action = Marker.ADD
        text.pose.position.x = tx
        text.pose.position.y = ty
        text.pose.position.z = tz
        text.pose.orientation.w = 1.0
        text.scale.z = 0.04
        text.color.r = 1.0
        text.color.g = 1.0
        text.color.b = 0.0
        text.color.a = 1.0
        # prompt를 / 기준으로 마지막 단어만 표시 (phase name)
        short = self._prompt.split()[-1] if self._prompt else ''
        grip_str = 'GRIP' if self._gripper > 0.5 else 'open'
        text.text = f'{short}\nz={self._tcp_z:.0f}mm {grip_str}'
        ma.markers.append(text)

        # status text (아래)
        status_text = Marker()
        status_text.header.frame_id = self._frame_id
        status_text.header.stamp = now
        status_text.ns = 'status_text'
        status_text.id = 1
        status_text.type = Marker.TEXT_VIEW_FACING
        status_text.action = Marker.ADD
        status_text.pose.position.x = tx
        status_text.pose.position.y = ty
        status_text.pose.position.z = tz - 0.06
        status_text.pose.orientation.w = 1.0
        status_text.scale.z = 0.03
        status_text.color.r = 0.8
        status_text.color.g = 0.8
        status_text.color.b = 1.0
        status_text.color.a = 1.0
        status_text.text = self._status
        ma.markers.append(status_text)

        self._pub_text.publish(ma)

    def _publish_tcp_sphere(self, now):
        """현재 TCP 위치 sphere marker"""
        if self._tcp_z is None or self._state is None:
            return

        j1_rad = math.radians(self._state[0])
        r = 0.3
        ma = MarkerArray()

        sphere = Marker()
        sphere.header.frame_id = self._frame_id
        sphere.header.stamp = now
        sphere.ns = 'tcp_current'
        sphere.id = 0
        sphere.type = Marker.SPHERE
        sphere.action = Marker.ADD
        sphere.pose.position.x = r * math.cos(j1_rad)
        sphere.pose.position.y = r * math.sin(j1_rad)
        sphere.pose.position.z = self._tcp_z / 1000.0
        sphere.pose.orientation.w = 1.0
        sphere.scale.x = sphere.scale.y = sphere.scale.z = 0.03
        # 흡착 ON이면 빨간색, OFF면 파란색
        sphere.color.r = 1.0 if self._gripper > 0.5 else 0.2
        sphere.color.g = 0.2
        sphere.color.b = 0.2 if self._gripper > 0.5 else 1.0
        sphere.color.a = 1.0
        ma.markers.append(sphere)

        self._pub_tcp_marker.publish(ma)


def main():
    rclpy.init()
    node = E6VisualizationNode()
    rclpy.spin(node)
    rclpy.shutdown()


if __name__ == '__main__':
    main()
