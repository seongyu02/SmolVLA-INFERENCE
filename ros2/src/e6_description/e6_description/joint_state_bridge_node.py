#!/usr/bin/env python3
import math
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray
from sensor_msgs.msg import JointState

JOINT_NAMES = ['joint1', 'joint2', 'joint3', 'joint4', 'joint5', 'joint6']


class JointStateBridgeNode(Node):
    def __init__(self):
        super().__init__('joint_state_bridge')
        self._pub = self.create_publisher(JointState, '/joint_states', 10)
        self.create_subscription(Float32MultiArray, '/e6/robot/state', self._cb, 10)
        self.get_logger().info('joint_state_bridge ready: /e6/robot/state -> /joint_states')

    def _cb(self, msg: Float32MultiArray):
        if len(msg.data) < 6:
            return
        js = JointState()
        js.header.stamp = self.get_clock().now().to_msg()
        js.name = JOINT_NAMES
        js.position = [math.radians(float(d)) for d in msg.data[:6]]
        self._pub.publish(js)


def main():
    rclpy.init()
    node = JointStateBridgeNode()
    rclpy.spin(node)
    rclpy.shutdown()


if __name__ == '__main__':
    main()
