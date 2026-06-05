import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    xacro_file = os.path.join(
        get_package_share_directory('e6_description'), 'urdf', 'me6_robot.xacro'
    )
    with open(xacro_file) as f:
        robot_description = f.read()

    return LaunchDescription([
        DeclareLaunchArgument('enable_foxglove', default_value='true'),
        DeclareLaunchArgument('foxglove_port',   default_value='8765'),

        Node(
            package='robot_state_publisher',
            executable='robot_state_publisher',
            name='robot_state_publisher',
            parameters=[{'robot_description': robot_description}],
        ),
        Node(
            package='e6_description',
            executable='joint_state_bridge_node.py',
            name='joint_state_bridge',
        ),
        Node(
            package='foxglove_bridge',
            executable='foxglove_bridge',
            name='foxglove_bridge',
            output='screen',
            parameters=[{
                'port': LaunchConfiguration('foxglove_port'),
                'address': '0.0.0.0',
                'tls': False,
                'topic_whitelist': ['.*'],
                'send_buffer_limit': 10000000,
                'asset_uri_allowlist': ['^package://(?!\\.)[^./][^/]*/.*'],
            }],
            condition=IfCondition(LaunchConfiguration('enable_foxglove')),
        ),
    ])
