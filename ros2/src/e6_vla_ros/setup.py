from setuptools import setup

setup(
    name='e6_vla_ros',
    version='0.1.0',
    packages=['e6_vla_ros'],
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/e6_vla_ros']),
        ('share/e6_vla_ros', ['package.xml']),
        ('share/e6_vla_ros/launch', ['launch/e6_vla.launch.py',
                                     'launch/smolvla.launch.py',
                                     'launch/smolvla_7d_expert.launch.py',
                                     'launch/smolvla_7d_lora.launch.py']),
    ],
    install_requires=['setuptools'],
    entry_points={
        'console_scripts': [
            'camera_state_node        = e6_vla_ros.camera_state_node:main',
            'inference_bridge_node    = e6_vla_ros.inference_bridge_node:main',
            'executor_supervisor_node = e6_vla_ros.executor_supervisor_node:main',
            'task_node                = e6_vla_ros.task_node:main',
            'voice_command_node       = e6_vla_ros.voice_command_node:main',
            'e6_visualization_node    = e6_vla_ros.e6_visualization_node:main',
            'smolvla_bridge_node      = e6_vla_ros.smolvla_bridge_node:main',
            'smolvla_bridge_7d_node   = e6_vla_ros.smolvla_bridge_7d_node:main',
        ],
    },
)
