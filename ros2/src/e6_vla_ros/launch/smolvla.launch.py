import datetime
import os
from ament_index_python.packages import get_package_share_directory

from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument, ExecuteProcess
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration, PythonExpression, Command
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    _urdf_file = os.path.join(
        get_package_share_directory('e6_description'), 'urdf', 'me6_robot.xacro'
    )
    _robot_description = ParameterValue(Command(['xacro ', _urdf_file]), value_type=str)

    return LaunchDescription([
        # ── 인자 선언 (e6_vla.launch.py와 동일) ──────────────────────────────
        DeclareLaunchArgument("robot_ip",          default_value="192.168.5.1"),
        DeclareLaunchArgument("server_host",       default_value="127.0.0.1"),
        DeclareLaunchArgument("server_port",       default_value="8000"),
        DeclareLaunchArgument("task_sequence",     default_value="pick_from_left"),
        DeclareLaunchArgument("stage_timeout_sec", default_value="0.0"),
        DeclareLaunchArgument("loop_sequence",     default_value="false"),
        DeclareLaunchArgument("dry_run",           default_value="false"),
        DeclareLaunchArgument("no_camera",         default_value="false"),
        DeclareLaunchArgument("max_delta_deg",     default_value="3.0"),
        DeclareLaunchArgument("min_tool_z",        default_value="75.0"),
        DeclareLaunchArgument("steps_per_inference", default_value="1"),   # SmolVLA: 1스텝씩 재추론
        DeclareLaunchArgument("executor_hz",       default_value="10.0"),   # SmolVLA 기본 10Hz
        DeclareLaunchArgument("approach_z_done",   default_value="85.0"),
        DeclareLaunchArgument("lift_z_done",       default_value="200.0"),
        DeclareLaunchArgument("stage_done_steps",  default_value="3"),
        DeclareLaunchArgument("save_debug_images", default_value="false"),
        DeclareLaunchArgument("movj_velocity",     default_value="30"),     # SmolVLA 기본 30
        DeclareLaunchArgument("movj_accel",        default_value="20"),     # SmolVLA 기본 20
        DeclareLaunchArgument("record_mcap",       default_value="false"),
        DeclareLaunchArgument("mcap_output_dir",   default_value="/media/billye6/새 볼륨/Dobot/smolvla_mcap"),
        DeclareLaunchArgument("mcap_session_id",   default_value=datetime.datetime.now().strftime("%Y%m%d_%H%M%S")),
        DeclareLaunchArgument("foxglove",          default_value="true"),
        DeclareLaunchArgument("foxglove_port",     default_value="8765"),
        DeclareLaunchArgument("action_mode",       default_value="delta"),
        DeclareLaunchArgument("gripper_mode",      default_value="absolute"),
        DeclareLaunchArgument("action_scale",      default_value="1.0"),
        DeclareLaunchArgument("control_mode",      default_value="movj"),
        DeclareLaunchArgument("servoj_t",          default_value="-1.0"),
        DeclareLaunchArgument("servoj_aheadtime",  default_value="-1.0"),
        DeclareLaunchArgument("servoj_gain",       default_value="-1.0"),
        DeclareLaunchArgument("prompt_mode",       default_value="single"),   # SmolVLA 기본 single
        DeclareLaunchArgument("source_side",       default_value="left"),
        DeclareLaunchArgument("target_side",       default_value="right"),
        DeclareLaunchArgument("prompt_variant",    default_value="-1"),
        DeclareLaunchArgument("prompt_text",       default_value="pick up the orange box from the left side and place it on the right side"),
        DeclareLaunchArgument("prompt_dataset",    default_value="v13"),
        DeclareLaunchArgument("max_steps",         default_value="3000"),
        DeclareLaunchArgument("min_steps",         default_value="100"),
        DeclareLaunchArgument("home_tol_deg",      default_value="5.0"),
        DeclareLaunchArgument("home_consec_req",   default_value="16"),
        DeclareLaunchArgument("grasp_z_max",             default_value="130.0"),
        DeclareLaunchArgument("min_hold_frames",         default_value="16"),
        DeclareLaunchArgument("vacuum_check_enabled",    default_value="false"),
        DeclareLaunchArgument("vacuum_check_z",          default_value="85.0"),
        DeclareLaunchArgument("vacuum_timeout_sec",      default_value="1.0"),
        DeclareLaunchArgument("place_force_release_enabled", default_value="true"),
        DeclareLaunchArgument("place_z_threshold",       default_value="110.0"),
        DeclareLaunchArgument("grip_enable_z",           default_value="125.0"),
        DeclareLaunchArgument("pick_prearm_z",           default_value="159.0"),
        DeclareLaunchArgument("scripted_lift_enabled",   default_value="true"),
        DeclareLaunchArgument("scripted_lift_target_z",  default_value="185.0"),
        DeclareLaunchArgument("scripted_lift_wait_frames", default_value="48"),
        DeclareLaunchArgument("scripted_lift_stall_z",   default_value="160.0"),
        DeclareLaunchArgument("scripted_lift_dz_thresh", default_value="0.3"),
        DeclareLaunchArgument("use_voice",         default_value="false"),
        DeclareLaunchArgument("use_mic",           default_value="true"),
        DeclareLaunchArgument("voice_model_size",  default_value="base"),
        DeclareLaunchArgument("voice_language",    default_value="ko"),
        DeclareLaunchArgument("voice_device_index", default_value="-1"),
        DeclareLaunchArgument("voice_vad_amplitude", default_value="0.02"),
        DeclareLaunchArgument("voice_silence_sec", default_value="1.5"),

        # ── 노드 1: camera_state_node (512px 이미지 추가 발행) ────────────────
        Node(
            package="e6_vla_ros",
            executable="camera_state_node",
            name="camera_state_node",
            output="screen",
            parameters=[{
                "robot_ip":           LaunchConfiguration("robot_ip"),
                "dry_run":            LaunchConfiguration("dry_run"),
                "no_camera":          LaunchConfiguration("no_camera"),
                "pub_smolvla_images": True,   # 512×512 이미지 발행 활성화
            }],
        ),

        # ── 노드 2: smolvla_bridge_node (HTTP POST /act) ──────────────────────
        Node(
            package="e6_vla_ros",
            executable="smolvla_bridge_node",
            name="smolvla_bridge_node",
            output="screen",
            parameters=[{
                "server_host":       LaunchConfiguration("server_host"),
                "server_port":       LaunchConfiguration("server_port"),
                "save_debug_images": LaunchConfiguration("save_debug_images"),
            }],
        ),

        # ── 노드 3: executor_supervisor_node (변경 없음) ──────────────────────
        Node(
            package="e6_vla_ros",
            executable="executor_supervisor_node",
            name="executor_supervisor_node",
            output="screen",
            parameters=[{
                "robot_ip":      LaunchConfiguration("robot_ip"),
                "dry_run":       LaunchConfiguration("dry_run"),
                "no_camera":     LaunchConfiguration("no_camera"),
                "max_delta_deg":       LaunchConfiguration("max_delta_deg"),
                "min_tool_z":          LaunchConfiguration("min_tool_z"),
                "steps_per_inference": LaunchConfiguration("steps_per_inference"),
                "executor_hz":         LaunchConfiguration("executor_hz"),
                "movj_velocity":       LaunchConfiguration("movj_velocity"),
                "movj_accel":          LaunchConfiguration("movj_accel"),
                "approach_z_done":     LaunchConfiguration("approach_z_done"),
                "lift_z_done":         LaunchConfiguration("lift_z_done"),
                "stage_done_steps":    LaunchConfiguration("stage_done_steps"),
                "action_mode":         LaunchConfiguration("action_mode"),
                "gripper_mode":        LaunchConfiguration("gripper_mode"),
                "action_scale":        LaunchConfiguration("action_scale"),
                "control_mode":        LaunchConfiguration("control_mode"),
                "servoj_t":            LaunchConfiguration("servoj_t"),
                "servoj_aheadtime":    LaunchConfiguration("servoj_aheadtime"),
                "servoj_gain":         LaunchConfiguration("servoj_gain"),
                "vacuum_check_enabled":        LaunchConfiguration("vacuum_check_enabled"),
                "vacuum_check_z":              LaunchConfiguration("vacuum_check_z"),
                "vacuum_timeout_sec":          LaunchConfiguration("vacuum_timeout_sec"),
                "place_force_release_enabled": LaunchConfiguration("place_force_release_enabled"),
                "place_z_threshold":           LaunchConfiguration("place_z_threshold"),
                "grip_enable_z":               LaunchConfiguration("grip_enable_z"),
                "scripted_lift_enabled":       LaunchConfiguration("scripted_lift_enabled"),
                "scripted_lift_target_z":      LaunchConfiguration("scripted_lift_target_z"),
                "scripted_lift_wait_frames":   LaunchConfiguration("scripted_lift_wait_frames"),
                "scripted_lift_stall_z":       LaunchConfiguration("scripted_lift_stall_z"),
                "scripted_lift_dz_thresh":     LaunchConfiguration("scripted_lift_dz_thresh"),
                "max_steps":                   LaunchConfiguration("max_steps"),
                "min_steps":                   LaunchConfiguration("min_steps"),
                "home_tol_deg":                LaunchConfiguration("home_tol_deg"),
                "home_consec_req":             LaunchConfiguration("home_consec_req"),
            }],
        ),

        # ── 노드 4: task_node (변경 없음) ─────────────────────────────────────
        Node(
            package="e6_vla_ros",
            executable="task_node",
            name="task_node",
            output="screen",
            parameters=[{
                "task_sequence":     LaunchConfiguration("task_sequence"),
                "stage_timeout_sec": LaunchConfiguration("stage_timeout_sec"),
                "loop_sequence":     LaunchConfiguration("loop_sequence"),
                "prompt_mode":       LaunchConfiguration("prompt_mode"),
                "source_side":       LaunchConfiguration("source_side"),
                "target_side":       LaunchConfiguration("target_side"),
                "prompt_variant":    LaunchConfiguration("prompt_variant"),
                "prompt_text":       LaunchConfiguration("prompt_text"),
                "prompt_dataset":    LaunchConfiguration("prompt_dataset"),
                "grasp_z_max":       LaunchConfiguration("grasp_z_max"),
                "min_hold_frames":   LaunchConfiguration("min_hold_frames"),
                "pick_prearm_z":     LaunchConfiguration("pick_prearm_z"),
            }],
        ),

        # ── 노드 5: e6_visualization_node ────────────────────────────────────
        Node(
            package='e6_vla_ros',
            executable='e6_visualization_node',
            name='e6_visualization_node',
            output='screen',
        ),

        # ── 노드 6: robot_state_publisher ────────────────────────────────────
        Node(
            package='robot_state_publisher',
            executable='robot_state_publisher',
            name='robot_state_publisher',
            parameters=[{'robot_description': _robot_description}],
        ),

        # ── 노드 7: joint_state_bridge ───────────────────────────────────────
        Node(
            package='e6_description',
            executable='joint_state_bridge_node.py',
            name='joint_state_bridge',
        ),

        # ── 노드 8: foxglove_bridge ───────────────────────────────────────────
        Node(
            package="foxglove_bridge",
            executable="foxglove_bridge",
            name="foxglove_bridge",
            output="screen",
            parameters=[{
                "port": LaunchConfiguration("foxglove_port"),
                "address": "0.0.0.0",
                "tls": False,
                "topic_whitelist": [".*"],
                "send_buffer_limit": 10000000,
                "asset_uri_allowlist": ["^package://(?!\\.)[^./][^/]*/.*"],
            }],
            condition=IfCondition(LaunchConfiguration("foxglove")),
        ),

        # ── 노드 9: voice_command_node ────────────────────────────────────────
        Node(
            package="e6_vla_ros",
            executable="voice_command_node",
            name="voice_command_node",
            output="screen",
            parameters=[{
                "use_mic":              LaunchConfiguration("use_mic"),
                "model_size":           LaunchConfiguration("voice_model_size"),
                "language":             LaunchConfiguration("voice_language"),
                "device_index":         LaunchConfiguration("voice_device_index"),
                "vad_min_amplitude":    LaunchConfiguration("voice_vad_amplitude"),
                "silence_duration_sec": LaunchConfiguration("voice_silence_sec"),
            }],
            condition=IfCondition(LaunchConfiguration("use_voice")),
        ),

        # ── MCAP 레코더 ───────────────────────────────────────────────────────
        ExecuteProcess(
            cmd=[
                "ros2", "bag", "record",
                "--storage", "mcap",
                "--output", PythonExpression([
                    "'", LaunchConfiguration("mcap_output_dir"), "/' + '",
                    LaunchConfiguration("mcap_session_id"), "'"
                ]),
                "/e6/camera/image_512",
                "/e6/camera/zed_image_512",
                "/e6/robot/state",
                "/e6/robot/tcp",
                "/e6/task/prompt",
                "/e6/policy/action_chunk",
                "/e6/task/status",
            ],
            output="screen",
            condition=IfCondition(LaunchConfiguration("record_mcap")),
        ),
    ])
