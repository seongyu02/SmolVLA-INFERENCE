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
        # ── 인자 선언 ──────────────────────────────────────────────────────
        DeclareLaunchArgument("robot_ip",       default_value="192.168.5.1"),
        DeclareLaunchArgument("server_host",    default_value="127.0.0.1"),
        DeclareLaunchArgument("server_port",    default_value="8000"),
        DeclareLaunchArgument("task_sequence",  default_value="pick_from_left"),
        DeclareLaunchArgument("stage_timeout_sec", default_value="0.0"),
        DeclareLaunchArgument("loop_sequence",  default_value="false"),
        DeclareLaunchArgument("dry_run",             default_value="false"),
        DeclareLaunchArgument("no_camera",           default_value="false"),
        DeclareLaunchArgument("max_delta_deg",       default_value="3.0"),
        DeclareLaunchArgument("min_tool_z",          default_value="75.0"),
        DeclareLaunchArgument("infer_hz",            default_value="2.0"),
        DeclareLaunchArgument("steps_per_inference", default_value="8"),
        DeclareLaunchArgument("executor_hz",         default_value="16.0"),
        DeclareLaunchArgument("approach_z_done",     default_value="85.0"),
        DeclareLaunchArgument("lift_z_done",         default_value="200.0"),
        DeclareLaunchArgument("stage_done_steps",    default_value="3"),
        DeclareLaunchArgument("save_debug_images",   default_value="false"),
        DeclareLaunchArgument("movj_velocity",       default_value="70"),
        DeclareLaunchArgument("movj_accel",          default_value="60"),
        DeclareLaunchArgument("record_mcap",         default_value="false"),
        DeclareLaunchArgument("mcap_output_dir",     default_value="/media/billye6/새 볼륨/Dobot/inference_mcap"),
        DeclareLaunchArgument("mcap_session_id",     default_value=datetime.datetime.now().strftime("%Y%m%d_%H%M%S")),
        DeclareLaunchArgument("foxglove",            default_value="true"),
        DeclareLaunchArgument("foxglove_port",       default_value="8765"),
        DeclareLaunchArgument("action_mode",         default_value="delta"),     # "absolute" (v6) | "delta" (v8/v13)
        DeclareLaunchArgument("gripper_mode",        default_value="absolute"),  # "delta" (v13/v14 누산) | "absolute" (v16/v17 직접 threshold)
        DeclareLaunchArgument("action_scale",        default_value="1.0"),
        DeclareLaunchArgument("control_mode",        default_value="movj"),      # "movj" | "servoj"
        DeclareLaunchArgument("servoj_t",            default_value="-1.0"),      # ServoJ 목표 도달 시간(s). -1=Dobot 기본값
        DeclareLaunchArgument("servoj_aheadtime",    default_value="-1.0"),      # ServoJ aheadtime [20,100]. -1=기본값(50)
        DeclareLaunchArgument("servoj_gain",         default_value="-1.0"),      # ServoJ gain [200,1000]. -1=기본값(500)
        DeclareLaunchArgument("prompt_mode",         default_value="per_frame_v16"),  # "episode" (v6) | "per_frame" (v8) | "single" (v13) | "per_frame_v16"
        DeclareLaunchArgument("source_side",         default_value="left"),      # v8 per_frame / v13 single / per_frame_v16 모드 필수
        DeclareLaunchArgument("target_side",         default_value="right"),
        DeclareLaunchArgument("prompt_variant",      default_value="-1"),        # v13: 0~2 고정, -1이면 랜덤
        DeclareLaunchArgument("prompt_text",         default_value=""),          # 직접 입력 시 source_side/variant 무시
        DeclareLaunchArgument("prompt_dataset",      default_value="v13"),       # "v13" (6 variant 랜덤) | "v14" (anchor 2개)
        DeclareLaunchArgument("max_steps",           default_value="3000"),      # v13 종료 B: 최대 step
        DeclareLaunchArgument("min_steps",           default_value="100"),       # v13 종료 가드: 시작 후 최소 step
        DeclareLaunchArgument("home_tol_deg",        default_value="5.0"),       # v13 종료 C: j1..j3 허용 오차
        DeclareLaunchArgument("home_consec_req",     default_value="16"),        # v13 종료 C: 연속 만족 프레임 수
        DeclareLaunchArgument("grasp_z_max",             default_value="130.0"),      # grasp 진입 허용 최대 TCP Z (mm)
        DeclareLaunchArgument("min_hold_frames",         default_value="16"),         # phase 최소 유지 프레임 (빠른 사이클 방지)
        DeclareLaunchArgument("vacuum_check_enabled",        default_value="true"),
        DeclareLaunchArgument("vacuum_check_z",              default_value="85.0"),
        DeclareLaunchArgument("vacuum_timeout_sec",          default_value="1.0"),
        DeclareLaunchArgument("place_force_release_enabled", default_value="true"),
        DeclareLaunchArgument("place_z_threshold",           default_value="110.0"),
        DeclareLaunchArgument("grip_enable_z",               default_value="125.0"),  # 0=비활성, >0이면 이 Z 아래에서만 흡착 허용
        DeclareLaunchArgument("pick_prearm_z",               default_value="159.0"),  # 0=비활성, >0이면 Z ≤ 이 값에서 pick_up phase 선진입
        DeclareLaunchArgument("scripted_lift_enabled",       default_value="true"),   # lift stall 시 강제 상승
        DeclareLaunchArgument("scripted_lift_target_z",      default_value="185.0"),  # 강제 상승 목표 Z (mm)
        DeclareLaunchArgument("scripted_lift_wait_frames",   default_value="48"),     # stall 판정 대기 프레임 (3초@16Hz)
        DeclareLaunchArgument("scripted_lift_stall_z",       default_value="160.0"),  # 이 Z 미달 시 stall
        DeclareLaunchArgument("scripted_lift_dz_thresh",     default_value="0.3"),    # mm/frame 이하 시 stall
        DeclareLaunchArgument("scripted_return_enabled",     default_value="true"),   # release stall 시 강제 상승
        DeclareLaunchArgument("scripted_return_target_z",    default_value="200.0"),  # 강제 상승 목표 Z (mm)
        DeclareLaunchArgument("scripted_return_wait_frames", default_value="16"),     # 1초@16Hz 대기
        DeclareLaunchArgument("scripted_return_stall_z",     default_value="150.0"),  # 이 Z 미달 시 stall

        # ── 음성 명령 인자 ─────────────────────────────────────────────────────
        DeclareLaunchArgument("use_voice",              default_value="false"),   # voice_command_node 활성화
        DeclareLaunchArgument("use_mic",                default_value="true"),    # 마이크 캡처 활성화 (false=텍스트 전용)
        DeclareLaunchArgument("voice_model_size",       default_value="base"),    # tiny/base/small/medium
        DeclareLaunchArgument("voice_language",         default_value="ko"),      # STT 언어
        DeclareLaunchArgument("voice_device_index",     default_value="-1"),      # 마이크 장치 인덱스
        DeclareLaunchArgument("voice_vad_amplitude",    default_value="0.02"),    # 음성 감지 최소 진폭
        DeclareLaunchArgument("voice_silence_sec",      default_value="1.5"),     # 발화 종료 판정 침묵 시간

        # ── 노드 1: camera_state_node ──────────────────────────────────────
        Node(
            package="e6_vla_ros",
            executable="camera_state_node",
            name="camera_state_node",
            output="screen",
            parameters=[{
                "robot_ip":  LaunchConfiguration("robot_ip"),
                "dry_run":   LaunchConfiguration("dry_run"),
                "no_camera": LaunchConfiguration("no_camera"),
            }],
        ),

        # ── 노드 2: inference_bridge_node ──────────────────────────────────
        Node(
            package="e6_vla_ros",
            executable="inference_bridge_node",
            name="inference_bridge_node",
            output="screen",
            parameters=[{
                "server_host":       LaunchConfiguration("server_host"),
                "server_port":       LaunchConfiguration("server_port"),
                "infer_hz":          LaunchConfiguration("infer_hz"),
                "save_debug_images": LaunchConfiguration("save_debug_images"),
                "action_mode":       LaunchConfiguration("action_mode"),
            }],
            additional_env={
                "PYTHONPATH": "/home/billye6/E6-VLA_INFERENCE/packages/openpi-client/src"
                              + (":" + os.environ["PYTHONPATH"] if os.environ.get("PYTHONPATH") else ""),
            },
        ),

        # ── 노드 3: executor_supervisor_node ───────────────────────────────
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
                "scripted_return_enabled":     LaunchConfiguration("scripted_return_enabled"),
                "scripted_return_target_z":    LaunchConfiguration("scripted_return_target_z"),
                "scripted_return_wait_frames": LaunchConfiguration("scripted_return_wait_frames"),
                "scripted_return_stall_z":     LaunchConfiguration("scripted_return_stall_z"),
                "max_steps":                   LaunchConfiguration("max_steps"),
                "min_steps":                   LaunchConfiguration("min_steps"),
                "home_tol_deg":                LaunchConfiguration("home_tol_deg"),
                "home_consec_req":             LaunchConfiguration("home_consec_req"),
            }],
        ),

        # ── 노드 4: task_node ──────────────────────────────────────────────
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

        # ── 노드 5-b: e6_visualization_node ──────────────────────────────────
        Node(
            package='e6_vla_ros',
            executable='e6_visualization_node',
            name='e6_visualization_node',
            output='screen',
        ),

        # ── 노드 5: robot_state_publisher ─────────────────────────────────────
        Node(
            package='robot_state_publisher',
            executable='robot_state_publisher',
            name='robot_state_publisher',
            parameters=[{'robot_description': _robot_description}],
        ),

        # ── 노드 6: joint_state_bridge (/e6/robot/state deg → /joint_states rad)
        Node(
            package='e6_description',
            executable='joint_state_bridge_node.py',
            name='joint_state_bridge',
        ),

        # ── 노드 7: foxglove_bridge (foxglove:=true 일 때만 실행) ─────────────
        # 사용: ros2 launch e6_vla_ros e6_vla.launch.py foxglove:=true
        # Foxglove Studio에서 ws://<jetson-ip>:8765 로 연결하면 실시간 시각화
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

        # ── 노드 8: voice_command_node (use_voice:=true 일 때만 실행) ────────
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

        # ── MCAP 레코더 (record_mcap:=true 일 때만 실행) ───────────────────
        # 기록 토픽: 카메라 입력 2개 + 로봇 상태 + 프롬프트 + AI 출력 + 태스크 상태
        # 사용: ros2 launch e6_vla_ros e6_vla.launch.py record_mcap:=true
        # Foxglove Studio에서 .mcap 파일 열면 타임라인·카메라·관절값 동시 재생 가능
        ExecuteProcess(
            cmd=[
                "ros2", "bag", "record",
                "--storage", "mcap",
                "--output", PythonExpression([
                    "'", LaunchConfiguration("mcap_output_dir"), "/' + '",
                    LaunchConfiguration("mcap_session_id"), "'"
                ]),
                "/e6/camera/image",
                "/e6/camera/zed_image",
                "/e6/robot/state",
                "/e6/task/prompt",
                "/e6/policy/action_chunk",
                "/e6/task/status",
            ],
            output="screen",
            condition=IfCondition(LaunchConfiguration("record_mcap")),
        ),
    ])
