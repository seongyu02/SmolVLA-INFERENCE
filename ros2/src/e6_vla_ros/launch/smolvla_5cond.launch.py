"""
SmolVLA 5-condition(Orange-512) 통합 추론 launch.

5개 조건은 추론 코드가 동일하고 **체크포인트(model_path)와 프롬프트(prompt_text)만**
다르다. 그래서 launch 1개로 통합하고 cond:=1~5 로 선택한다.

  cond 1 = exp1_expert_only          (512, vision LoRA 없음)        directional prompt
  cond 2 = exp2_vision_lora_0_10     (512, vision 0~10)             directional prompt
  cond 3 = exp3_vision_lora_6_10     (512, vision 6~10)             directional prompt
  cond 4 = exp4_nolr_vision_lora_0_10(512-nolr, vision 0~10)        non-directional prompt
  cond 5 = exp5_nolr_vision_lora_6_10(512-nolr, vision 6~10)        non-directional prompt

프롬프트는 prompt_text 로 직접 넣는다(과거의 prompt_mode/single/per_frame 류 전부 제거).
prompt_text 를 안 주면(=auto) cond에 맞는 학습 프롬프트가 자동 적용된다:
  cond 1~3 → "pick up the orange box from the left side and place it on the right side"
  cond 4~5 → "pick up the orange box and place it on the other side"   (방향 제거 단일 프롬프트)

사용 예)
  ros2 launch e6_vla_ros smolvla_5cond.launch.py cond:=3
  ros2 launch e6_vla_ros smolvla_5cond.launch.py cond:=1 \
      prompt_text:="pick up the orange box from the right side and place it on the left side"
  ros2 launch e6_vla_ros smolvla_5cond.launch.py \
      model_path:=/media/billye6/.../expX/checkpoints/020000/pretrained_model
"""
import datetime
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, OpaqueFunction
from launch.conditions import IfCondition
from launch.substitutions import Command, LaunchConfiguration, PythonExpression
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue

# 5cond bridge 래퍼 (venv_SmolVLA310 Python + CUDA + ROS2 pypath)
_BRIDGE_5COND_SH = os.path.expanduser(
    "~/SmolVLA/SmolVLA-INFERENCE/scripts/run_bridge_5cond_node.sh"
)

# 5cond 체크포인트 루트 (이 젯슨). cond → pretrained_model 경로.
_CKPT_ROOT = "/media/billye6/새 볼륨1/SmolVLA/smolvla_5cond"
_COND_DIRS = {
    "1": "exp1_expert_only",
    "2": "exp2_vision_lora_0_10",
    "3": "exp3_vision_lora_6_10",
    "4": "exp4_nolr_vision_lora_0_10",
    "5": "exp5_nolr_vision_lora_6_10",
}

# cond별 학습 프롬프트 기본값 (Convert_SmolVLADataset_Orange_512[_nolr].py 와 정확히 일치).
_PROMPT_DIRECTIONAL = "pick up the orange box from the left side and place it on the right side"
_PROMPT_NOLR = "pick up the orange box and place it on the other side"


def _resolve_model_path(cond: str, model_path: str) -> str:
    """model_path가 명시되면 그대로, 아니면(=auto/빈값) cond로 자동 지정."""
    mp = (model_path or "").strip()
    if mp and mp.lower() != "auto":
        return mp
    sub = _COND_DIRS.get(cond)
    if sub is None:
        raise RuntimeError(
            f"cond={cond!r} 은 1~5 가 아니며 model_path도 없습니다. "
            f"cond:=1~5 또는 model_path:= 를 지정하세요."
        )
    return os.path.join(_CKPT_ROOT, sub, "checkpoints", "020000", "pretrained_model")


def _resolve_prompt(cond: str, prompt_text: str) -> str:
    """prompt_text가 명시되면 그대로, 아니면(=auto/빈값) cond로 학습 프롬프트 자동 적용."""
    pt = (prompt_text or "").strip()
    if pt and pt.lower() != "auto":
        return pt
    return _PROMPT_NOLR if cond in ("4", "5") else _PROMPT_DIRECTIONAL


def launch_setup(context, *args, **kwargs):
    def cfg(name):
        return LaunchConfiguration(name).perform(context)

    cond = cfg("cond")
    model_path = _resolve_model_path(cond, cfg("model_path"))
    prompt_text = _resolve_prompt(cond, cfg("prompt_text"))
    direct512 = cfg("cam_mode").strip().lower() == "direct512"

    print(f"[5cond] cond={cond}  model_path={model_path}")
    print(f"[5cond] prompt_text={prompt_text!r}")
    print(f"[5cond] cam_mode={cfg('cam_mode')} (direct512={direct512})")

    urdf_file = os.path.join(
        get_package_share_directory("e6_description"), "urdf", "me6_robot.xacro"
    )
    robot_description = ParameterValue(Command(["xacro ", urdf_file]), value_type=str)

    nodes = [
        # camera_state_node: pub_5cond_images=True 로 512 LANCZOS 토픽 발행
        #   /e6/smolvla5/image_hik_512(OBS_IMAGE_1), /e6/smolvla5/image_zed_512(OBS_IMAGE_2)
        Node(
            package="e6_vla_ros",
            executable="camera_state_node",
            name="camera_state_node",
            output="screen",
            parameters=[{
                "robot_ip": LaunchConfiguration("robot_ip"),
                "dry_run": LaunchConfiguration("dry_run"),
                "no_camera": LaunchConfiguration("no_camera"),
                "pub_5cond_images": True,
                "no_crop": False,  # 224 crop → LANCZOS 512 (학습 변환과 일치). 끄지 말 것.
                "cond5_direct512": direct512,  # cam_mode:=direct512 시 학습 크롭 없이 raw→512
            }],
        ),

        # SmolVLA 5cond 직접 추론 브릿지 (venv_SmolVLA310 + CUDA, 래퍼로 실행)
        ExecuteProcess(
            cmd=[
                "bash",
                _BRIDGE_5COND_SH,
                "--ros-args",
                "-r", "__node:=smolvla_bridge_5cond_node",
                "-p", f"model_path:={model_path}",
                "-p", ["base_model_path:=", LaunchConfiguration("base_model_path")],
                "-p", ["infer_hz:=", LaunchConfiguration("infer_hz")],
                "-p", ["init_pose_tol_deg:=", LaunchConfiguration("home_tol_deg")],
                "-p", "wait_for_init_pose:=true",
                "-p", ["save_image_dir:=", LaunchConfiguration("save_image_dir")],
                "-p", ["profile_resource:=", LaunchConfiguration("profile")],
                "-p", ["profile_runs:=", LaunchConfiguration("profile_runs")],
                "-p", ["profile_warmup:=", LaunchConfiguration("profile_warmup")],
                "-p", ["profile_out:=", LaunchConfiguration("profile_out")],
            ],
            output="screen",
        ),

        Node(
            package="e6_vla_ros",
            executable="executor_supervisor_node",
            name="executor_supervisor_node",
            output="screen",
            parameters=[{
                "robot_ip": LaunchConfiguration("robot_ip"),
                "dry_run": LaunchConfiguration("dry_run"),
                "no_camera": LaunchConfiguration("no_camera"),
                "max_delta_deg": LaunchConfiguration("max_delta_deg"),
                "min_tool_z": LaunchConfiguration("min_tool_z"),
                "steps_per_inference": LaunchConfiguration("steps_per_inference"),
                "executor_hz": LaunchConfiguration("executor_hz"),
                "movj_velocity": LaunchConfiguration("movj_velocity"),
                "movj_accel": LaunchConfiguration("movj_accel"),
                "approach_z_done": LaunchConfiguration("approach_z_done"),
                "lift_z_done": LaunchConfiguration("lift_z_done"),
                "stage_done_steps": LaunchConfiguration("stage_done_steps"),
                # 5cond action 계약: joint=DELTA, gripper=ABSOLUTE (변환 코드와 일치)
                "action_mode": LaunchConfiguration("action_mode"),
                "gripper_mode": LaunchConfiguration("gripper_mode"),
                "action_scale": LaunchConfiguration("action_scale"),
                "control_mode": LaunchConfiguration("control_mode"),
                "servoj_t": LaunchConfiguration("servoj_t"),
                "servoj_aheadtime": LaunchConfiguration("servoj_aheadtime"),
                "servoj_gain": LaunchConfiguration("servoj_gain"),
                "vacuum_check_enabled": LaunchConfiguration("vacuum_check_enabled"),
                "vacuum_check_z": LaunchConfiguration("vacuum_check_z"),
                "vacuum_timeout_sec": LaunchConfiguration("vacuum_timeout_sec"),
                "place_force_release_enabled": LaunchConfiguration("place_force_release_enabled"),
                "place_z_threshold": LaunchConfiguration("place_z_threshold"),
                "grip_enable_z": LaunchConfiguration("grip_enable_z"),
                "scripted_lift_enabled": LaunchConfiguration("scripted_lift_enabled"),
                "scripted_lift_target_z": LaunchConfiguration("scripted_lift_target_z"),
                "scripted_lift_wait_frames": LaunchConfiguration("scripted_lift_wait_frames"),
                "scripted_lift_stall_z": LaunchConfiguration("scripted_lift_stall_z"),
                "scripted_lift_dz_thresh": LaunchConfiguration("scripted_lift_dz_thresh"),
                "scripted_return_enabled": LaunchConfiguration("scripted_return_enabled"),
                "scripted_return_target_z": LaunchConfiguration("scripted_return_target_z"),
                "scripted_return_wait_frames": LaunchConfiguration("scripted_return_wait_frames"),
                "scripted_return_stall_z": LaunchConfiguration("scripted_return_stall_z"),
                "max_steps": LaunchConfiguration("max_steps"),
                "min_steps": LaunchConfiguration("min_steps"),
                "home_tol_deg": LaunchConfiguration("home_tol_deg"),
                "home_consec_req": LaunchConfiguration("home_consec_req"),
            }],
        ),

        # task_node: prompt_mode=single 고정(사용자에겐 숨김), prompt_text만 발행.
        #   단일 프롬프트로 학습했으므로 phase/per_frame 류는 쓰지 않는다.
        Node(
            package="e6_vla_ros",
            executable="task_node",
            name="task_node",
            output="screen",
            parameters=[{
                "prompt_mode": "single",
                "prompt_text": prompt_text,
                "grasp_z_max": LaunchConfiguration("grasp_z_max"),
                "min_hold_frames": LaunchConfiguration("min_hold_frames"),
                "pick_prearm_z": LaunchConfiguration("pick_prearm_z"),
            }],
        ),

        Node(
            package="e6_vla_ros",
            executable="e6_visualization_node",
            name="e6_visualization_node",
            output="screen",
        ),

        Node(
            package="robot_state_publisher",
            executable="robot_state_publisher",
            name="robot_state_publisher",
            parameters=[{"robot_description": robot_description}],
        ),

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

        ExecuteProcess(
            cmd=[
                "ros2", "bag", "record",
                "--storage", "mcap",
                "--output", PythonExpression([
                    "'", LaunchConfiguration("mcap_output_dir"), "/' + '",
                    LaunchConfiguration("mcap_session_id"), "'",
                ]),
                "/e6/camera/image",
                "/e6/camera/zed_image",
                "/e6/smolvla5/image_hik_512",
                "/e6/smolvla5/image_zed_512",
                "/e6/robot/state",
                "/e6/robot/tcp",
                "/e6/task/prompt",
                "/e6/policy/action_chunk",
                "/e6/task/status",
            ],
            output="screen",
            condition=IfCondition(LaunchConfiguration("record_mcap")),
        ),
    ]
    return nodes


def generate_launch_description():
    return LaunchDescription([
        # ── 핵심 선택 인자 ──
        DeclareLaunchArgument("cond", default_value="3",
                              description="5cond 조건 1~5 (model_path 미지정 시 자동 경로)"),
        DeclareLaunchArgument("model_path", default_value="auto",
                              description="auto면 cond로 자동, 직접 주면 그 체크포인트 사용"),
        DeclareLaunchArgument("prompt_text", default_value="auto",
                              description="auto면 cond별 학습 프롬프트, 직접 주면 그 문장 사용"),
        DeclareLaunchArgument(
            "base_model_path",
            default_value="/media/billye6/새 볼륨1/SmolVLA/SmolVLA_base_dobot7d_local/pretrained_model",
            description="PEFT adapter의 base (이 젯슨 로컬 경로)",
        ),

        # ── 공통 ──
        DeclareLaunchArgument("robot_ip", default_value="192.168.5.1"),
        DeclareLaunchArgument("infer_hz", default_value="1.25"),
        # 추론 자원(GPU/CPU/RAM) 프로파일링: profile:=true 면 처음 profile_runs 회
        # 추론의 사용률 평균을 JSON+PNG(scripts/infer_resource_live.*) 로 저장.
        DeclareLaunchArgument("profile", default_value="false"),
        DeclareLaunchArgument("profile_runs", default_value="10"),
        DeclareLaunchArgument("profile_warmup", default_value="0"),
        DeclareLaunchArgument("profile_out", default_value="auto"),
        # 카메라 이미지 생성 방식: "crop"(기본, 학습 크롭→512 LANCZOS) | "direct512"(크롭없이 raw→512)
        DeclareLaunchArgument("cam_mode", default_value="crop"),
        # 로그(입력 이미지 PNG + meta.csv) 자동 저장 위치. 기본 = SmolVLA/log 아래.
        # cond별 <exp태그>_<타임스탬프>/ 폴더가 자동 생성됨. "none"이면 저장 안 함.
        DeclareLaunchArgument(
            "save_image_dir",
            default_value="/media/billye6/새 볼륨1/SmolVLA/log",
        ),
        DeclareLaunchArgument("dry_run", default_value="false"),
        DeclareLaunchArgument("no_camera", default_value="false"),

        # ── executor / 모션 ──
        DeclareLaunchArgument("max_delta_deg", default_value="3.0"),
        DeclareLaunchArgument("min_tool_z", default_value="75.0"),
        DeclareLaunchArgument("steps_per_inference", default_value="3"),
        DeclareLaunchArgument("executor_hz", default_value="16.0"),
        DeclareLaunchArgument("approach_z_done", default_value="85.0"),
        DeclareLaunchArgument("lift_z_done", default_value="200.0"),
        DeclareLaunchArgument("stage_done_steps", default_value="3"),
        DeclareLaunchArgument("movj_velocity", default_value="30"),
        DeclareLaunchArgument("movj_accel", default_value="20"),
        DeclareLaunchArgument("action_mode", default_value="delta"),
        DeclareLaunchArgument("gripper_mode", default_value="absolute"),
        DeclareLaunchArgument("action_scale", default_value="2.0"),
        DeclareLaunchArgument("control_mode", default_value="movj"),
        DeclareLaunchArgument("servoj_t", default_value="-1.0"),
        DeclareLaunchArgument("servoj_aheadtime", default_value="-1.0"),
        DeclareLaunchArgument("servoj_gain", default_value="-1.0"),
        DeclareLaunchArgument("max_steps", default_value="3000"),
        DeclareLaunchArgument("min_steps", default_value="100"),
        DeclareLaunchArgument("home_tol_deg", default_value="15.0"),
        DeclareLaunchArgument("home_consec_req", default_value="16"),
        DeclareLaunchArgument("grasp_z_max", default_value="130.0"),
        DeclareLaunchArgument("min_hold_frames", default_value="16"),
        DeclareLaunchArgument("pick_prearm_z", default_value="159.0"),
        DeclareLaunchArgument("vacuum_check_enabled", default_value="false"),
        DeclareLaunchArgument("vacuum_check_z", default_value="85.0"),
        DeclareLaunchArgument("vacuum_timeout_sec", default_value="1.0"),
        DeclareLaunchArgument("place_force_release_enabled", default_value="true"),
        DeclareLaunchArgument("place_z_threshold", default_value="110.0"),
        DeclareLaunchArgument("grip_enable_z", default_value="125.0"),
        DeclareLaunchArgument("scripted_lift_enabled", default_value="true"),
        DeclareLaunchArgument("scripted_lift_target_z", default_value="185.0"),
        DeclareLaunchArgument("scripted_lift_wait_frames", default_value="48"),
        DeclareLaunchArgument("scripted_lift_stall_z", default_value="160.0"),
        DeclareLaunchArgument("scripted_lift_dz_thresh", default_value="0.3"),
        DeclareLaunchArgument("scripted_return_enabled", default_value="true"),
        DeclareLaunchArgument("scripted_return_target_z", default_value="200.0"),
        DeclareLaunchArgument("scripted_return_wait_frames", default_value="16"),
        DeclareLaunchArgument("scripted_return_stall_z", default_value="150.0"),

        # ── 로깅/시각화 ──
        DeclareLaunchArgument("foxglove", default_value="true"),
        DeclareLaunchArgument("foxglove_port", default_value="8765"),
        DeclareLaunchArgument("record_mcap", default_value="false"),
        DeclareLaunchArgument(
            "mcap_output_dir",
            default_value="/media/billye6/새 볼륨1/Dobot/smolvla_mcap",
        ),
        DeclareLaunchArgument(
            "mcap_session_id",
            default_value=datetime.datetime.now().strftime("%Y%m%d_%H%M%S"),
        ),

        OpaqueFunction(function=launch_setup),
    ])
