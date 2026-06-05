#!/usr/bin/env python3
"""
executor_supervisor_node — 액션 실행 + 안전 감시

구독 토픽:
  /e6/policy/action_chunk std_msgs/Float32MultiArray  (16*7 flatten)
  /e6/camera/image        sensor_msgs/Image           (bad_camera 감시용)
  /e6/robot/state         std_msgs/Float32MultiArray  [j1..j6 deg, gripper]
  /e6/robot/tcp_z         std_msgs/Float32

발행 토픽:
  /e6/supervisor/status   std_msgs/String  10Hz
    값: "RUNNING" | "STAGE_DONE:<stage>" | "FAIL_SAFETY:<reason>"

서비스:
  /e6/emergency_stop      std_srvs/Trigger

파라미터:
  robot_ip              (str,   default "192.168.5.1")
  dry_run               (bool,  default False)
  max_delta_deg         (float, default 3.0)
  min_tool_z            (float, default 101.0)   mm
  grip_close_threshold  (float, default 0.5)
  grip_open_threshold   (float, default 0.4)
  grip_close_latch_steps(int,   default 0)
  movj_velocity         (int,   default 70)
  movj_accel            (int,   default 60)
  chunk_staleness_sec   (float, default 5.0)
  steps_per_inference   (int,   default 8)    — 청크에서 실행할 스텝 수 (0=전체)
  executor_hz           (float, default 16.0) — MovJ 전송 주파수 (낮출수록 느려짐)
  approach_z_done       (float, default 85.0) — approach 완료: TCP Z ≤ 이 값 (mm)
  lift_z_done           (float, default 200.0)— pick/place 완료: TCP Z ≥ 이 값 (mm)
  stage_done_steps      (int,   default 3)    — 완료 조건 연속 만족 스텝 수
  bad_camera_consecutive(int,   default 10)
  camera_black_mean     (float, default 8.0)
  max_steps             (int,   default 500)  — 안전망 B: step 초과 시 강제 종료 (31초@16Hz)
  min_steps             (int,   default 100)  — 종료 체크 시작 최소 step 수 (6초 가드)
  home_tol_deg          (float, default 5.0)  — 종료 조건 C: j1..j3 init 오차 허용 범위 (deg)
  home_consec_req       (int,   default 16)   — 종료 조건 C: 연속 만족 프레임 수 (1초@16Hz)
  gripper_mode          (str,   default "delta") — "delta": 누산 ±0.5 (v13/v14) | "absolute": action[6] 직접 0.5 threshold (v16/v17)
"""
from __future__ import annotations

import re
import signal
import sys
import threading
import time
from pathlib import Path

import math
import xml.etree.ElementTree as ET

import numpy as np
import rclpy
import xacro
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Float32MultiArray, Float32, String, Int32
from std_srvs.srv import Trigger

# ── Dobot SDK 경로 ───────────────────────────────────────────────────────────
def _find_repo_root() -> Path:
    for parent in Path(__file__).resolve().parents:
        if (parent / "hardware" / "dobot" / "dobot_api.py").exists():
            return parent
    raise RuntimeError("repo root (hardware/dobot/dobot_api.py) not found")

_REPO = _find_repo_root()
_HARDWARE = _REPO / "hardware"
_DOBOT_SDK = _HARDWARE / "dobot"
for _p in [str(_HARDWARE), str(_DOBOT_SDK)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

ACTION_DIM = 7
ACTION_HORIZON = 16

_XACRO_PATH = _REPO / "ros2/src/e6_description/urdf/me6_robot.xacro"
_JOINT_NAMES = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6"]


def _load_joint_limits_deg() -> tuple[np.ndarray, np.ndarray]:
    """me6_robot.xacro에서 revolute 관절 위치 한계를 읽어 degree 단위로 반환."""
    try:
        doc = xacro.process_file(str(_XACRO_PATH))
        root = ET.fromstring(doc.toxml())
    except Exception as exc:
        raise RuntimeError(f"xacro 파싱 실패: {exc}")

    limits: dict[str, tuple[float, float]] = {}
    for joint in root.iter("joint"):
        name = joint.get("name", "")
        if name not in _JOINT_NAMES:
            continue
        lim = joint.find("limit")
        if lim is None:
            continue
        lower = math.degrees(float(lim.get("lower", "-360")))
        upper = math.degrees(float(lim.get("upper",  "360")))
        limits[name] = (lower, upper)

    mins = np.array([limits[n][0] for n in _JOINT_NAMES], dtype=np.float32)
    maxs = np.array([limits[n][1] for n in _JOINT_NAMES], dtype=np.float32)
    return mins, maxs


class ExecutorSupervisorNode(Node):

    def __init__(self):
        super().__init__("executor_supervisor_node")

        # 파라미터
        self.declare_parameter("robot_ip", "192.168.5.1")
        self.declare_parameter("dry_run", False)
        self.declare_parameter("no_camera", False)
        self.declare_parameter("action_mode", "absolute")  # "absolute" (v6) | "delta" (v8)
        self.declare_parameter("gripper_mode", "delta")    # "delta" (v13/v14 누산) | "absolute" (v16/v17 직접 threshold)
        self.declare_parameter("control_mode", "movj")     # "movj" | "servoj"
        # ServoJ 튜닝 파라미터 (control_mode=servoj 전용)
        # t: 목표 도달 시간(s). 보통 1/executor_hz 로 설정. -1=Dobot 기본값
        # aheadtime: PID D항 유사, 범위 [20, 100], -1=기본값(50)
        # gain: PID P항 유사, 범위 [200, 1000], -1=기본값(500)
        self.declare_parameter("servoj_t", -1.0)
        self.declare_parameter("servoj_aheadtime", -1.0)
        self.declare_parameter("servoj_gain", -1.0)
        self.declare_parameter("action_scale", 1.0)        # delta mode 출력 배율 (속도 조절)
        self.declare_parameter("max_delta_deg", 3.0)
        self.declare_parameter("min_tool_z", 101.0)
        self.declare_parameter("grip_close_threshold", 0.5)
        self.declare_parameter("grip_open_threshold", 0.4)
        self.declare_parameter("grip_close_latch_steps", 0)
        self.declare_parameter("movj_velocity", 70)
        self.declare_parameter("movj_accel", 60)
        self.declare_parameter("chunk_staleness_sec", 5.0)
        self.declare_parameter("steps_per_inference", 8)
        self.declare_parameter("executor_hz", 16.0)
        self.declare_parameter("approach_z_done", 85.0)
        self.declare_parameter("lift_z_done", 200.0)
        self.declare_parameter("stage_done_steps", 3)
        self.declare_parameter("bad_camera_consecutive", 10)
        self.declare_parameter("vacuum_check_enabled", True)
        self.declare_parameter("vacuum_check_z", 85.0)  # 미사용 — grip_enable_z가 z 게이트 역할 대신함
        self.declare_parameter("vacuum_timeout_sec", 1.0)
        self.declare_parameter("camera_black_mean", 8.0)
        self.declare_parameter("place_force_release_enabled", False)
        self.declare_parameter("place_z_threshold", 120.0)
        self.declare_parameter("max_steps", 500)
        self.declare_parameter("min_steps", 100)
        self.declare_parameter("home_tol_deg", 5.0)
        self.declare_parameter("home_consec_req", 16)
        self.declare_parameter("grip_enable_z", 0.0)  # 0=비활성, >0이면 tcp_z > 이 값일 때 gripper 강제 OFF
        # scripted lift: VLA가 lift phase에서 상승 못할 때 강제 MovL로 올림
        self.declare_parameter("scripted_lift_enabled", False)   # True=활성화
        self.declare_parameter("scripted_lift_target_z", 185.0)  # 목표 Z (mm)
        self.declare_parameter("scripted_lift_wait_frames", 48)  # 대기 프레임 수 (3초@16Hz)
        self.declare_parameter("scripted_lift_stall_z", 160.0)   # 이 Z 미달이면 stall 판정
        self.declare_parameter("scripted_lift_dz_thresh", 0.3)   # mm/frame 이하이면 stall
        # scripted return: VLA가 release phase에서 상승 못할 때 강제 MovL로 올림
        self.declare_parameter("scripted_return_enabled", True)
        self.declare_parameter("scripted_return_target_z", 200.0)  # 목표 Z (mm)
        self.declare_parameter("scripted_return_wait_frames", 16)  # 1초@16Hz 대기
        self.declare_parameter("scripted_return_stall_z", 150.0)   # 이 Z 미달이면 stall 판정

        self._dry_run = self.get_parameter("dry_run").value
        self._no_camera = self.get_parameter("no_camera").value
        self._action_mode = self.get_parameter("action_mode").value
        self._gripper_mode = self.get_parameter("gripper_mode").value
        self._action_scale = self.get_parameter("action_scale").value
        self._control_mode = self.get_parameter("control_mode").value
        self._servoj_t = self.get_parameter("servoj_t").value
        self._servoj_aheadtime = self.get_parameter("servoj_aheadtime").value
        self._servoj_gain = self.get_parameter("servoj_gain").value
        self._max_delta = self.get_parameter("max_delta_deg").value
        self._min_tool_z = self.get_parameter("min_tool_z").value
        self._grip_close = self.get_parameter("grip_close_threshold").value
        self._grip_open = self.get_parameter("grip_open_threshold").value
        self._grip_latch_steps = self.get_parameter("grip_close_latch_steps").value
        self._movj_v = self.get_parameter("movj_velocity").value
        self._movj_a = self.get_parameter("movj_accel").value
        self._staleness_sec = self.get_parameter("chunk_staleness_sec").value
        _spi = self.get_parameter("steps_per_inference").value
        self._steps_per_inference = _spi if _spi > 0 else ACTION_HORIZON
        _executor_hz = float(self.get_parameter("executor_hz").value)
        self._approach_z_done = self.get_parameter("approach_z_done").value
        self._lift_z_done = self.get_parameter("lift_z_done").value
        self._stage_done_steps = self.get_parameter("stage_done_steps").value
        self._bad_cam_limit = self.get_parameter("bad_camera_consecutive").value
        self._vacuum_check_enabled = self.get_parameter("vacuum_check_enabled").value
        self._vacuum_check_z = self.get_parameter("vacuum_check_z").value
        self._vacuum_timeout_sec = self.get_parameter("vacuum_timeout_sec").value
        self._black_mean = self.get_parameter("camera_black_mean").value
        self._place_force_release_enabled = self.get_parameter("place_force_release_enabled").value
        self._place_z_threshold = self.get_parameter("place_z_threshold").value
        self._max_steps = self.get_parameter("max_steps").value
        self._min_steps = self.get_parameter("min_steps").value
        self._home_tol_deg = self.get_parameter("home_tol_deg").value
        self._home_consec_req = self.get_parameter("home_consec_req").value
        self._grip_enable_z = self.get_parameter("grip_enable_z").value
        self._scripted_lift_enabled = self.get_parameter("scripted_lift_enabled").value
        self._scripted_lift_target_z = self.get_parameter("scripted_lift_target_z").value
        self._scripted_lift_wait_frames = self.get_parameter("scripted_lift_wait_frames").value
        self._scripted_lift_stall_z = self.get_parameter("scripted_lift_stall_z").value
        self._scripted_lift_dz_thresh = self.get_parameter("scripted_lift_dz_thresh").value
        self._scripted_return_enabled = self.get_parameter("scripted_return_enabled").value
        self._scripted_return_target_z = self.get_parameter("scripted_return_target_z").value
        self._scripted_return_wait_frames = self.get_parameter("scripted_return_wait_frames").value
        self._scripted_return_stall_z = self.get_parameter("scripted_return_stall_z").value

        # URDF 관절 위치 한계 (degree)
        self._joint_min_deg, self._joint_max_deg = _load_joint_limits_deg()
        self.get_logger().info(
            f"URDF joint limits 로드 완료: "
            f"min={self._joint_min_deg.tolist()} "
            f"max={self._joint_max_deg.tolist()}"
        )

        # 청크 상태
        self._chunk: np.ndarray | None = None       # (16, 7)
        self._chunk_idx = 0
        self._chunk_t = 0.0                          # 수신 시각
        self._chunk_lock = threading.Lock()

        # 로봇 상태 (camera_state_node에서 구독)
        self._current_deg = np.zeros(6, dtype=np.float32)
        self._tcp_z: float | None = None
        self._last_gripper = 0
        self._grip_latch_remaining = 0
        self._grip_cont = 0.0  # delta 모드 연속 gripper 누산기

        # 안전 감시 상태
        self._bad_streak = 0
        self._emergency_stop = False
        self._status = "RUNNING"

        # stage 완료 감지
        self._current_stage = "approach"   # 현재 stage (prompt로 갱신)
        self._done_streak = 0              # 완료 조건 연속 만족 카운트
        self._stage_done_published = False # 같은 stage에서 중복 발행 방지

        # 흡착 확인 (ToolDI 기반) — 값은 get_parameter에서 설정됨
        self._gripper_on_since: float | None = None  # gripper ON 시작 시각
        self._vacuum_confirmed = False                # 이 에피소드에서 흡착 확인됨
        self._arm_crossed_transport = False           # vacuum 확인 후 lift_z_done 통과 여부

        # task 완료 플래그
        self._task_complete = False

        # B+C 종료 조건 상태
        self._step_count = 0

        # ── 성능 측정 변수 ────────────────────────────────────────────────────
        self._move_start_time: float | None = None   # 첫 움직임 시각
        self._total_chunk_rcv: int = 0               # 수신된 청크 총 수
        self._chunk_count_at_move_start: int = 0     # 첫 움직임 시점의 청크 수
        self._infer_count: int = 0                   # 실제 궤적 계산 호출 수 (inference_bridge)
        self._infer_count_at_move_start: int = 0     # 첫 움직임 시점의 궤적 계산 수
        self._metrics_saved: bool = False            # suction ON 시 1회만 저장
        self._home_consec = 0
        self._left_init_pose = False   # j1..j3가 init tol 밖으로 나간 적 있을 때만 HOME 종료
        # j1..j3 기준 init 자세 (INIT_POSE_DEG 고정값 사용)
        self._init_joints_j123 = np.array(self.INIT_POSE_DEG[:3], dtype=np.float32)

        # 시작 자세 확인: canonical home에 도달하기 전 inference chunk 실행 차단
        # 이유: 이전 에피소드가 OOD 자세(예: j3=46°)로 끝난 뒤 재실행하면
        #       모델이 학습 분포(j3=53.8°) 밖에서 시작해 near-zero action 출력
        self._init_hold = True
        self._init_consec = 0

        # scripted lift 상태
        self._lift_frame_count = 0        # vacuum 확인 후 경과 프레임
        self._scripted_lifting = False    # 현재 강제 상승 중
        self._lift_vacuum_confirmed = False  # lift phase에서 ToolDI=1 확인됨
        self._scripted_movl_sent = False  # RelMovLUser 1회 발행 완료 여부
        self._post_lift_grip_hold = 0     # scripted lift 완료 후 grip 유지 잔여 프레임

        # scripted return 상태 (release phase에서 상승 못할 때 강제 상승)
        self._return_frame_count = 0
        self._scripted_returning = False
        self._scripted_return_sent = False

        # 구독
        self.create_subscription(Float32MultiArray, "/e6/policy/action_chunk", self._cb_chunk,       10)
        self.create_subscription(Int32,             "/e6/inference/count",     self._cb_infer_count, 10)
        self.create_subscription(Image,             "/e6/camera/image",         self._cb_img,         10)
        self.create_subscription(Float32MultiArray, "/e6/robot/state",          self._cb_state,       10)
        self.create_subscription(Float32,           "/e6/robot/tcp_z",          self._cb_tcpz,        10)
        self.create_subscription(String,            "/e6/task/prompt",          self._cb_prompt,      10)
        self.create_subscription(String,            "/e6/task/status",          self._cb_task_status, 10)
        self.create_subscription(String,            "/e6/supervisor/voice_override", self._cb_voice_override, 10)

        # 발행
        self._status_pub      = self.create_publisher(String, "/e6/supervisor/status",    10)
        self._gripper_cmd_pub = self.create_publisher(Float32, "/e6/gripper/commanded",   10)

        # 서비스
        self.create_service(Trigger, "/e6/emergency_stop", self._cb_estop)

        # 로봇 연결
        self._dashboard = None
        robot_ip = self.get_parameter("robot_ip").value
        if not self._dry_run:
            self._init_robot(robot_ip)

        # 타이머
        self.create_timer(1.0 / _executor_hz, self._executor_tick)
        self.create_timer(0.10,  self._supervisor_tick) # 10Hz

        self.get_logger().info(
            f"executor_supervisor_node 시작 — "
            f"robot={'연결됨' if self._dashboard else 'dry_run'} "
            f"control_mode={self._control_mode} "
            f"executor_hz={_executor_hz} max_delta={self._max_delta}° "
            f"min_tool_z={self._min_tool_z}mm steps_per_inference={self._steps_per_inference}/{ACTION_HORIZON}"
        )

    # ── 초기화 ──────────────────────────────────────────────────────────────

    # e6_v10 학습 데이터 기준 초기 자세 (degree) — robot_server.py V10_JOINT_HOME
    INIT_POSE_DEG = [91.3, 37.7, 53.8, -1.5, -87.8, 173.3]

    def _init_robot(self, robot_ip: str):
        try:
            from dobot_api import DobotApiDashboard  # type: ignore
            self._dashboard = DobotApiDashboard(robot_ip, 29999)
            self._dashboard.EnableRobot()
            time.sleep(0.5)
            j1, j2, j3, j4, j5, j6 = self.INIT_POSE_DEG
            self._dashboard.MovJ(j1, j2, j3, j4, j5, j6, 1, v=50, a=40)
            self.get_logger().info(f"초기 자세 이동 중: {self.INIT_POSE_DEG}")
            time.sleep(8.0)
            self.get_logger().info(f"Dobot dashboard 연결: {robot_ip}:29999")
        except KeyboardInterrupt:
            if self._dashboard is not None:
                try:
                    self._dashboard.StopRobot()
                except Exception:
                    pass
            raise
        except Exception as exc:
            self.get_logger().warn(f"Dobot 연결 실패 ({exc}) → dry_run 모드")
            self._dashboard = None

    # ── 구독 콜백 ────────────────────────────────────────────────────────────

    def _cb_infer_count(self, msg: Int32):
        self._infer_count = msg.data

    def _cb_chunk(self, msg: Float32MultiArray):
        data = np.array(msg.data, dtype=np.float32)
        if data.size != ACTION_HORIZON * ACTION_DIM:
            self.get_logger().warn(f"chunk 크기 이상: {data.size}")
            return
        self._total_chunk_rcv += 1
        with self._chunk_lock:
            self._chunk = data.reshape(ACTION_HORIZON, ACTION_DIM)[:self._steps_per_inference]
            self._chunk_idx = 0
            self._chunk_t = time.monotonic()

    def _cb_img(self, msg: Image):
        frame = np.frombuffer(msg.data, dtype=np.uint8)
        self._frame_mean = float(frame.mean()) if frame.size else 255.0

    def _cb_state(self, msg: Float32MultiArray):
        d = np.array(msg.data, dtype=np.float32)
        if d.size >= 6:
            self._current_deg = d[:6]

    def _cb_tcpz(self, msg: Float32):
        self._tcp_z = msg.data

    def _cb_task_status(self, msg: String):
        if msg.data == "TASK_COMPLETE" and not self._task_complete:
            self._task_complete = True
            self.get_logger().info("TASK_COMPLETE 수신 → 모션 정지")
            with self._chunk_lock:
                self._chunk = None
                self._chunk_idx = 0

    def _cb_prompt(self, msg: String):
        prompt = msg.data
        # prompt → stage 이름 추출
        if "approach" in prompt:
            stage = "approach"
        elif "pick" in prompt:
            stage = "pick"
        elif "lift" in prompt:
            stage = "lift"
        elif "move" in prompt:
            stage = "move"
        elif "place" in prompt:
            stage = "place"
        elif "release" in prompt:
            stage = "release"
        else:
            stage = prompt
        if stage != self._current_stage:
            self.get_logger().info(f"stage 변경: {self._current_stage} → {stage}")
            self._current_stage = stage
            self._done_streak = 0
            self._stage_done_published = False
            # approach로 돌아오면 흡착 상태 리셋 (새 에피소드)
            if stage == "approach":
                self._gripper_on_since = None
                self._vacuum_confirmed = False
                self._arm_crossed_transport = False
            # lift 진입 시 scripted lift 카운터 리셋
            if stage == "lift":
                self._lift_frame_count = 0
                self._scripted_lifting = False
                self._lift_vacuum_confirmed = False
                self._scripted_movl_sent = False
                self._post_lift_grip_hold = 0
            # release 진입 시 scripted return 카운터 리셋
            if stage == "release":
                self._return_frame_count = 0
                self._scripted_returning = False
                self._scripted_return_sent = False

    # ── 20Hz 실행 루프 ───────────────────────────────────────────────────────

    def _executor_tick(self):
        if self._emergency_stop or self._task_complete:
            return

        # 시작 자세 대기: j1..j3이 INIT_POSE 기준 home_tol_deg 이내 8프레임 연속 → 해제
        if self._init_hold:
            j_diff = np.abs(self._current_deg[:3] - self._init_joints_j123)
            if j_diff.max() < self._home_tol_deg:
                self._init_consec += 1
                if self._init_consec >= 8:
                    self._init_hold = False
                    self.get_logger().info(
                        f"[INIT_READY] 시작 자세 확인 → inference 시작 "
                        f"j1..j3={self._current_deg[:3].tolist()} "
                        f"diff_max={j_diff.max():.1f}°"
                    )
            else:
                self._init_consec = 0
                self.get_logger().info(
                    f"[INIT_WAIT] 시작 자세 대기 중 "
                    f"j1..j3={self._current_deg[:3].tolist()} "
                    f"diff_max={j_diff.max():.1f}° (허용={self._home_tol_deg}°)",
                    throttle_duration_sec=2.0,
                )
            return

        with self._chunk_lock:
            chunk = self._chunk
            idx = self._chunk_idx
            chunk_t = self._chunk_t

        if chunk is None:
            # 청크 없어도 마지막 gripper 상태 유지 (흡착 해제 방지)
            if self._dashboard is not None and not self._dry_run:
                try:
                    self._dashboard.ToolDO(1, self._last_gripper)
                except Exception:
                    pass
            return

        # staleness 체크
        if time.monotonic() - chunk_t > self._staleness_sec:
            with self._chunk_lock:
                self._chunk = None
            self.get_logger().warn("chunk staleness 초과 → 폐기")
            return

        if idx >= len(chunk):
            return  # 청크 소진, 다음 chunk 대기

        a = chunk[idx]  # (7,)

        if self._action_mode == "delta":
            # v8: velocity delta — action[:6] = deg/frame, 현재 위치에 누산
            delta = np.asarray(a[:6], dtype=np.float32) * self._action_scale
            clipped = np.abs(delta) > self._max_delta
            if clipped.any():
                delta = np.clip(delta, -self._max_delta, self._max_delta)
                self.get_logger().warn(
                    f"delta clamp: joints {np.where(clipped)[0].tolist()} "
                    f"max={np.abs(delta).max():.2f}°",
                    throttle_duration_sec=1.0,
                )
            target_deg = self._current_deg + delta
        else:
            # v6: 절대 목표 관절각 (action = next-position degree)
            target_deg = np.asarray(a[:6], dtype=np.float32)
            delta = target_deg - self._current_deg
            clipped = np.abs(delta) > self._max_delta
            if clipped.any():
                target_deg = self._current_deg + np.clip(delta, -self._max_delta, self._max_delta)
                self.get_logger().warn(
                    f"delta clamp: joints {np.where(clipped)[0].tolist()} "
                    f"max={np.abs(delta).max():.2f}°",
                    throttle_duration_sec=1.0,
                )

        # URDF 관절 위치 한계 클램프 (속도 제한과 별개로 절대 범위 초과 방지)
        over_min = target_deg < self._joint_min_deg
        over_max = target_deg > self._joint_max_deg
        if over_min.any() or over_max.any():
            self.get_logger().warn(
                f"joint limit 초과 클램프: "
                f"min_viol={np.where(over_min)[0].tolist()} "
                f"max_viol={np.where(over_max)[0].tolist()}",
                throttle_duration_sec=1.0,
            )
        target_deg = np.clip(target_deg, self._joint_min_deg, self._joint_max_deg)

        # 첫 움직임 감지
        if self._move_start_time is None and np.max(np.abs(delta)) > 0.05:
            self._move_start_time = time.monotonic()
            self._chunk_count_at_move_start = self._total_chunk_rcv
            self._infer_count_at_move_start = self._infer_count

        # 그리퍼 hysteresis
        if self._action_mode == "delta" and self._gripper_mode == "delta":
            # v13/v14: 누산 — accum이 ±0.5 threshold
            self._grip_cont += float(a[6] if len(a) > 6 else 0.0)
            grip_raw = self._grip_cont
            if grip_raw > 0.5:
                hys = 1
            elif grip_raw < -0.5:
                hys = 0
            else:
                hys = self._last_gripper
        else:
            # v6/v16/v17: action[6] 절대값 직접 threshold (0.0 or 1.0)
            grip_raw = float(a[6]) if len(a) > 6 else 0.0
            if grip_raw >= self._grip_close:
                hys = 1
            elif grip_raw <= self._grip_open:
                hys = 0
            else:
                hys = self._last_gripper

        if self._grip_latch_steps > 0:
            if hys == 1:
                self._grip_latch_remaining = max(self._grip_latch_remaining, self._grip_latch_steps)
            if self._grip_latch_remaining > 0:
                hys = 1
                self._grip_latch_remaining -= 1

        tool_on = int(hys)

        # 접근 중 조기 흡착 방지: tcp_z > grip_enable_z 이면 gripper 강제 OFF
        # 효과: 모델이 높은 Z에서 grip=1 출력해도 흡착되지 않음 → state에 gripper=0 유지
        #       → 학습 데이터 분포(approach는 grip=0)와 일치 → 모델이 계속 하강
        if (self._grip_enable_z > 0
                and tool_on == 1
                and self._tcp_z is not None
                and self._tcp_z > self._grip_enable_z):
            tool_on = 0

        # transport 중 suction sensor=1이면 OFF 명령 차단
        if (self._action_mode == "delta"
                and tool_on == 0
                and self._last_gripper == 1
                and self._tcp_z is not None
                and self._tcp_z > self._lift_z_done
                and self._dashboard is not None
                and not self._dry_run):
            di = None
            try:
                res = self._dashboard.ToolDI(1)
                if res:
                    m = re.search(r"\{(\d+)\}", str(res))
                    if m:
                        di = int(m.group(1))
            except Exception:
                pass
            if di == 1:
                tool_on = 1
                self.get_logger().info(
                    f"[GUARD] transport 중 suction=1 → OFF 차단 (z={self._tcp_z:.1f}mm)",
                    throttle_duration_sec=1.0,
                )

        # ── transport 통과 추적 ──────────────────────────────────────────────
        if (self._vacuum_confirmed
                and self._tcp_z is not None
                and self._tcp_z > self._lift_z_done):
            self._arm_crossed_transport = True

        # ── 강제 release (vacuum 확인 + transport 통과 + place 높이 도달) ────
        if (self._place_force_release_enabled
                and self._vacuum_confirmed
                and self._arm_crossed_transport
                and self._tcp_z is not None
                and self._tcp_z <= self._place_z_threshold
                and tool_on == 1):
            tool_on = 0
            self.get_logger().info(
                f"[PLACE] 강제 release (z={self._tcp_z:.1f}mm ≤ {self._place_z_threshold:.1f}mm)"
            )

        # lift 단계 gripper 유지: VLA grip≈0.026 → ToolDO(1,0) 방지
        # 핵심: grip 1→0 전환이 PhaseTracker._released=True 를 세팅 → z>z_lift 후
        #       lift lock 해제 시 "return"→"release" 전환, arm 재하강의 원인
        if (self._current_stage == "lift"
                and self._last_gripper == 1
                and tool_on == 0):
            tool_on = 1

        # scripted lift 중 + 완료 후 N프레임 gripper 강제 ON (이중 보호)
        if self._scripted_lifting or self._post_lift_grip_hold > 0:
            tool_on = 1
            if self._post_lift_grip_hold > 0:
                self._post_lift_grip_hold -= 1

        # bad camera → MovJ 스킵 (현재 위치 유지)
        camera_hold = hasattr(self, "_frame_mean") and self._frame_mean < self._black_mean

        # ── scripted lift: lift phase에서 VLA가 상승 못할 때 강제 상승 ──────
        # 트리거: lift stage + ToolDI=1(vacuum 확인) + wait_frames 경과 + Z stall
        # 데이터: pick Z=121.8mm → 180mm 평균 47프레임, 목표 185mm
        use_scripted_lift = False
        if (self._scripted_lift_enabled
                and self._current_stage == "lift"
                and self._dashboard is not None
                and not self._dry_run
                and self._tcp_z is not None
                and not camera_hold):
            z_now = self._tcp_z

            # ToolDI(1) 로 실제 흡착 확인 (lift phase 진입 후 1회만 폴링)
            if not self._lift_vacuum_confirmed:
                di = None
                try:
                    res = self._dashboard.ToolDI(1)
                    if res:
                        m = re.search(r"\{(\d+)\}", str(res))
                        if m:
                            di = int(m.group(1))
                except Exception:
                    pass
                if di == 1:
                    self._lift_vacuum_confirmed = True
                    self.get_logger().info(
                        f"[SCRIPTED_LIFT] vacuum 확인 (ToolDI=1), 대기 시작 z={z_now:.1f}mm"
                    )

            # vacuum 확인됐으면 프레임 카운트
            if self._lift_vacuum_confirmed:
                self._lift_frame_count += 1

            # stall 판정: vacuum 확인 + wait_frames 경과 + Z 낮음
            stall = (self._lift_vacuum_confirmed
                     and self._lift_frame_count >= self._scripted_lift_wait_frames
                     and z_now < self._scripted_lift_stall_z)

            if stall or self._scripted_lifting:
                if not self._scripted_lifting:
                    self._scripted_lifting = True
                    self.get_logger().info(
                        f"[SCRIPTED_LIFT] lift stall 감지 → 강제 상승 시작 "
                        f"z={z_now:.1f}mm frame={self._lift_frame_count}"
                    )
                if z_now < self._scripted_lift_target_z:
                    use_scripted_lift = True
                    # 1회만 RelMovLUser 발행 (non-blocking 큐 방식)
                    if not self._scripted_movl_sent:
                        dz = self._scripted_lift_target_z - z_now
                        try:
                            self._dashboard.RelMovLUser(0, 0, dz, 0, 0, 0,
                                                        v=self._movj_v, a=self._movj_a)
                            self._scripted_movl_sent = True
                            self.get_logger().info(
                                f"[SCRIPTED_LIFT] RelMovLUser dz=+{dz:.1f}mm 발행 "
                                f"(z={z_now:.1f}mm → {self._scripted_lift_target_z:.1f}mm)"
                            )
                        except Exception as exc:
                            self.get_logger().warn(f"[SCRIPTED_LIFT] RelMovLUser 실패: {exc}")
                else:
                    self._scripted_lifting = False
                    self._scripted_movl_sent = False
                    self._lift_vacuum_confirmed = False
                    self._post_lift_grip_hold = 32  # 2초@16Hz — transport 안정화 대기
                    self.get_logger().info(
                        f"[SCRIPTED_LIFT] 목표 도달 z={z_now:.1f}mm → VLA 복귀, grip hold 32f"
                    )

        # ── scripted return: release phase에서 VLA가 상승 못할 때 강제 상승 ────
        use_scripted_return = False
        if (self._scripted_return_enabled
                and self._current_stage == "release"
                and self._dashboard is not None
                and not self._dry_run
                and self._tcp_z is not None
                and not camera_hold
                and self._last_gripper == 0):
            z_now = self._tcp_z
            self._return_frame_count += 1
            stall = (self._return_frame_count >= self._scripted_return_wait_frames
                     and z_now < self._scripted_return_stall_z)
            if stall or self._scripted_returning:
                if not self._scripted_returning:
                    self._scripted_returning = True
                    self.get_logger().info(
                        f"[SCRIPTED_RETURN] stall 감지 → 강제 상승 시작 "
                        f"z={z_now:.1f}mm frame={self._return_frame_count}"
                    )
                if z_now < self._scripted_return_target_z:
                    use_scripted_return = True
                    if not self._scripted_return_sent:
                        dz = self._scripted_return_target_z - z_now
                        try:
                            self._dashboard.RelMovLUser(0, 0, dz, 0, 0, 0,
                                                        v=self._movj_v, a=self._movj_a)
                            self._scripted_return_sent = True
                            self.get_logger().info(
                                f"[SCRIPTED_RETURN] RelMovLUser dz=+{dz:.1f}mm "
                                f"(z={z_now:.1f}mm → {self._scripted_return_target_z:.1f}mm)"
                            )
                        except Exception as exc:
                            self.get_logger().warn(f"[SCRIPTED_RETURN] RelMovLUser 실패: {exc}")
                else:
                    self._scripted_returning = False
                    self._scripted_return_sent = False
                    self.get_logger().info(
                        f"[SCRIPTED_RETURN] 목표 도달 z={z_now:.1f}mm → VLA 복귀"
                    )

        # 로봇 명령 전송
        if self._dashboard is not None:
            try:
                if use_scripted_lift:
                    # RelMovLUser 실행 중: 관절 명령 완전 스킵 (VLA와 충돌 방지)
                    self.get_logger().info(
                        f"[SCRIPTED_LIFT] 상승 중 z={self._tcp_z:.1f}mm",
                        throttle_duration_sec=1.0,
                    )
                elif use_scripted_return:
                    self.get_logger().info(
                        f"[SCRIPTED_RETURN] 상승 중 z={self._tcp_z:.1f}mm",
                        throttle_duration_sec=1.0,
                    )
                elif not camera_hold:
                    j1, j2, j3, j4, j5, j6 = (float(x) for x in target_deg)
                    if self._control_mode == "servoj":
                        self._dashboard.ServoJ(
                            j1, j2, j3, j4, j5, j6,
                            t=self._servoj_t,
                            aheadtime=self._servoj_aheadtime,
                            gain=self._servoj_gain,
                        )
                    else:
                        self._dashboard.MovJ(j1, j2, j3, j4, j5, j6, 1,
                                             v=self._movj_v, a=self._movj_a)
                self._dashboard.ToolDO(1, tool_on)

                # 실행 로그: 처음 5스텝 + 이후 10스텝마다
                if self._step_count < 5 or self._step_count % 10 == 0:
                    stale_ms = (time.monotonic() - chunk_t) * 1000.0
                    self.get_logger().info(
                        f"[EXEC] step={self._step_count} mode={self._control_mode} "
                        f"chunk_idx={idx}/{len(chunk)} stale={stale_ms:.0f}ms "
                        f"delta=[{', '.join(f'{d:+.2f}' for d in delta)}] "
                        f"target=[{', '.join(f'{x:.1f}' for x in target_deg)}] "
                        f"grip={tool_on}"
                    )
            except Exception as exc:
                self.get_logger().warn(f"로봇 명령 실패 ({self._control_mode}): {exc}", throttle_duration_sec=2.0)

        # ── suction ON 전환 감지 → 성능 측정 저장 ───────────────────────────
        if tool_on == 1 and self._last_gripper == 0 and not self._metrics_saved:
            self._metrics_saved = True
            suction_on_time = time.monotonic()
            suction_on_str = __import__("datetime").datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            elapsed = (suction_on_time - self._move_start_time) if self._move_start_time is not None else float("nan")
            chunks_used = self._total_chunk_rcv - self._chunk_count_at_move_start
            infer_used = self._infer_count - self._infer_count_at_move_start
            lines = [
                "=" * 50,
                f"[Inference Metrics]",
                f"  suction ON 시각     : {suction_on_str}",
                f"  이동 시작 → ON 시간 : {elapsed:.2f} 초",
                f"  궤적 계산 호출 횟수 : {infer_used} 회",
                f"  총 소요 시간        : {elapsed:.2f} 초",
                "=" * 50,
                "",
            ]
            log_path = __import__("os").path.expanduser("~/Desktop/inference_metrics.txt")
            with open(log_path, "a", encoding="utf-8") as f:
                f.write("\n".join(lines) + "\n")
            self.get_logger().info(
                f"[METRICS] suction ON — 이동→ON: {elapsed:.2f}s, 청크: {chunks_used}회 → {log_path}"
            )

        self._last_gripper = tool_on
        self._gripper_cmd_pub.publish(Float32(data=float(tool_on)))

        # ── ToolDI 흡착 확인 ────────────────────────────────────────────────
        # 위치/stage 조건 없이 gripper ON 시 센서값만으로 흡착 판정
        if (self._vacuum_check_enabled and self._dashboard is not None
                and not self._vacuum_confirmed):
            now = time.monotonic()
            # gripper ON 시점 기록 (z 임계값 이하에서 처음 ON 됐을 때만)
            if tool_on == 1 and self._gripper_on_since is None:
                self._gripper_on_since = now

            if self._gripper_on_since is not None:
                # ToolDI(1) 읽기
                di = None
                try:
                    res = self._dashboard.ToolDI(1)
                    if res:
                        m = re.search(r"\{(\d+)\}", str(res))
                        if m:
                            di = int(m.group(1))
                except Exception:
                    pass

                if di == 1:
                    # 흡착 확인 → 청크 리셋 + lift 시작
                    self._vacuum_confirmed = True
                    with self._chunk_lock:
                        self._chunk = None
                        self._chunk_idx = 0
                    self.get_logger().info(
                        f"[VACUUM] 흡착 확인 (z={self._tcp_z:.1f}mm) → 청크 리셋, lift 시작"
                        if self._tcp_z is not None else "[VACUUM] 흡착 확인 → 청크 리셋"
                    )
                elif (now - self._gripper_on_since) > self._vacuum_timeout_sec:
                    # timeout이지만 emergency stop 없이 계속 하강
                    # 물리 접촉까지 계속 내려가다가 ToolDI=1 감지되면 올라감
                    self.get_logger().warn(
                        f"[VACUUM] {self._vacuum_timeout_sec:.1f}s 내 흡착 미감지 — 계속 하강",
                        throttle_duration_sec=2.0,
                    )

        with self._chunk_lock:
            self._chunk_idx += 1

        # ── B+C 종료 조건 ─────────────────────────────────────────────────────
        self._step_count += 1

        # B: 최대 step 초과 → 강제 종료
        if self._step_count > self._max_steps:
            self.get_logger().warn(
                f"TIMEOUT: {self._step_count} step 초과 ({self._max_steps}) → 강제 종료"
            )
            self._task_complete = True
            self._status_pub.publish(String(data="TASK_COMPLETE"))
            return

        # C: init 자세 복귀 감지 → 정상 종료 (min_steps 이후 + init 이탈 경험 후만)
        j_diff = np.abs(self._current_deg[:3] - self._init_joints_j123)
        if j_diff.max() >= self._home_tol_deg:
            self._left_init_pose = True

        if self._step_count > self._min_steps and self._left_init_pose:
            if j_diff.max() < self._home_tol_deg:
                self._home_consec += 1
                if self._home_consec >= self._home_consec_req:
                    self.get_logger().info(
                        f"HOME: j1..j3 init 복귀 {self._home_consec}프레임 연속 "
                        f"(diff_max={j_diff.max():.2f}°) → 정상 종료"
                    )
                    self._task_complete = True
                    self._status_pub.publish(String(data="TASK_COMPLETE"))
                    return
            else:
                self._home_consec = 0

    # ── 10Hz 감시 루프 ───────────────────────────────────────────────────────

    def _supervisor_tick(self):
        if self._emergency_stop:
            self._status_pub.publish(String(data="FAIL_SAFETY:emergency_stop"))
            return

        if self._task_complete:
            return

        status = "RUNNING"

        # bad camera 스트릭 (no_camera 또는 dry_run이면 스킵)
        if not self._no_camera and not self._dry_run:
            frame_mean = getattr(self, "_frame_mean", 255.0)
            if frame_mean < self._black_mean:
                self._bad_streak += 1
                if self._bad_streak > self._bad_cam_limit:
                    status = "FAIL_SAFETY:bad_camera"
                    self.get_logger().warn(f"bad_camera streak={self._bad_streak}")
            else:
                self._bad_streak = 0

        # min_tool_z 안전 한계
        if (not self._dry_run
                and self._dashboard is not None
                and self._tcp_z is not None
                and self._tcp_z <= self._min_tool_z):
            status = f"FAIL_SAFETY:min_tool_z({self._tcp_z:.1f}mm)"
            self.get_logger().warn(f"Tool Z={self._tcp_z:.1f}mm ≤ {self._min_tool_z:.1f}mm")

        # STAGE_DONE 판정 (FAIL이 없을 때만)
        if status == "RUNNING" and not self._stage_done_published and self._tcp_z is not None:
            done = False
            stage = self._current_stage
            z = self._tcp_z
            g = self._last_gripper

            if stage == "approach":
                done = z <= self._approach_z_done
            elif stage == "pick":
                done = z >= self._lift_z_done and g == 1
            elif stage == "place":
                done = z >= self._lift_z_done and g == 0
            # move: timeout에 맡김 (센서 조건 없음)

            if done:
                self._done_streak += 1
                if self._done_streak >= self._stage_done_steps:
                    status = f"STAGE_DONE:{stage}"
                    self._stage_done_published = True
                    self.get_logger().info(
                        f"[STAGE_DONE] {stage} | TCP_Z={z:.1f}mm gripper={g}"
                    )
            else:
                self._done_streak = 0

        self._status = status
        self._status_pub.publish(String(data=status))

    # ── 긴급 정지 서비스 ─────────────────────────────────────────────────────

    def _cb_voice_override(self, msg: String):
        if msg.data.strip().upper() == "STOP":
            self._emergency_stop = True
            self.get_logger().error("[VOICE] 음성 STOP 명령 수신 → 긴급 정지")
            if self._dashboard is not None:
                try:
                    self._dashboard.EmergencyStop(0)
                except Exception:
                    pass

    def _cb_estop(self, request, response):
        self._emergency_stop = True
        self.get_logger().error("긴급 정지 호출!")
        if self._dashboard is not None:
            try:
                self._dashboard.EmergencyStop(0)
            except Exception:
                pass
        response.success = True
        response.message = "emergency stop activated"
        return response


def main(args=None):
    rclpy.init(args=args)
    node = None

    def _shutdown(signum, frame):
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, _shutdown)

    try:
        node = ExecutorSupervisorNode()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node is not None:
            with node._chunk_lock:
                node._chunk = None
            if node._dashboard is not None:
                try:
                    node._dashboard.StopRobot()
                except Exception:
                    pass
            node.destroy_node()
        try:
            rclpy.shutdown()
        except Exception:
            pass


if __name__ == "__main__":
    main()
