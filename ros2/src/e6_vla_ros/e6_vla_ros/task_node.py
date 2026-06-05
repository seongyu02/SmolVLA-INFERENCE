#!/usr/bin/env python3
"""
task_node — task_sequence 상태머신 (episode 모드) / per-frame phase 감지 (per_frame 모드) /
            단일 고정 프롬프트 (single 모드, v13).

구독 토픽:
  /e6/supervisor/status   std_msgs/String
  /e6/robot/state         std_msgs/Float32MultiArray  [j1..j6 deg, gripper]  (per_frame 모드)
  /e6/robot/tcp_z         std_msgs/Float32                                    (per_frame 모드)

발행 토픽:
  /e6/task/prompt         std_msgs/String  (QoS: transient_local)
  /e6/task/status         std_msgs/String  10

파라미터 (공통):
  prompt_mode    (str,   default "episode")  "episode" (v6) | "per_frame" (v8) | "single" (v13)
  task_sequence  (str,   default "pick_from_left")
  stage_timeout_sec (float, default 0.0)
  loop_sequence  (bool,  default False)

파라미터 (per_frame 모드 전용):
  source_side    (str,   default "left")   "left" | "right"
  target_side    (str,   default "right")  "right" | "left"
  z_lift         (float, default 180.0)    phase 구분 기준 TCP Z (mm)
  grip_threshold (float, default 0.5)      gripper ON 판단 임계값
  phase_hz       (float, default 16.0)     phase 감지 + 프롬프트 발행 주파수
  return_z_done  (float, default 180.0)    return 완료 기준 TCP Z (mm, 이 값 이상)
  return_done_steps (int, default 5)       return 완료 조건 연속 만족 프레임 수

파라미터 (single 모드 전용):
  source_side    (str,   default "left")   "left" | "right"
  prompt_variant (int,   default -1)       0~2 고정 선택, -1이면 랜덤
"""
from __future__ import annotations

import random

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, DurabilityPolicy
from std_msgs.msg import Float32MultiArray, Float32, String

V16_PHASE_PROMPTS: dict[str, dict[str, str]] = {
    "left": {
        "approach":  "approach the orange box on the left side",
        "pick_up":   "pick up the orange box on the left side",
        "lift":      "lift the orange box",
        "transport": "move the orange box to the right section",
        "place":     "place the orange box in the right section",
        "release":   "release the orange box",
    },
    "right": {
        "approach":  "approach the orange box on the right side",
        "pick_up":   "pick up the orange box on the right side",
        "lift":      "lift the orange box",
        "transport": "move the orange box to the left section",
        "place":     "place the orange box in the left section",
        "release":   "release the orange box",
    },
}

# PhaseTracker "grasp"/"return" → v16 key 변환
_V16_PHASE_KEY: dict[str, str] = {
    "approach":  "approach",
    "grasp":     "pick_up",
    "lift":      "lift",
    "transport": "transport",
    "place":     "place",
    "release":   "release",
    "return":    "release",  # v16/v17에 return phase 없음 → release 유지
}

V8_PHASE_PROMPTS: dict[str, str] = {
    "approach":  "move the arm down to approach the orange box on the {source}",
    "grasp":     "grasp the orange box on the {source}",
    "lift":      "lift the orange box from the {source}",
    "transport": "lift and carry the orange box to the {target}",
    "place":     "lower the orange box onto the {target}",
    "release":   "release the orange box on the {target}",
    "return":    "return the arm to the ready position",
}


class PhaseTracker:
    """gripper 상태 + TCP Z → 7-phase 실시간 분류 (v8 contract)."""

    Z_LIFT = 180.0
    TRANSITION_FRAMES = 5

    def __init__(self, z_lift: float = 180.0, transition_frames: int = 5,
                 grip_threshold: float = 0.5, grasp_z_max: float = 120.0,
                 min_hold_frames: int = 16, pick_prearm_z: float = 0.0):
        self.z_lift = z_lift
        self.transition_frames = transition_frames
        self.grip_threshold = grip_threshold
        self.grasp_z_max = grasp_z_max
        self.min_hold_frames = min_hold_frames
        self.pick_prearm_z = pick_prearm_z

        self._phase = "approach"
        self._prev_gripper = 0
        self._crossed_lift = False
        self._released = False
        self._trans_counter = 0
        self._trans_type: str | None = None
        self._high_z_grip = False
        self._hold_counter = 0      # 현재 phase 최소 유지 잔여 프레임

    def reset(self):
        self._phase = "approach"
        self._prev_gripper = 0
        self._crossed_lift = False
        self._released = False
        self._trans_counter = 0
        self._trans_type = None
        self._high_z_grip = False
        self._hold_counter = 0

    def update(self, gripper_raw: float, tcp_z: float) -> str:
        gripper = 1 if gripper_raw >= self.grip_threshold else 0
        valid_grasp_z = tcp_z <= self.grasp_z_max

        # gripper 전환 감지
        if gripper != self._prev_gripper:
            if gripper == 1 and not valid_grasp_z:
                # 높은 Z에서 grip spike → 무시 (접근 중 오발동)
                self._high_z_grip = True
            elif gripper == 0 and self._high_z_grip:
                # 높은 Z spike 해제 → 전환 무시, approach 유지
                self._high_z_grip = False
            else:
                self._high_z_grip = False
                self._trans_counter = self.transition_frames
                self._trans_type = "close" if gripper == 1 else "open"
                if gripper == 0:
                    self._released = True

        # crossed_lift 플래그 갱신
        if gripper == 1 and tcp_z > self.z_lift:
            self._crossed_lift = True
        if gripper == 0:
            self._crossed_lift = False

        # phase 결정 (우선순위 순)
        if self._high_z_grip:
            phase = "approach"
        elif self._trans_counter > 0 and self._trans_type == "close":
            phase = "grasp"
        elif self._trans_counter > 0 and self._trans_type == "open":
            phase = "release"
        elif gripper == 0 and self._released:
            phase = "return"
        elif gripper == 0 and self.pick_prearm_z > 0 and tcp_z <= self.pick_prearm_z:
            phase = "grasp"  # Z 기반 미리 pick_up (학습 레이블 타이밍 맞춤)
        elif gripper == 0:
            phase = "approach"
        elif gripper == 1 and tcp_z > self.z_lift:
            phase = "transport"
        elif gripper == 1 and self._crossed_lift:
            phase = "place"
        else:
            phase = "lift"

        # lift lock: tcp_z > z_lift 가 돼야만 "lift" 탈출 (brief grip drop/spike 무시)
        # 효과: z=85mm에서 잡고 들어올리는 동안 grip 노이즈로 release→pick_up 사이클 방지
        if self._phase == "lift" and tcp_z <= self.z_lift:
            if phase not in ("lift", "transport"):
                if not hasattr(self, "_lift_lock_logged"):
                    self._lift_lock_logged = True
                self._lift_lock_count = getattr(self, "_lift_lock_count", 0) + 1
                phase = "lift"

        # min_hold_frames: phase 전환 시 최소 N 프레임 유지 (빠른 사이클 방지)
        if phase != self._phase:
            if self._hold_counter > 0:
                phase = self._phase  # 아직 전환 불가
            else:
                self._hold_counter = self.min_hold_frames

        if self._trans_counter > 0:
            self._trans_counter -= 1
        if self._hold_counter > 0:
            self._hold_counter -= 1

        self._prev_gripper = gripper
        self._phase = phase
        return phase

    @property
    def phase(self) -> str:
        return self._phase


V14_PROMPTS: dict[str, str] = {
    "left":  "pick up the orange box from the left side and place it on the right side",
    "right": "pick up the orange box from the right side and place it on the left side",
}

V13_PROMPTS: dict[str, list[str]] = {
    "left": [
        "move the orange box from the left to the right",
        "pick up the orange box from the left side and place it on the right side",
        "grasp the orange box on the left and put it down on the right",
    ],
    "right": [
        "move the orange box from the right to the left",
        "pick up the orange box from the right side and place it on the left side",
        "grasp the orange box on the right and put it down on the left",
    ],
    # can generalization test (vision frozen → language grounding)
    "can_left": [
        "pick up the can from the left side and place it on the right side",
        "grasp the can on the left and put it down on the right",
        "move the can from the left to the right",
    ],
    "can_right": [
        "pick up the can from the right side and place it on the left side",
        "grasp the can on the right and put it down on the left",
        "move the can from the right to the left",
    ],
    # egg generalization test
    "egg_left": [
        "pick up the egg from the left side and place it on the right side",
        "grasp the egg on the left and put it down on the right",
        "move the egg from the left to the right",
    ],
    "egg_right": [
        "pick up the egg from the right side and place it on the left side",
        "grasp the egg on the right and put it down on the left",
        "move the egg from the right to the left",
    ],
}

TASK_PRESETS: dict[str, str] = {
    # v2 episode-level prompts (orange box, A↔B)
    "pick_from_left":  "pick up the orange box from the left side and place it on the right side",
    "pick_from_right": "pick up the orange box from the right side and place it on the left side",
    # v1 segment-level prompts (red block) — kept for backward compatibility
    "approach":    "approach red object",
    "pick":        "pick red object",
    "move_left":   "move object to left",
    "move_right":  "move object to right",
    "move_middle": "move object to middle",
    "place_left":  "place object to left",
    "place_right": "place object to right",
    "place_middle": "place object to middle",
    "return":      "return",
    "init_hold":   "init_hold",
}


class TaskNode(Node):

    def __init__(self):
        super().__init__("task_node")

        # 공통 파라미터
        self.declare_parameter("prompt_mode", "episode")  # "episode" | "per_frame" | "single"
        self.declare_parameter("task_sequence", "pick_from_left")
        self.declare_parameter("stage_timeout_sec", 0.0)
        self.declare_parameter("loop_sequence", False)

        # per_frame / single 모드 공용
        self.declare_parameter("source_side", "left")
        self.declare_parameter("target_side", "right")

        # per_frame 모드 전용 파라미터
        self.declare_parameter("z_lift", 180.0)
        self.declare_parameter("grip_threshold", 0.5)
        self.declare_parameter("grasp_z_max", 120.0)   # grasp 진입 허용 최대 TCP Z (mm)
        self.declare_parameter("min_hold_frames", 16)  # phase 최소 유지 프레임 (빠른 사이클 방지)
        self.declare_parameter("pick_prearm_z", 0.0)   # Z 기반 pick_up 선진입 임계값 (mm), 0=비활성
        self.declare_parameter("phase_hz", 16.0)
        self.declare_parameter("return_z_done", 180.0)
        self.declare_parameter("return_done_steps", 5)

        # single 모드 전용
        self.declare_parameter("prompt_variant", -1)   # 0~2 고정 선택, -1이면 랜덤
        self.declare_parameter("prompt_text", "")      # 직접 입력 시 이 값 우선
        self.declare_parameter("prompt_dataset", "v13")  # "v13" (6 variant) | "v14" (anchor 2개)

        self._prompt_mode = self.get_parameter("prompt_mode").value
        seq_str = self.get_parameter("task_sequence").value
        self._timeout = self.get_parameter("stage_timeout_sec").value
        self._loop = self.get_parameter("loop_sequence").value

        self._seq = [s.strip() for s in seq_str.split(",") if s.strip()]
        self._idx = 0
        self._stage_start = self.get_clock().now()
        self._done = False

        # transient_local: 나중에 구독해도 최신값 받음
        qos = QoSProfile(durability=DurabilityPolicy.TRANSIENT_LOCAL, depth=1)
        self._prompt_pub  = self.create_publisher(String, "/e6/task/prompt",  qos)
        self._status_pub  = self.create_publisher(String, "/e6/task/status",  10)

        # supervisor status 구독 (episode 모드에서만 stage 전환에 사용)
        self.create_subscription(String, "/e6/supervisor/status", self._cb_status, 10)

        # 음성 명령 구독 — voice_command_node가 변환한 prompt를 즉시 반영
        self.create_subscription(String, "/e6/task/voice_command", self._cb_voice_command, 10)

        if self._prompt_mode == "per_frame":
            self._source_side = self.get_parameter("source_side").value
            self._target_side = self.get_parameter("target_side").value
            self._return_z_done = self.get_parameter("return_z_done").value
            self._return_done_steps = self.get_parameter("return_done_steps").value

            self._phase_tracker = PhaseTracker(
                z_lift=self.get_parameter("z_lift").value,
                transition_frames=5,
                grip_threshold=self.get_parameter("grip_threshold").value,
                grasp_z_max=self.get_parameter("grasp_z_max").value,
                min_hold_frames=self.get_parameter("min_hold_frames").value,
                pick_prearm_z=self.get_parameter("pick_prearm_z").value,
            )
            self._latest_gripper: float = 0.0
            self._latest_tcp_z: float = 200.0
            self._return_streak: int = 0

            self.create_subscription(Float32MultiArray, "/e6/robot/state", self._cb_state,  10)
            self.create_subscription(Float32,           "/e6/robot/tcp_z", self._cb_tcpz,   10)

            phase_hz = self.get_parameter("phase_hz").value
            self.create_timer(1.0 / phase_hz, self._phase_tick)

            self.get_logger().info(
                f"task_node 시작 (per_frame) — "
                f"source={self._source_side} target={self._target_side} "
                f"z_lift={self.get_parameter('z_lift').value}mm "
                f"phase_hz={phase_hz}"
            )

        elif self._prompt_mode == "per_frame_v16":
            self._source_side = self.get_parameter("source_side").value
            self._target_side = self.get_parameter("target_side").value

            self._phase_tracker = PhaseTracker(
                z_lift=self.get_parameter("z_lift").value,
                transition_frames=5,
                grip_threshold=self.get_parameter("grip_threshold").value,
                grasp_z_max=self.get_parameter("grasp_z_max").value,
                min_hold_frames=self.get_parameter("min_hold_frames").value,
                pick_prearm_z=self.get_parameter("pick_prearm_z").value,
            )
            self._latest_gripper: float = 0.0
            self._latest_tcp_z: float = 200.0

            self.create_subscription(Float32MultiArray, "/e6/robot/state", self._cb_state, 10)
            self.create_subscription(Float32, "/e6/robot/tcp_z", self._cb_tcpz, 10)

            phase_hz = self.get_parameter("phase_hz").value
            self.create_timer(1.0 / phase_hz, self._phase_tick_v16)

            self.get_logger().info(
                f"task_node 시작 (per_frame_v16) — "
                f"source={self.get_parameter('source_side').value} "
                f"z_lift={self.get_parameter('z_lift').value}mm "
                f"phase_hz={phase_hz}"
            )

        elif self._prompt_mode == "single":
            custom = self.get_parameter("prompt_text").value.strip()
            if custom:
                chosen = custom
            else:
                source = self.get_parameter("source_side").value
                dataset = self.get_parameter("prompt_dataset").value
                if dataset == "v14":
                    anchor = V14_PROMPTS.get(source)
                    if anchor is None:
                        self.get_logger().error(
                            f"source_side='{source}' 는 V14_PROMPTS에 없는 키입니다."
                        )
                        raise ValueError(f"invalid source_side: {source!r}")
                    chosen = anchor
                else:
                    variants = V13_PROMPTS.get(source)
                    if variants is None:
                        self.get_logger().error(
                            f"source_side='{source}' 는 V13_PROMPTS에 없는 키입니다."
                        )
                        raise ValueError(f"invalid source_side: {source!r}")
                    variant_idx = self.get_parameter("prompt_variant").value
                    if variant_idx < 0 or variant_idx >= len(variants):
                        chosen = random.choice(variants)
                    else:
                        chosen = variants[variant_idx]
            self._prompt_pub.publish(String(data=chosen))
            self.get_logger().info(
                f"task_node 시작 (single) — dataset={self.get_parameter('prompt_dataset').value} prompt={chosen!r}"
            )

        else:
            # episode 모드: 기존 stage-based 동작
            if self._timeout > 0:
                self.create_timer(0.5, self._check_timeout)
            self._publish_current()
            self.get_logger().info(
                f"task_node 시작 (episode) — sequence={self._seq} "
                f"timeout={self._timeout}s loop={self._loop}"
            )

    # ── 음성 명령 콜백 ───────────────────────────────────────────────────────

    def _cb_voice_command(self, msg: String):
        prompt = msg.data.strip()
        if not prompt:
            return
        self.get_logger().info(f"[VOICE] 음성 명령 수신 → prompt 즉시 발행: '{prompt}'")
        self._prompt_pub.publish(String(data=prompt))

        # per_frame_v16 모드: source_side도 갱신해 이후 PhaseTracker 프롬프트 방향 반영
        if self._prompt_mode == "per_frame_v16":
            if "left side" in prompt and "from the left" in prompt:
                self._source_side = "left"
                self._target_side = "right"
            elif "right side" in prompt and "from the right" in prompt:
                self._source_side = "right"
                self._target_side = "left"

    # ── 구독 콜백 (per_frame 모드) ────────────────────────────────────────────

    def _cb_state(self, msg: Float32MultiArray):
        d = msg.data
        if len(d) >= 7:
            self._latest_gripper = float(d[6])

    def _cb_tcpz(self, msg: Float32):
        self._latest_tcp_z = msg.data

    # ── per_frame phase 감지 타이머 ───────────────────────────────────────────

    def _phase_tick(self):
        if self._done:
            return

        phase = self._phase_tracker.update(self._latest_gripper, self._latest_tcp_z)

        # return 완료 감지 → TASK_COMPLETE
        if phase == "return" and self._latest_tcp_z >= self._return_z_done:
            self._return_streak += 1
            if self._return_streak >= self._return_done_steps:
                self.get_logger().info("=" * 60)
                self.get_logger().info("TASK_COMPLETE (return phase 완료)")
                self.get_logger().info("=" * 60)
                self._status_pub.publish(String(data="TASK_COMPLETE"))
                self._done = True
                return
        else:
            self._return_streak = 0

        prompt = V8_PHASE_PROMPTS[phase].format(
            source=self._source_side,
            target=self._target_side,
        )
        self._prompt_pub.publish(String(data=prompt))

    # task_id 매핑 (tasks.jsonl 순서와 일치)
    _V16_TASK_ID: dict[str, dict[str, int]] = {
        "left":  {"approach": 0, "pick_up": 1, "lift": 2, "transport": 3, "place": 4, "release": 5},
        "right": {"approach": 6, "pick_up": 7, "lift": 2, "transport": 8, "place": 9, "release": 5},
    }

    def _phase_tick_v16(self):
        if self._done:
            return
        phase = self._phase_tracker.update(self._latest_gripper, self._latest_tcp_z)
        key = _V16_PHASE_KEY.get(phase, "release")
        prompt = V16_PHASE_PROMPTS[self._source_side][key]
        task_id = self._V16_TASK_ID.get(self._source_side, {}).get(key, -1)

        if not hasattr(self, "_last_v16_key") or self._last_v16_key != key:
            lock_cnt = getattr(self, "_lift_lock_count", 0)
            self.get_logger().info(
                f"[phase_v16] {getattr(self, '_last_v16_key', '?')} → {key} "
                f"(task_id={task_id}) grip={self._latest_gripper:.2f} z={self._latest_tcp_z:.1f}mm"
                + (f" [lift_lock held {lock_cnt}f]" if lock_cnt > 0 else "")
            )
            self._last_v16_key = key
            self._lift_lock_count = 0  # 전환 시 카운터 리셋

        self._prompt_pub.publish(String(data=prompt))

    # ── supervisor status 콜백 ────────────────────────────────────────────────

    def _cb_status(self, msg: String):
        if self._done:
            return
        status = msg.data

        if status.startswith("STAGE_DONE:") and self._prompt_mode == "episode":
            self.get_logger().info(f"supervisor STAGE_DONE 수신: {status}")
            self._advance_stage()

        elif status == "TASK_COMPLETE" and self._prompt_mode in ("single", "per_frame_v16"):
            # single 모드: executor가 B+C 종료를 감지 → /e6/supervisor/status 로 발행
            # → 여기서 /e6/task/status 로 중계 (MCAP 기록 + executor 수신 용)
            self.get_logger().info("=" * 60)
            self.get_logger().info("TASK_COMPLETE (executor 종료 감지)")
            self.get_logger().info("=" * 60)
            self._status_pub.publish(String(data="TASK_COMPLETE"))
            self._done = True

        elif status.startswith("FAIL_SAFETY"):
            self.get_logger().error(f"안전 이상 감지 — task 중단: {status}")
            self._status_pub.publish(String(data=status))
            self._done = True

    # ── timeout 체크 타이머 ───────────────────────────────────────────────────

    def _check_timeout(self):
        if self._done:
            return
        elapsed = (self.get_clock().now() - self._stage_start).nanoseconds / 1e9
        if elapsed >= self._timeout:
            self.get_logger().info(
                f"stage '{self._seq[self._idx]}' timeout ({elapsed:.1f}s >= {self._timeout}s)"
            )
            self._advance_stage()

    # ── stage 전환 ────────────────────────────────────────────────────────────

    def _advance_stage(self):
        self._idx += 1
        if self._idx >= len(self._seq):
            if self._loop:
                self._idx = 0
                self.get_logger().info("전체 sequence 완료 → 처음으로 루프")
            else:
                self.get_logger().info("=" * 60)
                self.get_logger().info("TASK_COMPLETE: 모든 stage 완료! 모션 정지")
                self.get_logger().info("=" * 60)
                self._status_pub.publish(String(data="TASK_COMPLETE"))
                self._done = True
                return
        self._stage_start = self.get_clock().now()
        self._publish_current()

    # ── 현재 stage 발행 ───────────────────────────────────────────────────────

    def _publish_current(self):
        if self._idx >= len(self._seq):
            return
        key = self._seq[self._idx]
        prompt = TASK_PRESETS.get(key, key)  # 프리셋에 없으면 key 자체를 prompt로
        self._prompt_pub.publish(String(data=prompt))
        self.get_logger().info(f"[stage {self._idx}/{len(self._seq)-1}] prompt={prompt!r}")


def main(args=None):
    rclpy.init(args=args)
    node = TaskNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        try:
            rclpy.shutdown()
        except Exception:
            pass


if __name__ == "__main__":
    main()
