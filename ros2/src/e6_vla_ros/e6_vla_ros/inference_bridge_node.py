#!/usr/bin/env python3
"""
inference_bridge_node — obs 조립 → WebSocket 추론 → action_chunk 발행

구독 토픽:
  /e6/camera/image        sensor_msgs/Image                HIK 카메라 (exterior_image_1_left)
  /e6/camera/zed_image    sensor_msgs/Image                ZED 카메라 (exterior_image_2_left)
  /e6/robot/state         std_msgs/Float32MultiArray       [j1..j6 deg, gripper]
  /e6/task/prompt         std_msgs/String

발행 토픽:
  /e6/policy/action_chunk std_msgs/Float32MultiArray       (16*7 flatten)

파라미터:
  server_host  (str,   default "127.0.0.1")
  server_port  (int,   default 8000)
  infer_hz          (float, default 1.25)   — action_horizon(16) / hz(20) ≈ 0.8s 주기
  save_debug_images (bool,  default False)  — 추론마다 HIK+ZED 이미지 ~/debug_inference/ 저장
"""
from __future__ import annotations

import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, DurabilityPolicy
from sensor_msgs.msg import Image
from std_msgs.msg import Float32MultiArray, String, Int32

# ── openpi_client 경로 ───────────────────────────────────────────────────────
_REPO = Path(__file__).resolve().parents[4]
_CLIENT_SRC = _REPO / "packages" / "openpi-client" / "src"
_VENV_SITE = Path.home() / "move-one" / "min-imum" / "move-one" / "lib" / "python3.10" / "site-packages"
for _p in [str(_CLIENT_SRC), str(_VENV_SITE)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)


def _image_msg_to_numpy(msg: Image) -> np.ndarray:
    return np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, 3).copy()


class InferenceBridgeNode(Node):

    def __init__(self):
        super().__init__("inference_bridge_node")

        # 파라미터
        self.declare_parameter("server_host", "127.0.0.1")
        self.declare_parameter("server_port", 8000)
        self.declare_parameter("infer_hz", 2.0)
        self.declare_parameter("save_debug_images", False)
        self.declare_parameter("action_mode", "absolute")  # "absolute" (v6) | "delta" (v8)

        host = self.get_parameter("server_host").value
        port = self.get_parameter("server_port").value
        infer_hz = self.get_parameter("infer_hz").value
        self._save_debug = self.get_parameter("save_debug_images").value
        self._action_mode = self.get_parameter("action_mode").value
        self._debug_dir = Path.home() / "debug_inference"
        self._debug_count = 0
        if self._save_debug:
            self._debug_dir.mkdir(parents=True, exist_ok=True)
            self.get_logger().info(f"디버그 이미지 저장: {self._debug_dir}")

        # 최신 obs 캐시 (항상 가장 최근 값 유지)
        self._latest_img: np.ndarray | None = None      # HIK
        self._latest_zed: np.ndarray | None = None      # ZED
        self._latest_state: np.ndarray | None = None
        self._latest_prompt: str = "pick up the orange box from the left side and place it on the right side"
        self._lock = threading.Lock()

        # 추론 상태
        self._inference_running = False
        self._task_complete = False
        self._executor = ThreadPoolExecutor(max_workers=1)

        # 구독
        self.create_subscription(Image,             "/e6/camera/image",     self._cb_img,     10)
        self.create_subscription(Image,             "/e6/camera/zed_image", self._cb_zed,     10)
        self.create_subscription(Float32MultiArray, "/e6/robot/state",      self._cb_state,   10)

        qos_transient = QoSProfile(
            durability=DurabilityPolicy.TRANSIENT_LOCAL, depth=1
        )
        self.create_subscription(String, "/e6/task/prompt",  self._cb_prompt,      qos_transient)
        self.create_subscription(String, "/e6/task/status",  self._cb_task_status, 10)

        # 발행
        self._chunk_pub = self.create_publisher(Float32MultiArray, "/e6/policy/action_chunk", 10)
        self._infer_count_pub = self.create_publisher(Int32, "/e6/inference/count", 10)
        self._infer_call_count: int = 0

        self._server_host = host
        self._server_port = port

        # WebSocket 서버 연결 — 백그라운드에서 시도 (서버가 없어도 크래시 없음)
        self._policy = None
        self._connecting = False
        threading.Thread(target=self._connect_policy_bg, daemon=True).start()

        # 추론 트리거 타이머
        self.create_timer(1.0 / infer_hz, self._maybe_infer)
        # 연결 재시도 타이머 (10초마다)
        self.create_timer(10.0, self._retry_connect)
        self.get_logger().info(
            f"inference_bridge_node 시작 — server=ws://{host}:{port} "
            f"infer_interval={1.0/infer_hz:.2f}s"
        )

    # ── WebSocket 연결 ───────────────────────────────────────────────────────

    def _connect_policy_bg(self):
        """백그라운드 스레드에서 서버 연결 시도 (블로킹 허용)."""
        if self._connecting:
            return
        self._connecting = True
        try:
            from openpi_client.websocket_client_policy import WebsocketClientPolicy  # type: ignore
            self.get_logger().info(
                f"정책 서버 연결 시도: ws://{self._server_host}:{self._server_port}"
            )
            policy = WebsocketClientPolicy(host=self._server_host, port=self._server_port)
            meta = policy.get_server_metadata()
            self._policy = policy
            self.get_logger().info(f"정책 서버 연결 완료: {meta}")
        except Exception as exc:
            self.get_logger().warn(f"정책 서버 연결 실패 (재시도 대기 중): {exc}")
            self._policy = None
        finally:
            self._connecting = False

    def _retry_connect(self):
        """서버 미연결 상태면 재시도."""
        if self._policy is None and not self._connecting:
            threading.Thread(target=self._connect_policy_bg, daemon=True).start()

    # ── 구독 콜백 ────────────────────────────────────────────────────────────

    def _cb_img(self, msg: Image):
        with self._lock:
            self._latest_img = _image_msg_to_numpy(msg)

    def _cb_zed(self, msg: Image):
        with self._lock:
            self._latest_zed = _image_msg_to_numpy(msg)

    def _cb_state(self, msg: Float32MultiArray):
        with self._lock:
            self._latest_state = np.array(msg.data, dtype=np.float32)

    def _cb_task_status(self, msg: String):
        if not self._task_complete and (
            msg.data == "TASK_COMPLETE" or msg.data.startswith("FAIL_SAFETY")
        ):
            self._task_complete = True
            self.get_logger().info(f"추론 정지 ({msg.data})")

    def _cb_prompt(self, msg: String):
        with self._lock:
            changed = self._latest_prompt != msg.data
            self._latest_prompt = msg.data
        if changed:
            self.get_logger().info(f"prompt 변경: {msg.data!r}")

    # ── 추론 트리거 ──────────────────────────────────────────────────────────

    def _maybe_infer(self):
        if self._policy is None or self._task_complete:
            return
        if self._inference_running:
            return  # 이전 추론 진행 중 → 스킵

        with self._lock:
            img = self._latest_img
            zed = self._latest_zed
            state = self._latest_state
            prompt = self._latest_prompt

        if img is None or state is None:
            return  # obs 아직 없음

        # ZED 없으면 zeros로 대체 (카메라 미연결 시 graceful degradation)
        zed_frame = zed.copy() if zed is not None else np.zeros((224, 224, 3), dtype=np.uint8)

        # obs 스냅샷 (추론 중 덮어써도 안전하게 복사)
        obs = {
            "observation/exterior_image_1_left": img.copy(),
            "observation/exterior_image_2_left": zed_frame,
            "observation/state":                 state.copy(),
            "prompt":                            prompt,
        }
        # 디버그 이미지 저장 (매 추론마다)
        if self._save_debug:
            self._save_debug_images(obs)

        self._inference_running = True
        self._executor.submit(self._run_infer, obs)

    def _save_debug_images(self, obs: dict):
        try:
            import cv2  # type: ignore
            n = self._debug_count
            hik = obs["observation/exterior_image_1_left"]
            zed = obs["observation/exterior_image_2_left"]
            cv2.imwrite(str(self._debug_dir / f"{n:04d}_hik.jpg"),
                        cv2.cvtColor(hik, cv2.COLOR_RGB2BGR))
            cv2.imwrite(str(self._debug_dir / f"{n:04d}_zed.jpg"),
                        cv2.cvtColor(zed, cv2.COLOR_RGB2BGR))
            self._debug_count += 1
        except Exception as exc:
            self.get_logger().warn(f"디버그 이미지 저장 실패: {exc}", throttle_duration_sec=5.0)

    def _run_infer(self, obs: dict):
        try:
            self._infer_call_count += 1
            self._infer_count_pub.publish(Int32(data=self._infer_call_count))
            result = self._policy.infer(obs)
            actions = np.asarray(result["actions"], dtype=np.float32)  # (16, 7)
            state = obs["observation/state"]
            self.get_logger().info(
                f"추론 완료 shape={actions.shape} "
                f"prompt={obs['prompt']!r}"
            )
            grip_vals = [f"{actions[i,6]:+.3f}" for i in range(len(actions))]
            self.get_logger().info(f"state:      {np.round(state[:6],1).tolist()}  grip={state[6]:.1f}")
            self.get_logger().info(f"action[0]:  {np.round(actions[0,:6],1).tolist()}  grip={actions[0,6]:+.3f}")
            self.get_logger().info(f"action[15]: {np.round(actions[15,:6],1).tolist()}  grip={actions[15,6]:+.3f}")
            self.get_logger().info(f"grip_seq:   {grip_vals}")
            if self._action_mode == "delta":
                # delta mode: action 자체가 delta (deg/frame)
                self.get_logger().info(
                    f"delta[0]:  {np.round(actions[0,:6],2).tolist()}  max={np.abs(actions[0,:6]).max():.2f}°"
                )
                self.get_logger().info(
                    f"delta[15]: {np.round(actions[15,:6],2).tolist()}  max={np.abs(actions[15,:6]).max():.2f}°"
                )
                max_delta0 = np.abs(actions[0, :6]).max()
            else:
                # absolute mode: action - state = 이동 거리
                delta0  = actions[0,  :6] - state[:6]
                delta15 = actions[15, :6] - state[:6]
                self.get_logger().info(
                    f"delta[0]:  {['%+.1f'%d for d in delta0]}  max={np.abs(delta0).max():.1f}°"
                )
                self.get_logger().info(
                    f"delta[15]: {['%+.1f'%d for d in delta15]}  max={np.abs(delta15).max():.1f}°"
                )
                max_delta0 = np.abs(delta0).max()
            # J5 부호 반전 = mode flip → reject (absolute mode 전용)
            # delta mode에서는 act_j5가 속도값이므로 부호 체크 불필요
            if self._action_mode == "absolute":
                cur_j5  = float(state[4])
                act_j5  = float(actions[0, 4])
                if (cur_j5 < 0) != (act_j5 < 0):
                    self.get_logger().warn(
                        f"chunk 거부: J5 부호 반전 cur={cur_j5:.1f}° → act={act_j5:.1f}° (mode flip) → 재추론 대기"
                    )
                    return
            msg = Float32MultiArray(data=actions.flatten().tolist())
            self._chunk_pub.publish(msg)
        except Exception as exc:
            self.get_logger().error(f"추론 실패: {exc}")
        finally:
            self._inference_running = False


def main(args=None):
    rclpy.init(args=args)
    node = InferenceBridgeNode()
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
