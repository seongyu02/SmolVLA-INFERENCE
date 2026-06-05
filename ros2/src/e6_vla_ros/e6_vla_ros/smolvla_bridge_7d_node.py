#!/usr/bin/env python3
"""
smolvla_bridge_7d_node — SmolVLA 7D HTTP 정책 서버 연결 브릿지

smolvla_bridge_node 의 7D 복사본.
7D 서버(serve_policy_smolvla_7d.py)에 state/action 7D 그대로 전달.

구독 토픽:
  /e6/camera/image_512     sensor_msgs/Image          512×512 RGB (HIK)
  /e6/camera/zed_image_512 sensor_msgs/Image          512×512 RGB (ZED)
  /e6/robot/state          std_msgs/Float32MultiArray [j1..j6 deg, gripper]
  /e6/task/prompt          std_msgs/String
  /e6/task/status          std_msgs/String

발행 토픽:
  /e6/policy/action_chunk  std_msgs/Float32MultiArray (chunk_size*7 flatten)

파라미터:
  server_host        (str,   default "127.0.0.1")
  server_port        (int,   default 8001)
  infer_hz           (float, default 1.25)
  save_debug_images  (bool,  default False)
"""
from __future__ import annotations

import base64
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np
import requests
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, DurabilityPolicy
from sensor_msgs.msg import Image
from std_msgs.msg import Float32MultiArray, String, Int32

IMG_SIZE = (512, 512)
STATE_DIM = 7
ACTION_DIM = 7
DEFAULT_PROMPT = "pick up the orange box from the left side and place it on the right side"
ACTION_HORIZON = 16  # executor_supervisor_node가 기대하는 고정 청크 길이
# executor_supervisor_node.INIT_POSE_DEG j1..j3 와 동일
INIT_POSE_J123 = np.array([91.3, 37.7, 53.8], dtype=np.float32)
HTTP_ACT_TIMEOUT_SEC = 5.0


class SmolVLABridge7DNode(Node):

    def __init__(self):
        super().__init__("smolvla_bridge_7d_node")

        self.declare_parameter("server_host", "127.0.0.1")
        self.declare_parameter("server_port", 8001)
        self.declare_parameter("infer_hz", 1.25)
        self.declare_parameter("save_debug_images", False)
        self.declare_parameter("init_pose_tol_deg", 5.0)
        self.declare_parameter("wait_for_init_pose", True)

        host = self.get_parameter("server_host").value
        port = self.get_parameter("server_port").value
        infer_hz = self.get_parameter("infer_hz").value
        self._save_debug = self.get_parameter("save_debug_images").value
        self._init_pose_tol = float(self.get_parameter("init_pose_tol_deg").value)
        self._wait_for_init_pose = bool(self.get_parameter("wait_for_init_pose").value)

        self._url = f"http://{host}:{port}"
        self._session = requests.Session()

        self._latest_img: np.ndarray | None = None
        self._latest_zed: np.ndarray | None = None
        self._latest_state: np.ndarray | None = None   # 7D [j1..j6, grip]
        self._latest_prompt: str = DEFAULT_PROMPT
        self._lock = threading.Lock()

        self._inference_running = False
        self._task_complete = False
        self._shutting_down = False
        self._executor = ThreadPoolExecutor(max_workers=1)
        self._infer_call_count = 0
        self._init_pose_armed = self._wait_for_init_pose  # 시작 1회만 init 확인

        qos_transient = QoSProfile(durability=DurabilityPolicy.TRANSIENT_LOCAL, depth=1)

        self.create_subscription(Image, "/e6/camera/image_512",
                                 self._cb_img, 10)
        self.create_subscription(Image, "/e6/camera/zed_image_512",
                                 self._cb_zed, 10)
        self.create_subscription(Float32MultiArray, "/e6/robot/state",
                                 self._cb_state, 10)
        self.create_subscription(String, "/e6/task/prompt",
                                 self._cb_prompt, qos_transient)
        self.create_subscription(String, "/e6/task/status",
                                 self._cb_task_status, 10)

        self._chunk_pub = self.create_publisher(Float32MultiArray,
                                                "/e6/policy/action_chunk", 10)
        self._infer_count_pub = self.create_publisher(Int32,
                                                      "/e6/inference/count", 10)

        threading.Thread(target=self._wait_for_server, daemon=True).start()

        self.create_timer(1.0 / infer_hz, self._maybe_infer)
        self.create_timer(10.0, self._retry_health)

        self.get_logger().info(
            f"[7D] smolvla_bridge_7d_node 시작 — server={self._url} "
            f"state_dim={STATE_DIM} action_dim={ACTION_DIM} "
            f"infer_interval={1.0/infer_hz:.2f}s"
        )

    def _wait_for_server(self):
        import time
        self.get_logger().info(f"[7D] SmolVLA 서버 대기 중: {self._url}")
        while rclpy.ok() and not self._shutting_down:
            try:
                resp = self._session.get(f"{self._url}/healthz", timeout=2)
                if resp.status_code == 200 and resp.json().get("model_loaded"):
                    self.get_logger().info(f"[7D] SmolVLA 서버 연결 완료: {resp.json()}")
                    return
            except Exception:
                pass
            time.sleep(1.0)

    def _retry_health(self):
        try:
            resp = self._session.get(f"{self._url}/healthz", timeout=2)
            if resp.status_code != 200 or not resp.json().get("model_loaded"):
                self.get_logger().warn("[7D] SmolVLA 서버 unhealthy", throttle_duration_sec=30.0)
        except Exception as exc:
            self.get_logger().warn(f"[7D] 헬스체크 실패: {exc}", throttle_duration_sec=30.0)

    def _cb_img(self, msg: Image):
        with self._lock:
            self._latest_img = np.frombuffer(msg.data, dtype=np.uint8).reshape(
                msg.height, msg.width, 3).copy()

    def _cb_zed(self, msg: Image):
        with self._lock:
            self._latest_zed = np.frombuffer(msg.data, dtype=np.uint8).reshape(
                msg.height, msg.width, 3).copy()

    def _cb_state(self, msg: Float32MultiArray):
        with self._lock:
            self._latest_state = np.array(msg.data, dtype=np.float32)

    def _cb_task_status(self, msg: String):
        if not self._task_complete and (
            msg.data == "TASK_COMPLETE" or msg.data.startswith("FAIL_SAFETY")
        ):
            self._task_complete = True
            self.get_logger().info(f"[7D] 추론 정지 ({msg.data})")

    def _cb_prompt(self, msg: String):
        with self._lock:
            changed = self._latest_prompt != msg.data
            self._latest_prompt = msg.data
        if changed:
            self.get_logger().info(f"[7D] prompt 변경: {msg.data!r}")

    def _maybe_infer(self):
        if self._shutting_down or self._task_complete or self._inference_running:
            return

        with self._lock:
            img = self._latest_img
            zed = self._latest_zed
            state7 = self._latest_state
            prompt = self._latest_prompt

        if img is None or state7 is None:
            return

        obs_state_7d = np.asarray(state7, dtype=np.float32).ravel()
        if obs_state_7d.shape[0] != STATE_DIM:
            self.get_logger().warn(
                f"[7D] state len={obs_state_7d.shape[0]}, expected {STATE_DIM}",
                throttle_duration_sec=5.0,
            )
            return

        if self._init_pose_armed:
            j_diff = float(np.abs(obs_state_7d[:3] - INIT_POSE_J123).max())
            if j_diff > self._init_pose_tol:
                self.get_logger().info(
                    f"[7D] init pose 대기 — current={np.round(obs_state_7d[:3], 1).tolist()} "
                    f"target={INIT_POSE_J123.tolist()} diff={j_diff:.1f}° "
                    f"(허용={self._init_pose_tol:.1f}°)",
                    throttle_duration_sec=2.0,
                )
                return
            self._init_pose_armed = False
            self.get_logger().info(
                f"[7D] init pose 확인 → 추론 시작 "
                f"j1..j3={np.round(obs_state_7d[:3], 1).tolist()}"
            )

        zed_frame = zed.copy() if zed is not None else np.zeros((*IMG_SIZE, 3), dtype=np.uint8)

        obs = {
            "img": img.copy(),
            "zed": zed_frame,
            "state7": obs_state_7d.copy(),
            "prompt": prompt,
        }
        self._inference_running = True
        self._executor.submit(self._run_infer, obs)

    def _run_infer(self, obs: dict):
        if self._shutting_down:
            self._inference_running = False
            return
        try:
            t_total0 = time.monotonic()
            t0 = time.monotonic()
            img_b64 = _image_msg_to_b64png_from_array(obs["img"])
            zed_b64 = _image_msg_to_b64png_from_array(obs["zed"])
            encode_ms = (time.monotonic() - t0) * 1000.0
            payload = {
                "state": obs["state7"].tolist(),
                "image1_b64": img_b64,
                "image2_b64": zed_b64,
                "task": obs["prompt"],
            }
            t0 = time.monotonic()
            resp = self._session.post(
                f"{self._url}/act", json=payload, timeout=HTTP_ACT_TIMEOUT_SEC,
            )
            resp.raise_for_status()
            data = resp.json()
            http_ms = (time.monotonic() - t0) * 1000.0

            actions7 = np.asarray(data["actions"], dtype=np.float32)
            if actions7.ndim == 1:
                actions7 = actions7.reshape(1, -1)
            if actions7.shape[-1] != ACTION_DIM:
                raise RuntimeError(
                    f"Expected action dim {ACTION_DIM}, got shape {actions7.shape}"
                )

            if actions7.shape[0] > ACTION_HORIZON:
                actions7 = actions7[:ACTION_HORIZON]
            elif actions7.shape[0] < ACTION_HORIZON:
                repeat = (ACTION_HORIZON + actions7.shape[0] - 1) // actions7.shape[0]
                actions7 = np.tile(actions7, (repeat, 1))[:ACTION_HORIZON]

            self._infer_call_count += 1
            self._infer_count_pub.publish(Int32(data=self._infer_call_count))

            grip_vals = [f"{actions7[i, 6]:+.3f}" for i in range(len(actions7))]
            total_ms = (time.monotonic() - t_total0) * 1000.0
            self.get_logger().info(
                f"[7D] 추론 완료 {total_ms:.0f}ms "
                f"(encode={encode_ms:.0f}ms http={http_ms:.0f}ms) "
                f"shape={actions7.shape} prompt={obs['prompt']!r}"
            )
            self.get_logger().info(
                f"[7D] generated state7={np.round(obs['state7'], 1).tolist()}"
            )
            self.get_logger().info(
                f"[7D] action_7={np.round(actions7[0], 3).tolist()}"
            )
            self.get_logger().info(
                f"[7D] joint_delta={np.round(actions7[0, :6], 3).tolist()}  "
                f"gripper_cmd={actions7[0, 6]:+.3f}"
            )
            self.get_logger().info(f"[7D] suction_seq: {grip_vals}")

            msg = Float32MultiArray(data=actions7.flatten().tolist())
            self._chunk_pub.publish(msg)

        except Exception as exc:
            self.get_logger().error(f"[7D] 추론 실패: {exc}")
        finally:
            self._inference_running = False

    def destroy_node(self):
        self._shutting_down = True
        try:
            self._executor.shutdown(wait=False, cancel_futures=True)
        except TypeError:
            self._executor.shutdown(wait=False)
        try:
            self._session.close()
        except Exception:
            pass
        super().destroy_node()


def _image_msg_to_b64png_from_array(arr: np.ndarray) -> str:
    """(H,W,3) uint8 RGB → base64 PNG."""
    _, buf = cv2.imencode(".png", cv2.cvtColor(arr, cv2.COLOR_RGB2BGR))
    return base64.b64encode(buf.tobytes()).decode()


def main(args=None):
    rclpy.init(args=args)
    node = SmolVLABridge7DNode()
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
