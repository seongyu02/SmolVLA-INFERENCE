#!/usr/bin/env python3
"""
smolvla_bridge_node — SmolVLA HTTP 정책 서버 연결 브릿지

inference_bridge_node 의 SmolVLA 대체 버전.
WebSocket 대신 HTTP POST /act, action 13D→7D 변환.

구독 토픽:
  /e6/camera/image_512     sensor_msgs/Image          512×512 RGB (HIK)
  /e6/camera/zed_image_512 sensor_msgs/Image          512×512 RGB (ZED)
  /e6/robot/state          std_msgs/Float32MultiArray [j1..j6 deg, gripper]
  /e6/robot/tcp            std_msgs/Float32MultiArray [tx,ty,tz,rx,ry,rz]
  /e6/task/prompt          std_msgs/String
  /e6/task/status          std_msgs/String

발행 토픽:
  /e6/policy/action_chunk  std_msgs/Float32MultiArray (chunk_size*7 flatten)

파라미터:
  server_host        (str,   default "127.0.0.1")
  server_port        (int,   default 8000)
  infer_hz           (float, default 1.25)
  save_debug_images  (bool,  default False)
"""
from __future__ import annotations

import base64
import io
import threading
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
DEFAULT_PROMPT = "pick up the orange box from the left side and place it on the right side"
ACTION_HORIZON = 16  # executor_supervisor_node가 기대하는 고정 청크 길이
ACTION_DIM = 7
HTTP_ACT_TIMEOUT_SEC = 5.0  # Ctrl+C 시 추론 스레드 블로킹 최소화 (추론 ~1.6s)


def _image_msg_to_b64png(msg: Image) -> str:
    arr = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, 3)
    _, buf = cv2.imencode(".png", cv2.cvtColor(arr, cv2.COLOR_RGB2BGR))
    return base64.b64encode(buf.tobytes()).decode()


class SmolVLABridgeNode(Node):

    def __init__(self):
        super().__init__("smolvla_bridge_node")

        self.declare_parameter("server_host", "127.0.0.1")
        self.declare_parameter("server_port", 8000)
        self.declare_parameter("infer_hz", 1.25)
        self.declare_parameter("save_debug_images", False)

        host = self.get_parameter("server_host").value
        port = self.get_parameter("server_port").value
        infer_hz = self.get_parameter("infer_hz").value
        self._save_debug = self.get_parameter("save_debug_images").value

        self._url = f"http://{host}:{port}"
        self._session = requests.Session()

        self._latest_img: np.ndarray | None = None
        self._latest_zed: np.ndarray | None = None
        self._latest_state: np.ndarray | None = None   # 7D [j1..j6, grip]
        self._latest_tcp: np.ndarray | None = None     # 6D [tx,ty,tz,rx,ry,rz]
        self._latest_prompt: str = DEFAULT_PROMPT
        self._lock = threading.Lock()

        self._inference_running = False
        self._task_complete = False
        self._shutting_down = False
        self._executor = ThreadPoolExecutor(max_workers=1)
        self._infer_call_count = 0

        qos_transient = QoSProfile(durability=DurabilityPolicy.TRANSIENT_LOCAL, depth=1)

        self.create_subscription(Image, "/e6/camera/image_512",
                                 self._cb_img, 10)
        self.create_subscription(Image, "/e6/camera/zed_image_512",
                                 self._cb_zed, 10)
        self.create_subscription(Float32MultiArray, "/e6/robot/state",
                                 self._cb_state, 10)
        self.create_subscription(Float32MultiArray, "/e6/robot/tcp",
                                 self._cb_tcp, 10)
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
            f"smolvla_bridge_node 시작 — server={self._url} "
            f"infer_interval={1.0/infer_hz:.2f}s"
        )

    # ── 서버 대기 / 헬스체크 ─────────────────────────────────────────────────

    def _wait_for_server(self):
        import time
        self.get_logger().info(f"SmolVLA 서버 대기 중: {self._url}")
        while rclpy.ok() and not self._shutting_down:
            try:
                resp = self._session.get(f"{self._url}/healthz", timeout=2)
                if resp.status_code == 200 and resp.json().get("model_loaded"):
                    self.get_logger().info(f"SmolVLA 서버 연결 완료: {resp.json()}")
                    return
            except Exception:
                pass
            time.sleep(1.0)

    def _retry_health(self):
        try:
            resp = self._session.get(f"{self._url}/healthz", timeout=2)
            if resp.status_code != 200 or not resp.json().get("model_loaded"):
                self.get_logger().warn("SmolVLA 서버 unhealthy", throttle_duration_sec=30.0)
        except Exception as exc:
            self.get_logger().warn(f"헬스체크 실패: {exc}", throttle_duration_sec=30.0)

    # ── 구독 콜백 ────────────────────────────────────────────────────────────

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

    def _cb_tcp(self, msg: Float32MultiArray):
        with self._lock:
            self._latest_tcp = np.array(msg.data, dtype=np.float32)

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
        if self._shutting_down or self._task_complete or self._inference_running:
            return

        with self._lock:
            img = self._latest_img
            zed = self._latest_zed
            state7 = self._latest_state
            tcp6 = self._latest_tcp
            prompt = self._latest_prompt

        if img is None or state7 is None:
            return

        zed_frame = zed.copy() if zed is not None else np.zeros((*IMG_SIZE, 3), dtype=np.uint8)
        tcp = tcp6.copy() if tcp6 is not None else np.zeros(6, dtype=np.float32)

        # 13D state: [j1..j6, tx,ty,tz,rx,ry,rz, gripper]
        state13 = np.concatenate([state7[:6], tcp, [state7[6]]]).astype(np.float32)

        obs = {
            "img": img.copy(),
            "zed": zed_frame,
            "state13": state13,
            "prompt": prompt,
        }
        self._inference_running = True
        self._executor.submit(self._run_infer, obs)

    def _run_infer(self, obs: dict):
        if self._shutting_down:
            self._inference_running = False
            return
        try:
            img_b64 = _image_msg_to_b64png_from_array(obs["img"])
            zed_b64 = _image_msg_to_b64png_from_array(obs["zed"])
            payload = {
                "state": obs["state13"].tolist(),
                "image1_b64": img_b64,
                "image2_b64": zed_b64,
                "task": obs["prompt"],
            }
            resp = self._session.post(
                f"{self._url}/act", json=payload, timeout=HTTP_ACT_TIMEOUT_SEC,
            )
            resp.raise_for_status()
            data = resp.json()

            # actions: (n_steps, 13) → 7D: [Δj1..j6, suction_abs]
            actions13 = np.asarray(data["actions"], dtype=np.float32)  # (N, 13)
            if actions13.ndim == 1:
                actions13 = actions13.reshape(1, -1)

            actions7 = np.concatenate(
                [actions13[:, 0:6], actions13[:, 12:13]], axis=1
            )  # (N, 7)

            # executor가 ACTION_HORIZON*7 크기를 요구 — 1스텝이면 반복 패딩
            if actions7.shape[0] < ACTION_HORIZON:
                repeat = (ACTION_HORIZON + actions7.shape[0] - 1) // actions7.shape[0]
                actions7 = np.tile(actions7, (repeat, 1))[:ACTION_HORIZON]

            self._infer_call_count += 1
            self._infer_count_pub.publish(Int32(data=self._infer_call_count))

            grip_vals = [f"{actions7[i, 6]:+.3f}" for i in range(len(actions7))]
            self.get_logger().info(
                f"추론 완료 shape={actions13.shape}→{actions7.shape} "
                f"prompt={obs['prompt']!r}"
            )
            self.get_logger().info(
                f"state13:    {np.round(obs['state13'], 1).tolist()}"
            )
            self.get_logger().info(
                f"action7[0]: {np.round(actions7[0, :6], 1).tolist()}  "
                f"suction={actions7[0, 6]:+.3f}"
            )
            self.get_logger().info(f"suction_seq: {grip_vals}")

            msg = Float32MultiArray(data=actions7.flatten().tolist())
            self._chunk_pub.publish(msg)

        except Exception as exc:
            self.get_logger().error(f"추론 실패: {exc}")
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
    node = SmolVLABridgeNode()
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
