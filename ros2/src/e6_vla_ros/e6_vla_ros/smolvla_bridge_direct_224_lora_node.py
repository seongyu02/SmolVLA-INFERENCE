#!/usr/bin/env python3
"""
SmolVLA 7D direct inference bridge node (224 LoRA).

HTTP 서버 없이 Jetson GPU에서 SmolVLA 모델을 직접 로드·추론합니다.
camera_state_node를 pub_smolvla_images=True로 실행 시 발행되는
512×512 이미지 토픽을 구독합니다.

구독:
  /e6/camera/image_512     sensor_msgs/Image   512×512 RGB (HIK)
  /e6/camera/zed_image_512 sensor_msgs/Image   512×512 RGB (ZED)
  /e6/robot/state          Float32MultiArray   7D [j1..j6 deg, gripper]
  /e6/task/prompt          String
  /e6/task/status          String

발행:
  /e6/policy/action_chunk  Float32MultiArray   (16×7 flatten)
  /e6/inference/count      Int32
"""
from __future__ import annotations

import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import rclpy
import torch
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile
from sensor_msgs.msg import Image
from std_msgs.msg import Float32MultiArray, Int32, String

STATE_DIM = 7
ACTION_DIM = 7
ACTION_HORIZON = 16
DEFAULT_PROMPT = "pick up the orange box from the left side and place it on the right side"
DEFAULT_MODEL_PATH = (
    "/media/billye6/새 볼륨1/Dobot/SmolVLA_outputs_orange_v3_lora/"
    "smolvla_orange_v3_7d_vision_lora_v23_from_smolvla_base_dobot7d/"
    "checkpoints/020000/pretrained_model"
)
DEFAULT_BASE_MODEL_PATH = (
    "/media/billye6/새 볼륨1/Dobot/SmolVLA_base_dobot7d_local/pretrained_model"
)
SAFETENSORS_SINGLE_FILE = "model.safetensors"
INIT_POSE_J123 = np.array([91.3, 37.7, 53.8], dtype=np.float32)


class SmolVLABridgeDirect224LoraNode(Node):
    def __init__(self):
        super().__init__("smolvla_bridge_direct_224_lora_node")

        self.declare_parameter("model_path", DEFAULT_MODEL_PATH)
        self.declare_parameter("infer_hz", 1.25)
        self.declare_parameter("init_pose_tol_deg", 5.0)
        self.declare_parameter("wait_for_init_pose", True)

        model_path = self.get_parameter("model_path").value
        infer_hz = float(self.get_parameter("infer_hz").value)
        self._init_pose_tol = float(self.get_parameter("init_pose_tol_deg").value)
        self._wait_for_init_pose = bool(self.get_parameter("wait_for_init_pose").value)

        self._model_path = _resolve_model_path(model_path)

        self._latest_img: np.ndarray | None = None
        self._latest_zed: np.ndarray | None = None
        self._latest_state: np.ndarray | None = None
        self._latest_prompt: str = DEFAULT_PROMPT
        self._lock = threading.Lock()

        self._model = None
        self._preprocessor = None
        self._postprocessor = None
        self._device: torch.device | None = None
        self._n_action_steps: int = 10
        self._model_ready = False

        self._inference_running = False
        self._task_complete = False
        self._shutting_down = False
        self._infer_call_count = 0
        self._init_pose_armed = self._wait_for_init_pose
        self._executor = ThreadPoolExecutor(max_workers=1)

        qos_transient = QoSProfile(durability=DurabilityPolicy.TRANSIENT_LOCAL, depth=1)

        self.create_subscription(Image, "/e6/camera/image_512", self._cb_img, 10)
        self.create_subscription(Image, "/e6/camera/zed_image_512", self._cb_zed, 10)
        self.create_subscription(Float32MultiArray, "/e6/robot/state", self._cb_state, 10)
        self.create_subscription(String, "/e6/task/prompt", self._cb_prompt, qos_transient)
        self.create_subscription(String, "/e6/task/status", self._cb_task_status, 10)

        self._chunk_pub = self.create_publisher(Float32MultiArray, "/e6/policy/action_chunk", 10)
        self._infer_count_pub = self.create_publisher(Int32, "/e6/inference/count", 10)

        threading.Thread(target=self._load_model, daemon=True).start()

        self.create_timer(1.0 / infer_hz, self._maybe_infer)

        self.get_logger().info(
            f"[direct-224-lora] bridge started model={self._model_path} "
            f"state_dim={STATE_DIM} action_dim={ACTION_DIM} "
            f"infer_interval={1.0 / infer_hz:.2f}s"
        )

    # ── 모델 로드 ──────────────────────────────────────────────────────────────

    def _load_model(self):
        self.get_logger().info(f"[direct-224-lora] loading model: {self._model_path}")
        t0 = time.monotonic()
        try:
            from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
            from lerobot.processor import PolicyProcessorPipeline
            from lerobot.processor.converters import (
                batch_to_transition,
                policy_action_to_transition,
                transition_to_batch,
                transition_to_policy_action,
            )
            from lerobot.utils.constants import (
                POLICY_POSTPROCESSOR_DEFAULT_NAME,
                POLICY_PREPROCESSOR_DEFAULT_NAME,
            )

            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model_dir = Path(self._model_path)
            full_weights = model_dir / SAFETENSORS_SINGLE_FILE
            adapter_weights = model_dir / "adapter_model.safetensors"

            if full_weights.is_file():
                model = SmolVLAPolicy.from_pretrained(self._model_path)
                self.get_logger().info("[direct-224-lora] loaded full checkpoint")
            elif adapter_weights.is_file():
                from peft import PeftConfig, PeftModel

                peft_config = PeftConfig.from_pretrained(self._model_path)
                base_path = _resolve_base_model_path(peft_config.base_model_name_or_path)
                self.get_logger().info(
                    f"[direct-224-lora] loading base model: {base_path}"
                )
                self.get_logger().info(
                    f"[direct-224-lora] loading adapter: {self._model_path}"
                )
                base_model = SmolVLAPolicy.from_pretrained(base_path)
                model = PeftModel.from_pretrained(
                    base_model,
                    self._model_path,
                    config=peft_config,
                    is_trainable=False,
                )
                self.get_logger().info("[direct-224-lora] PEFT adapter loaded")
            else:
                raise FileNotFoundError(
                    f"No {SAFETENSORS_SINGLE_FILE} or adapter_model.safetensors in {self._model_path}"
                )

            model.to(device)
            model.eval()
            if hasattr(model, "reset"):
                model.reset()

            preprocessor = PolicyProcessorPipeline.from_pretrained(
                self._model_path,
                config_filename=f"{POLICY_PREPROCESSOR_DEFAULT_NAME}.json",
                to_transition=batch_to_transition,
                to_output=transition_to_batch,
            )
            postprocessor = PolicyProcessorPipeline.from_pretrained(
                self._model_path,
                config_filename=f"{POLICY_POSTPROCESSOR_DEFAULT_NAME}.json",
                to_transition=policy_action_to_transition,
                to_output=transition_to_policy_action,
            )

            self._device = device
            self._n_action_steps = int(model.config.n_action_steps)
            self._model = model
            self._preprocessor = preprocessor
            self._postprocessor = postprocessor
            self._model_ready = True

            elapsed = time.monotonic() - t0
            self.get_logger().info(
                f"[direct-224-lora] model ready on {device} "
                f"n_action_steps={self._n_action_steps} "
                f"resize_imgs={getattr(model.config, 'resize_imgs_with_padding', '?')} "
                f"({elapsed:.1f}s)"
            )
        except Exception as exc:
            self.get_logger().error(f"[direct-224-lora] model load failed: {exc}")

    # ── ROS2 콜백 ─────────────────────────────────────────────────────────────

    def _cb_img(self, msg: Image):
        with self._lock:
            self._latest_img = np.frombuffer(msg.data, dtype=np.uint8).reshape(
                msg.height, msg.width, 3
            ).copy()

    def _cb_zed(self, msg: Image):
        with self._lock:
            self._latest_zed = np.frombuffer(msg.data, dtype=np.uint8).reshape(
                msg.height, msg.width, 3
            ).copy()

    def _cb_state(self, msg: Float32MultiArray):
        with self._lock:
            self._latest_state = np.array(msg.data, dtype=np.float32)

    def _cb_prompt(self, msg: String):
        with self._lock:
            changed = self._latest_prompt != msg.data
            self._latest_prompt = msg.data
        if changed:
            self.get_logger().info(f"[direct-224-lora] prompt changed: {msg.data!r}")

    def _cb_task_status(self, msg: String):
        if not self._task_complete and (
            msg.data == "TASK_COMPLETE" or msg.data.startswith("FAIL_SAFETY")
        ):
            self._task_complete = True
            self.get_logger().info(f"[direct-224-lora] inference stopped ({msg.data})")

    # ── 추론 루프 ──────────────────────────────────────────────────────────────

    def _maybe_infer(self):
        if (
            not self._model_ready
            or self._shutting_down
            or self._task_complete
            or self._inference_running
        ):
            if not self._model_ready:
                self.get_logger().info(
                    "[direct-224-lora] waiting for model to load...",
                    throttle_duration_sec=5.0,
                )
            return

        with self._lock:
            img = self._latest_img
            zed = self._latest_zed
            state7 = self._latest_state
            prompt = self._latest_prompt

        if img is None or state7 is None:
            return

        obs_state = np.asarray(state7, dtype=np.float32).ravel()
        if obs_state.shape[0] != STATE_DIM:
            self.get_logger().warn(
                f"[direct-224-lora] state len={obs_state.shape[0]} expected {STATE_DIM}",
                throttle_duration_sec=5.0,
            )
            return

        if self._init_pose_armed:
            j_diff = float(np.abs(obs_state[:3] - INIT_POSE_J123).max())
            if j_diff > self._init_pose_tol:
                self.get_logger().info(
                    f"[direct-224-lora] waiting for init pose "
                    f"current={np.round(obs_state[:3], 1).tolist()} "
                    f"target={INIT_POSE_J123.tolist()} diff={j_diff:.1f}deg",
                    throttle_duration_sec=2.0,
                )
                return
            self._init_pose_armed = False
            self.get_logger().info(
                f"[direct-224-lora] init pose confirmed, starting inference "
                f"j1..j3={np.round(obs_state[:3], 1).tolist()}"
            )

        zed_frame = zed if zed is not None else np.zeros((512, 512, 3), dtype=np.uint8)

        obs = {
            "img": img.copy(),
            "zed": zed_frame.copy(),
            "state7": obs_state.copy(),
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

            img_t = _rgb_uint8_to_tensor(obs["img"], self._device)
            zed_t = _rgb_uint8_to_tensor(obs["zed"], self._device)
            state_t = torch.from_numpy(obs["state7"]).unsqueeze(0).to(self._device)

            batch = {
                "observation.images.OBS_IMAGE_1": img_t,
                "observation.images.OBS_IMAGE_2": zed_t,
                "observation.state": state_t,
                "task": obs["prompt"],
            }

            t0 = time.monotonic()
            observation = self._preprocessor(batch)
            preproc_ms = (time.monotonic() - t0) * 1000.0

            with torch.no_grad():
                t0 = time.monotonic()
                action_chunk = self._model.predict_action_chunk(observation)
                if action_chunk.ndim == 2:
                    action_chunk = action_chunk.unsqueeze(0)
                action_chunk = action_chunk[:, : self._n_action_steps, :]
                predict_ms = (time.monotonic() - t0) * 1000.0

                t0 = time.monotonic()
                steps = action_chunk.shape[1]
                processed = [self._postprocessor(action_chunk[:, i, :]) for i in range(steps)]
                actions_t = torch.stack(processed, dim=1).squeeze(0)
                postproc_ms = (time.monotonic() - t0) * 1000.0

            actions_np = actions_t.detach().float().cpu().numpy()  # (n_steps, 7)
            if actions_np.ndim == 1:
                actions_np = actions_np.reshape(1, -1)

            # executor_supervisor_node는 ACTION_HORIZON×ACTION_DIM을 기대함
            if actions_np.shape[0] < ACTION_HORIZON:
                repeat = (ACTION_HORIZON + actions_np.shape[0] - 1) // actions_np.shape[0]
                actions_np = np.tile(actions_np, (repeat, 1))[:ACTION_HORIZON]
            else:
                actions_np = actions_np[:ACTION_HORIZON]

            total_ms = (time.monotonic() - t_total0) * 1000.0

            self._infer_call_count += 1
            self._infer_count_pub.publish(Int32(data=self._infer_call_count))

            grip_vals = [f"{actions_np[i, 6]:+.3f}" for i in range(len(actions_np))]
            self.get_logger().info(
                f"[direct-224-lora] inference {total_ms:.0f}ms "
                f"(preproc={preproc_ms:.0f}ms predict={predict_ms:.0f}ms "
                f"postproc={postproc_ms:.0f}ms) shape={actions_np.shape}"
            )
            self.get_logger().info(
                f"[direct-224-lora] state7={np.round(obs['state7'], 1).tolist()}"
            )
            self.get_logger().info(
                f"[direct-224-lora] action_7={np.round(actions_np[0], 3).tolist()}"
            )
            self.get_logger().info(f"[direct-224-lora] suction_seq: {grip_vals}")

            self._chunk_pub.publish(Float32MultiArray(data=actions_np.flatten().tolist()))

        except Exception as exc:
            self.get_logger().error(f"[direct-224-lora] inference failed: {exc}")
        finally:
            self._inference_running = False

    # ── 종료 ──────────────────────────────────────────────────────────────────

    def destroy_node(self):
        self._shutting_down = True
        try:
            self._executor.shutdown(wait=False, cancel_futures=True)
        except TypeError:
            self._executor.shutdown(wait=False)
        super().destroy_node()


# ── 유틸 ──────────────────────────────────────────────────────────────────────

def _resolve_model_path(raw: str) -> str:
    path = Path(raw).expanduser()
    if (path / "pretrained_model").is_dir():
        path = path / "pretrained_model"
    return str(path)


def _resolve_base_model_path(base_path: str) -> str:
    candidates = [base_path]
    if "/media/billy/" in base_path:
        candidates.append(base_path.replace("/media/billy/", "/media/billye6/"))
    candidates.append(DEFAULT_BASE_MODEL_PATH)
    if "/pretrained_model" not in base_path:
        candidates.append(str(Path(base_path) / "pretrained_model"))

    for candidate in candidates:
        p = Path(candidate).expanduser()
        if (p / "pretrained_model").is_dir():
            p = p / "pretrained_model"
        if p.is_dir() and (p / SAFETENSORS_SINGLE_FILE).is_file():
            if str(p) != base_path:
                rclpy.logging.get_logger("smolvla_bridge_direct_224_lora_node").warning(
                    f"[direct-224-lora] base model path corrected: {base_path} -> {p}"
                )
            return str(p)

    raise FileNotFoundError(f"LoRA base model not found. Tried: {candidates}")


def _rgb_uint8_to_tensor(arr: np.ndarray, device: torch.device) -> torch.Tensor:
    """(H, W, 3) uint8 RGB → (1, 3, H, W) float32 [0, 1] on device."""
    t = torch.from_numpy(arr.astype(np.float32) / 255.0)
    return t.permute(2, 0, 1).unsqueeze(0).to(device)


def main(args=None):
    rclpy.init(args=args)
    node = SmolVLABridgeDirect224LoraNode()
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
