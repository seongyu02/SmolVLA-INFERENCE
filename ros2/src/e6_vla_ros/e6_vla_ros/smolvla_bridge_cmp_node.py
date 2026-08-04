#!/usr/bin/env python3
"""
SmolVLA "Where-collapse" 비교실험(cond A/B/C/D) 직접 추론 브릿지 노드.

검증된 smolvla_bridge_7d_lora_node.py를 그대로 복붙해 만들고, cmp 체크포인트가
PEFT 어댑터(adapter_model.safetensors)인 점만 반영해 **adapter 직접 로딩**을
추가했다. merge_lora.py로 만든 merged(model.safetensors) 디렉토리도 그대로 로드된다.

cmp 4조건 (train_cmp_condition.sh):
  A = frozen 256 (cond_A_frozen_256)   B = lora 256 (cond_B_lora_256)
  C = lora   512 (cond_C_lora_512)      D = frozen 512 (cond_D_frozen_512)
→ 추론 코드는 4조건 공통. 256/512 차이는 모델 config(resize_imgs_with_padding)가
  내부에서 처리하므로 이 노드의 이미지 파이프라인은 손대지 않는다.

== 추론 contract (4조건 공통) ==
  입력 : observation.images.OBS_IMAGE_1(HIK), OBS_IMAGE_2(ZED) CHW float[0,1] (IDENTITY)
         observation.state 7D [j1..j6, gripper]  (MEAN_STD, preprocessor가 처리)
         task  6-phase 프롬프트 문자열
  출력 : action chunk (16,7), postprocessor가 MEAN_STD로 unnormalize
         joint[0:6]=DELTA(mean≈0)  → 실행단: target = current + action[0:6]
         gripper[6]=ABSOLUTE(mean0.537) → 실행단: 0.5 임계 on/off (누산 금지)
  chunk=16 / n_action_steps=16, 16Hz, flow-matching num_steps=10 (모델 config)

이미지 입력:
  camera_state_node의 224×224 토픽(/e6/camera/image, /e6/camera/zed_image)을 구독.
  cmp 데이터셋(SmolVLA-Dataset-Orange-v4)은 이 224 crop을 256/512로 resize하여
  학습했으므로, 224 토픽을 받아 preprocessor(resize_imgs_with_padding=256 or 512)로
  키우면 학습과 일치한다. (512 카메라 토픽은 crop·회전이 달라 사용 안 함)

구독:
  /e6/camera/image         sensor_msgs/Image   224×224 RGB (HIK)
  /e6/camera/zed_image     sensor_msgs/Image   224×224 RGB (ZED)
  /e6/robot/state          Float32MultiArray   7D [j1..j6 deg, gripper]
  /e6/task/prompt          String              (per_frame_v16 phase prompt 권장)
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
SAFETENSORS_SINGLE_FILE = "model.safetensors"
# PEFT adapter의 base가 로컬에 없을 때 폴백으로 쓸 HF hub id (cmp 학습 base).
BASE_MODEL_HUB_ID = "lerobot/smolvla_base"
# v4 데이터셋은 per-frame phase prompt(6종)로 학습됨 → 기본값도 phase 텍스트.
DEFAULT_PROMPT = "approach the orange box on the left side"
# cmp 체크포인트 placeholder. launch model_path 인자로 덮어씀 (cond_A~D 또는 _merged).
# HDD(새 볼륨)의 Dobot 폴더 아래에 둘 예정. 젯슨 마운트는 billye6.
DEFAULT_MODEL_PATH = (
    "/media/billye6/새 볼륨1/Dobot/smolvla_cmp/cond_A_frozen_256/"
    "checkpoints/last/pretrained_model"
)
INIT_POSE_J123 = np.array([91.3, 37.7, 53.8], dtype=np.float32)


class SmolVLABridgeCmpNode(Node):
    def __init__(self):
        super().__init__("smolvla_bridge_cmp_node")

        self.declare_parameter("model_path", DEFAULT_MODEL_PATH)
        self.declare_parameter("infer_hz", 1.25)
        self.declare_parameter("init_pose_tol_deg", 5.0)
        self.declare_parameter("wait_for_init_pose", True)
        # 비어있지 않으면 매 추론마다 모델 입력(256) 이미지를 이 디렉토리 아래
        # <cond태그>_<타임스탬프>/ 폴더에 저장 (디버깅/검증용). 예: "/media/billye6/새 볼륨1/Dobot"
        self.declare_parameter("save_image_dir", "")

        model_path = self.get_parameter("model_path").value
        infer_hz = float(self.get_parameter("infer_hz").value)
        self._init_pose_tol = float(self.get_parameter("init_pose_tol_deg").value)
        self._wait_for_init_pose = bool(self.get_parameter("wait_for_init_pose").value)

        self._model_path = _resolve_model_path(model_path)
        self._save_dir = _setup_save_dir(
            self.get_parameter("save_image_dir").value, self._model_path, self
        )

        self._latest_img: np.ndarray | None = None
        self._latest_zed: np.ndarray | None = None
        self._latest_state: np.ndarray | None = None
        self._latest_prompt: str = DEFAULT_PROMPT
        self._lock = threading.Lock()

        self._model = None
        self._preprocessor = None
        self._postprocessor = None
        self._device: torch.device | None = None
        self._n_action_steps: int = 16
        self._save_resize: int = 256  # 모델 로드 후 config.resize_imgs_with_padding로 갱신
        self._model_ready = False

        self._inference_running = False
        self._task_complete = False
        self._shutting_down = False
        self._infer_call_count = 0
        self._init_pose_armed = self._wait_for_init_pose
        self._executor = ThreadPoolExecutor(max_workers=1)

        qos_transient = QoSProfile(durability=DurabilityPolicy.TRANSIENT_LOCAL, depth=1)

        self.create_subscription(Image, "/e6/camera/image", self._cb_img, 10)
        self.create_subscription(Image, "/e6/camera/zed_image", self._cb_zed, 10)
        self.create_subscription(Float32MultiArray, "/e6/robot/state", self._cb_state, 10)
        self.create_subscription(String, "/e6/task/prompt", self._cb_prompt, qos_transient)
        self.create_subscription(String, "/e6/task/status", self._cb_task_status, 10)

        self._chunk_pub = self.create_publisher(Float32MultiArray, "/e6/policy/action_chunk", 10)
        self._infer_count_pub = self.create_publisher(Int32, "/e6/inference/count", 10)

        threading.Thread(target=self._load_model, daemon=True).start()

        self.create_timer(1.0 / infer_hz, self._maybe_infer)

        self.get_logger().info(
            f"[cmp] bridge started model={self._model_path} "
            f"state_dim={STATE_DIM} action_dim={ACTION_DIM} "
            f"infer_interval={1.0 / infer_hz:.2f}s"
        )

    # ── 모델 로드 ──────────────────────────────────────────────────────────────

    def _load_model(self):
        self.get_logger().info(f"[cmp] loading model: {self._model_path}")
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
                # merge_lora.py로 만든 merged(full) 체크포인트
                model = SmolVLAPolicy.from_pretrained(self._model_path)
                self.get_logger().info("[cmp] loaded merged/full checkpoint (model.safetensors)")
            elif adapter_weights.is_file():
                # cmp 원본 PEFT 어댑터 → base + adapter로 직접 로드 (merge 불필요)
                from peft import PeftConfig, PeftModel
                from lerobot.configs.policies import PreTrainedConfig

                peft_config = PeftConfig.from_pretrained(self._model_path)
                base_path = _resolve_base_model_path(
                    peft_config.base_model_name_or_path, self
                )
                # ★ 핵심: base를 cmp 체크포인트의 config로 인스턴스화해야 한다.
                #   안 그러면 smolvla_base 기본 config(camera1/2/3, chunk50, 512)가
                #   적용돼 입력 키/chunk/resize가 cmp(OBS_IMAGE_1/2, 16, 256)와 불일치.
                #   SmolVLA는 state/action을 max_dim으로 패딩하므로 base 가중치는
                #   shape 호환 → cmp config로 빌드 후 base 가중치를 로드해도 정상.
                #   (SmolVLAConfig.from_pretrained은 config.json의 'type' 필드에서
                #    draccus 에러 → PreTrainedConfig.from_pretrained로 choice 디코드)
                cmp_cfg = PreTrainedConfig.from_pretrained(self._model_path)
                self.get_logger().info(
                    f"[cmp LoRA] cmp config: chunk={cmp_cfg.chunk_size} "
                    f"n_action_steps={cmp_cfg.n_action_steps} "
                    f"resize={cmp_cfg.resize_imgs_with_padding} "
                    f"inputs={list(cmp_cfg.input_features.keys())}"
                )
                self.get_logger().info(f"[cmp LoRA] loading base: {base_path} (with cmp config)")
                self.get_logger().info(f"[cmp LoRA] loading adapter: {self._model_path}")
                base_model = SmolVLAPolicy.from_pretrained(
                    base_path, config=cmp_cfg, strict=False
                )
                model = PeftModel.from_pretrained(
                    base_model,
                    self._model_path,
                    config=peft_config,
                    is_trainable=False,
                )
                self.get_logger().info("[cmp LoRA] PEFT adapter loaded (merge 없이 직접 서빙)")
            else:
                raise FileNotFoundError(
                    f"No {SAFETENSORS_SINGLE_FILE} or adapter_model.safetensors "
                    f"in {self._model_path}"
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
            _rp = getattr(model.config, "resize_imgs_with_padding", None)
            if _rp:
                self._save_resize = int(_rp[0])
            self._model = model
            self._preprocessor = preprocessor
            self._postprocessor = postprocessor
            self._model_ready = True

            elapsed = time.monotonic() - t0
            self.get_logger().info(
                f"[cmp] model ready on {device} "
                f"n_action_steps={self._n_action_steps} "
                f"chunk_size={getattr(model.config, 'chunk_size', '?')} "
                f"resize_imgs={getattr(model.config, 'resize_imgs_with_padding', '?')} "
                f"({elapsed:.1f}s)"
            )
        except Exception as exc:
            self.get_logger().error(f"[cmp] model load failed: {exc}")

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
            self.get_logger().info(f"[cmp] prompt changed: {msg.data!r}")

    def _cb_task_status(self, msg: String):
        if not self._task_complete and (
            msg.data == "TASK_COMPLETE" or msg.data.startswith("FAIL_SAFETY")
        ):
            self._task_complete = True
            self.get_logger().info(f"[cmp] inference stopped ({msg.data})")

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
                    "[cmp] waiting for model to load...",
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
                f"[cmp] state len={obs_state.shape[0]} expected {STATE_DIM}",
                throttle_duration_sec=5.0,
            )
            return

        if self._init_pose_armed:
            j_diff = float(np.abs(obs_state[:3] - INIT_POSE_J123).max())
            if j_diff > self._init_pose_tol:
                self.get_logger().info(
                    f"[cmp] waiting for init pose "
                    f"current={np.round(obs_state[:3], 1).tolist()} "
                    f"target={INIT_POSE_J123.tolist()} diff={j_diff:.1f}deg",
                    throttle_duration_sec=2.0,
                )
                return
            self._init_pose_armed = False
            self.get_logger().info(
                f"[cmp] init pose confirmed, starting inference "
                f"j1..j3={np.round(obs_state[:3], 1).tolist()}"
            )

        zed_frame = zed if zed is not None else np.zeros((224, 224, 3), dtype=np.uint8)

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
                f"[cmp] inference {total_ms:.0f}ms "
                f"(preproc={preproc_ms:.0f}ms predict={predict_ms:.0f}ms "
                f"postproc={postproc_ms:.0f}ms) shape={actions_np.shape}"
            )
            self.get_logger().info(
                f"[cmp] state7={np.round(obs['state7'], 1).tolist()}"
            )
            self.get_logger().info(
                f"[cmp] action_7={np.round(actions_np[0], 3).tolist()}"
            )
            self.get_logger().info(f"[cmp] suction_seq(abs): {grip_vals}")

            self._chunk_pub.publish(Float32MultiArray(data=actions_np.flatten().tolist()))

            if self._save_dir is not None:
                _save_input_images(
                    self._save_dir, self._infer_call_count, obs,
                    actions_np[0], self._save_resize, self
                )

        except Exception as exc:
            self.get_logger().error(f"[cmp] inference failed: {exc}")
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


def _resolve_base_model_path(base_path: str, node: Node) -> str:
    """PEFT adapter config가 가리키는 base 경로를 이 머신 기준으로 해석.

    로컬 디렉토리(model.safetensors 포함)면 그대로/마운트 보정 후 사용,
    아니면 HF hub id(lerobot/smolvla_base)로 간주해 그대로 넘긴다.
    """
    candidates = [base_path]
    if base_path and "/media/billy/" in base_path:
        candidates.append(base_path.replace("/media/billy/", "/media/billye6/"))
    for path in candidates:
        if Path(path).is_dir() and (Path(path) / SAFETENSORS_SINGLE_FILE).is_file():
            if path != base_path:
                node.get_logger().warn(
                    f"[cmp LoRA] base path corrected: {base_path} -> {path}"
                )
            return path
    resolved = base_path or BASE_MODEL_HUB_ID
    node.get_logger().info(
        f"[cmp LoRA] base not local, treating as hub id: {resolved}"
    )
    return resolved


def _rgb_uint8_to_tensor(arr: np.ndarray, device: torch.device) -> torch.Tensor:
    """(H, W, 3) uint8 RGB → (1, 3, H, W) float32 [0, 1] on device."""
    t = torch.from_numpy(arr.astype(np.float32) / 255.0)
    return t.permute(2, 0, 1).unsqueeze(0).to(device)


def _cond_tag(model_path: str) -> str:
    """model_path에서 cmp 조건 태그(cond_B_lora_256 등) 추출. 없으면 'cmp'."""
    for part in Path(model_path).parts:
        if part.startswith("cond_"):
            return part
    return "cmp"


def _setup_save_dir(raw_dir, model_path: str, node: Node):
    """save_image_dir가 주어지면 <raw_dir>/<cond태그>_<타임스탬프>/ 생성 후 반환."""
    raw_dir = (raw_dir or "").strip()
    if not raw_dir or raw_dir.lower() in ("none", "off"):
        return None
    session = time.strftime("%Y%m%d_%H%M%S")
    out = Path(raw_dir).expanduser() / f"{_cond_tag(model_path)}_{session}"
    try:
        out.mkdir(parents=True, exist_ok=True)
        node.get_logger().info(f"[cmp] 입력 이미지 저장 활성화 → {out}")
        return out
    except Exception as exc:
        node.get_logger().error(f"[cmp] save_image_dir 생성 실패({exc}) → 저장 비활성화")
        return None


def _save_input_images(save_dir: Path, count: int, obs: dict, action0, size: int, node: Node):
    """모델 입력(size×size, resize_with_pad 동일) HIK/ZED PNG + meta.csv 한 줄 저장."""
    try:
        import cv2

        for name, arr in (("hik", obs["img"]), ("zed", obs["zed"])):
            # 모델 내부와 동일하게 정사각 224 → size resize (정사각이라 padding 불필요)
            img = cv2.resize(arr, (size, size), interpolation=cv2.INTER_LINEAR)
            cv2.imwrite(
                str(save_dir / f"{count:05d}_{name}_{size}.png"),
                cv2.cvtColor(img, cv2.COLOR_RGB2BGR),
            )
        meta = save_dir / "meta.csv"
        if not meta.exists():
            meta.write_text("count,prompt,state7,action0_7\n")
        with meta.open("a") as f:
            st = ",".join(f"{v:.3f}" for v in np.asarray(obs["state7"]).ravel())
            ac = ",".join(f"{v:.4f}" for v in np.asarray(action0).ravel())
            f.write(f'{count},"{obs["prompt"]}","{st}","{ac}"\n')
    except Exception as exc:
        node.get_logger().warn(
            f"[cmp] 이미지 저장 실패: {exc}", throttle_duration_sec=5.0
        )


def main(args=None):
    rclpy.init(args=args)
    node = SmolVLABridgeCmpNode()
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
