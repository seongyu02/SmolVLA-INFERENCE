#!/usr/bin/env python3
"""
SmolVLA 5-condition(Orange-512) 직접 추론 브릿지 노드.

검증된 smolvla_bridge_cmp_node.py를 베이스로, 5cond 실험(smolvla_5cond)의
계약에 맞춰 두 가지만 바꿨다.
  1) 이미지 입력: camera_state_node가 발행하는 **512 LANCZOS** 토픽을 구독
     (/e6/smolvla5/image_hik_512 → OBS_IMAGE_1, /e6/smolvla5/image_zed_512 → OBS_IMAGE_2).
     학습 데이터(Convert_SmolVLADataset_Orange_512.py)가 224 crop → LANCZOS 512 였으므로,
     동일하게 512 LANCZOS를 그대로 먹이면 모델 내부 resize_with_pad(512)는 no-op → 학습=추론 일치.
     (cmp 노드는 224를 먹여 모델 내부 bilinear 512로 키웠음 → 5cond와 filter 불일치라 분리)
  2) base 경로: 5cond adapter_config의 base는 학습서버 경로(/media/billy/새 볼륨4/...)라
     이 젯슨에서 자동 치환이 안 맞음. base_model_path 파라미터로 로컬 base를 명시한다.

5cond 5개 조건은 model_path(체크포인트)와 prompt(task)만 다르고 추론 코드는 공통.
  exp1 expert_only / exp2 vision_lora_0_10 / exp3 vision_lora_6_10
  exp4 nolr_vision_lora_0_10 / exp5 nolr_vision_lora_6_10
→ launch에서 model_path(또는 cond:=1~5)로 선택. 이 노드는 손대지 않는다.

== 추론 contract (5조건 공통) ==
  입력 : observation.images.OBS_IMAGE_1(HIK 512), OBS_IMAGE_2(ZED 512) CHW float[0,1] (IDENTITY)
         observation.state 7D [j1..j6 deg, gripper] (MEAN_STD, preprocessor가 처리)
         task  단일 프롬프트 문자열 (방향 포함=512 / 방향 제거=512-nolr)
  출력 : action chunk (16,7), postprocessor가 MEAN_STD로 unnormalize
         joint[0:6]=DELTA(mean≈0)   → 실행단: target = current + action[0:6]
         gripper[6]=ABSOLUTE        → 실행단: 0.5 임계 on/off (누산 금지)
  chunk=16 / n_action_steps=16, 16Hz, flow-matching num_steps(모델 config)

구독:
  /e6/smolvla5/image_hik_512  sensor_msgs/Image   512×512 RGB (HIK → OBS_IMAGE_1)
  /e6/smolvla5/image_zed_512  sensor_msgs/Image   512×512 RGB (ZED → OBS_IMAGE_2)
  /e6/robot/state             Float32MultiArray   7D [j1..j6 deg, gripper]
  /e6/task/prompt             String              단일 프롬프트
  /e6/task/status             String
발행:
  /e6/policy/action_chunk     Float32MultiArray   (16×7 flatten)
  /e6/inference/count         Int32
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import psutil
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
# 5cond adapter의 base가 가리키는 학습서버 경로가 안 맞을 때 쓸 로컬 base (이 젯슨).
# 이 로컬 base(우리 수정 config)를 못 쓰면 HF hub로 폴백하지 않고 추론을 중단한다.
DEFAULT_BASE_MODEL_PATH = (
    "/media/billye6/새 볼륨1/SmolVLA/SmolVLA_base_dobot7d_local/pretrained_model"
)
# 단일 프롬프트로 학습됨 → 기본값도 방향 포함 단일 문장(launch prompt_text로 덮어씀).
DEFAULT_PROMPT = "pick up the orange box from the left side and place it on the right side"
# 5cond 체크포인트 placeholder. launch model_path(또는 cond)로 덮어씀.
DEFAULT_MODEL_PATH = (
    "/media/billye6/새 볼륨1/SmolVLA/smolvla_5cond/exp3_vision_lora_6_10/"
    "checkpoints/020000/pretrained_model"
)
INIT_POSE_J123 = np.array([91.3, 37.7, 53.8], dtype=np.float32)


class _ResourceProfiler:
    """추론 N회의 자원 사용량(GPU/CPU/RAM %)을 재서 평균 JSON(+PNG) 으로 저장.

    - GPU : tegrastats GR3D_FREQ (Jetson GPU utilization %)
    - CPU : psutil.cpu_percent (시스템 전체 0~100%)
    - RAM : psutil.virtual_memory().percent (시스템 전체 0~100%)
    백그라운드로 ~sample_interval 마다 샘플링하고, _run_infer 가 끝날 때마다
    record() 로 그 추론 구간 [t0,t1] 의 샘플 평균을 1회분으로 적립한다.
    runs 회가 모이면 평균/표준편차를 JSON 으로 저장하고 PNG 그래프를 띄운다.
    """

    _GR3D = re.compile(r"GR3D_FREQ\s+(\d+)%")

    def __init__(self, node, runs, warmup, out_path,
                 sample_interval=0.05, tegra_interval_ms=100):
        self._node = node
        self.runs = int(runs)
        self.warmup = int(warmup)
        self.out_path = out_path
        self.sample_interval = float(sample_interval)
        self.gpu = 0.0
        self.samples = []   # (t, gpu, cpu, ram)
        self.per_run = []   # 추론 1회당 평균 dict
        self._lock = threading.Lock()
        self._stop = False
        self.done = False
        self._tproc = subprocess.Popen(
            ["tegrastats", "--interval", str(int(tegra_interval_ms))],
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True, bufsize=1,
        )
        threading.Thread(target=self._read_tegra, daemon=True).start()
        threading.Thread(target=self._sample, daemon=True).start()

    def _read_tegra(self):
        for line in self._tproc.stdout:
            if self._stop:
                break
            m = self._GR3D.search(line)
            if m:
                self.gpu = float(m.group(1))

    def _sample(self):
        psutil.cpu_percent(None)  # cpu_percent prime (첫 호출은 0)
        time.sleep(self.sample_interval)
        while not self._stop:
            t = time.monotonic()
            cpu = psutil.cpu_percent(None)
            ram = psutil.virtual_memory().percent
            with self._lock:
                self.samples.append((t, self.gpu, cpu, ram))
            time.sleep(self.sample_interval)

    def record(self, call_count, t0, t1, latency_ms):
        """_run_infer 종료 시 호출. call_count = 1-based 추론 인덱스."""
        if self.done or call_count <= self.warmup:
            return
        with self._lock:
            inside = [s for s in self.samples if t0 <= s[0] <= t1]
            if not inside and self.samples:
                inside = [min(self.samples, key=lambda s: abs(s[0] - t1))]
        if not inside:
            return
        g = sum(s[1] for s in inside) / len(inside)
        c = sum(s[2] for s in inside) / len(inside)
        r = sum(s[3] for s in inside) / len(inside)
        self.per_run.append({
            "gpu": g, "cpu": c, "ram": r,
            "latency_ms": latency_ms, "n_samples": len(inside),
        })
        self._node.get_logger().info(
            f"[profile] {len(self.per_run)}/{self.runs}  "
            f"GPU={g:.1f}% CPU={c:.1f}% RAM={r:.1f}%  ({latency_ms:.0f}ms)"
        )
        if len(self.per_run) >= self.runs:
            self._finalize()

    def _finalize(self):
        if self.done or not self.per_run:
            return
        self.done = True
        self._stop = True
        try:
            self._tproc.terminate()
        except Exception:
            pass

        def stat(k):
            a = np.array([x[k] for x in self.per_run], dtype=float)
            return float(a.mean()), float(a.std())

        means = {k: stat(k)[0] for k in ("gpu", "cpu", "ram")}
        stds = {k: stat(k)[1] for k in ("gpu", "cpu", "ram")}
        lat_m, lat_s = stat("latency_ms")
        result = {
            "meta": {
                "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
                "cond": getattr(self._node, "_profile_cond", "?"),
                "model_path": getattr(self._node, "_model_path", ""),
                "runs": len(self.per_run), "warmup": self.warmup,
                "total_ram_mb": round(psutil.virtual_memory().total / (1024 ** 2), 1),
                "n_cpu": psutil.cpu_count(),
                "source": "live_node",
            },
            "means": means, "stds": stds,
            "latency_ms": {"mean": lat_m, "std": lat_s},
            "per_run": self.per_run,
        }
        try:
            with open(self.out_path, "w") as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
        except Exception as e:
            self._node.get_logger().error(f"[profile] JSON 저장 실패: {e}")
            return
        self._node.get_logger().info(
            f"[profile] DONE {len(self.per_run)}회 평균 → "
            f"GPU={means['gpu']:.1f}% CPU={means['cpu']:.1f}% RAM={means['ram']:.1f}%"
        )
        self._node.get_logger().info(f"[profile] JSON → {self.out_path}")
        png = os.path.splitext(self.out_path)[0] + ".png"
        plot_script = os.path.expanduser(
            "~/SmolVLA/SmolVLA-INFERENCE/scripts/plot_infer_resource.py"
        )
        # 시스템 python3(matplotlib)으로 그림. venv PYTHONPATH/CUDA 를 물려주면
        # numpy 버전 충돌로 matplotlib import 가 깨지므로 깨끗한 env 로 띄운다.
        clean_env = os.environ.copy()
        for k in ("PYTHONPATH", "PYTHONNOUSERSITE", "LD_LIBRARY_PATH", "PYTHONHOME"):
            clean_env.pop(k, None)
        try:
            subprocess.Popen(
                ["/usr/bin/python3", plot_script, "--json", self.out_path, "--out", png],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, env=clean_env,
            )
            self._node.get_logger().info(f"[profile] PNG → {png} (그래프 생성 중)")
        except Exception as e:
            self._node.get_logger().warn(
                f"[profile] 자동 그래프 실패({e}). 수동: /usr/bin/python3 "
                f"{plot_script} --json {self.out_path} --out {png}"
            )

    def shutdown(self):
        # runs 에 못 미쳐 종료돼도 모인 만큼은 저장
        if not self.done and self.per_run:
            self._finalize()
        self._stop = True
        try:
            self._tproc.terminate()
        except Exception:
            pass


class SmolVLABridge5CondNode(Node):
    def __init__(self):
        super().__init__("smolvla_bridge_5cond_node")

        self.declare_parameter("model_path", DEFAULT_MODEL_PATH)
        self.declare_parameter("base_model_path", DEFAULT_BASE_MODEL_PATH)
        self.declare_parameter("infer_hz", 1.25)
        self.declare_parameter("init_pose_tol_deg", 5.0)
        self.declare_parameter("wait_for_init_pose", True)
        # 비어있지 않으면 매 추론마다 모델 입력 이미지를 이 디렉토리 아래
        # <exp태그>_<타임스탬프>/ 폴더에 저장 (디버깅/검증용).
        self.declare_parameter("save_image_dir", "")
        # 추론 자원 프로파일링: profile_resource=True 면 처음 profile_runs 회 추론의
        # GPU/CPU/RAM(%) 평균을 JSON(+PNG) 으로 저장. profile_warmup 만큼은 제외.
        self.declare_parameter("profile_resource", False)
        self.declare_parameter("profile_runs", 10)
        self.declare_parameter("profile_warmup", 0)
        self.declare_parameter("profile_out", "auto")

        model_path = self.get_parameter("model_path").value
        base_model_path = self.get_parameter("base_model_path").value
        infer_hz = float(self.get_parameter("infer_hz").value)
        self._init_pose_tol = float(self.get_parameter("init_pose_tol_deg").value)
        self._wait_for_init_pose = bool(self.get_parameter("wait_for_init_pose").value)

        self._model_path = _resolve_model_path(model_path)
        self._base_override = (base_model_path or "").strip()
        self._save_dir = _setup_save_dir(
            self.get_parameter("save_image_dir").value, self._model_path, self
        )

        # ── 자원 프로파일러 (옵션) ───────────────────────────────────────────
        self._profile = None
        if bool(self.get_parameter("profile_resource").value):
            p_runs = int(self.get_parameter("profile_runs").value)
            p_warm = int(self.get_parameter("profile_warmup").value)
            p_out = (self.get_parameter("profile_out").value or "").strip()
            if not p_out or p_out == "auto":
                p_out = os.path.expanduser(
                    "~/SmolVLA/SmolVLA-INFERENCE/scripts/infer_resource_live.json"
                )
            m = re.search(r"exp[1-5]", self._model_path)
            self._profile_cond = m.group(0) if m else "?"
            self._profile = _ResourceProfiler(self, p_runs, p_warm, p_out)
            self.get_logger().info(
                f"[profile] enabled: 처음 {p_runs}회 추론 평균 "
                f"(warmup {p_warm}회 제외) → {p_out}"
            )

        self._latest_hik: np.ndarray | None = None
        self._latest_zed: np.ndarray | None = None
        self._latest_state: np.ndarray | None = None
        self._latest_prompt: str = DEFAULT_PROMPT
        self._lock = threading.Lock()

        self._model = None
        self._preprocessor = None
        self._postprocessor = None
        self._device: torch.device | None = None
        self._n_action_steps: int = 16
        self._save_resize: int = 512  # 모델 로드 후 config.resize_imgs_with_padding로 갱신
        self._model_ready = False
        self._load_failed = False
        self._load_error = ""

        self._inference_running = False
        self._task_complete = False
        self._shutting_down = False
        self._infer_call_count = 0
        self._init_pose_armed = self._wait_for_init_pose
        self._executor = ThreadPoolExecutor(max_workers=1)

        qos_transient = QoSProfile(durability=DurabilityPolicy.TRANSIENT_LOCAL, depth=1)

        self.create_subscription(Image, "/e6/smolvla5/image_hik_512", self._cb_hik, 10)
        self.create_subscription(Image, "/e6/smolvla5/image_zed_512", self._cb_zed, 10)
        self.create_subscription(Float32MultiArray, "/e6/robot/state", self._cb_state, 10)
        self.create_subscription(String, "/e6/task/prompt", self._cb_prompt, qos_transient)
        self.create_subscription(String, "/e6/task/status", self._cb_task_status, 10)

        self._chunk_pub = self.create_publisher(Float32MultiArray, "/e6/policy/action_chunk", 10)
        self._infer_count_pub = self.create_publisher(Int32, "/e6/inference/count", 10)

        threading.Thread(target=self._load_model, daemon=True).start()

        self.create_timer(1.0 / infer_hz, self._maybe_infer)

        self.get_logger().info(
            f"[5cond] bridge started model={self._model_path} "
            f"base={self._base_override or '(adapter_config)'} "
            f"state_dim={STATE_DIM} action_dim={ACTION_DIM} "
            f"infer_interval={1.0 / infer_hz:.2f}s"
        )

    # ── 모델 로드 ──────────────────────────────────────────────────────────────

    def _load_model(self):
        self.get_logger().info(f"[5cond] loading model: {self._model_path}")
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
                self.get_logger().info("[5cond] loaded merged/full checkpoint (model.safetensors)")
            elif adapter_weights.is_file():
                # 5cond 원본 PEFT 어댑터 → base + adapter로 직접 로드 (merge 불필요)
                from peft import PeftConfig, PeftModel
                from lerobot.configs.policies import PreTrainedConfig

                peft_config = PeftConfig.from_pretrained(self._model_path)
                base_path = _resolve_base_model_path(
                    self._base_override, peft_config.base_model_name_or_path, self
                )
                # ★ base를 5cond 체크포인트의 config로 인스턴스화해야 입력 키/chunk/resize가 일치.
                #   (OBS_IMAGE_1/2, chunk16, resize 512) — smolvla_base 기본 config가 적용되면 불일치.
                cmp_cfg = PreTrainedConfig.from_pretrained(self._model_path)
                self.get_logger().info(
                    f"[5cond] config: chunk={cmp_cfg.chunk_size} "
                    f"n_action_steps={cmp_cfg.n_action_steps} "
                    f"resize={cmp_cfg.resize_imgs_with_padding} "
                    f"inputs={list(cmp_cfg.input_features.keys())}"
                )
                self.get_logger().info(f"[5cond] loading base: {base_path} (with 5cond config)")
                self.get_logger().info(f"[5cond] loading adapter: {self._model_path}")
                base_model = SmolVLAPolicy.from_pretrained(
                    base_path, config=cmp_cfg, strict=False
                )
                model = PeftModel.from_pretrained(
                    base_model,
                    self._model_path,
                    config=peft_config,
                    is_trainable=False,
                )
                self.get_logger().info("[5cond] PEFT adapter loaded (merge 없이 직접 서빙)")
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
                f"[5cond] model ready on {device} "
                f"n_action_steps={self._n_action_steps} "
                f"chunk_size={getattr(model.config, 'chunk_size', '?')} "
                f"resize_imgs={getattr(model.config, 'resize_imgs_with_padding', '?')} "
                f"({elapsed:.1f}s)"
            )
        except Exception as exc:
            self._load_failed = True
            self._load_error = str(exc)
            self.get_logger().error(f"[5cond] model load failed: {exc}")

    # ── ROS2 콜백 ─────────────────────────────────────────────────────────────

    def _cb_hik(self, msg: Image):
        with self._lock:
            self._latest_hik = np.frombuffer(msg.data, dtype=np.uint8).reshape(
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
            self.get_logger().info(f"[5cond] prompt changed: {msg.data!r}")

    def _cb_task_status(self, msg: String):
        if not self._task_complete and (
            msg.data == "TASK_COMPLETE" or msg.data.startswith("FAIL_SAFETY")
        ):
            self._task_complete = True
            self.get_logger().info(f"[5cond] inference stopped ({msg.data})")

    # ── 추론 루프 ──────────────────────────────────────────────────────────────

    def _maybe_infer(self):
        if (
            not self._model_ready
            or self._shutting_down
            or self._task_complete
            or self._inference_running
        ):
            if self._load_failed:
                self.get_logger().error(
                    f"[5cond] 모델 로드 실패로 추론하지 않습니다 → {self._load_error}",
                    throttle_duration_sec=5.0,
                )
            elif not self._model_ready:
                self.get_logger().info(
                    "[5cond] waiting for model to load...",
                    throttle_duration_sec=5.0,
                )
            return

        with self._lock:
            hik = self._latest_hik
            zed = self._latest_zed
            state7 = self._latest_state
            prompt = self._latest_prompt

        if hik is None or state7 is None:
            return

        obs_state = np.asarray(state7, dtype=np.float32).ravel()
        if obs_state.shape[0] != STATE_DIM:
            self.get_logger().warn(
                f"[5cond] state len={obs_state.shape[0]} expected {STATE_DIM}",
                throttle_duration_sec=5.0,
            )
            return

        if self._init_pose_armed:
            j_diff = float(np.abs(obs_state[:3] - INIT_POSE_J123).max())
            if j_diff > self._init_pose_tol:
                self.get_logger().info(
                    f"[5cond] waiting for init pose "
                    f"current={np.round(obs_state[:3], 1).tolist()} "
                    f"target={INIT_POSE_J123.tolist()} diff={j_diff:.1f}deg",
                    throttle_duration_sec=2.0,
                )
                return
            self._init_pose_armed = False
            self.get_logger().info(
                f"[5cond] init pose confirmed, starting inference "
                f"j1..j3={np.round(obs_state[:3], 1).tolist()}"
            )

        zed_frame = zed if zed is not None else np.zeros_like(hik)

        obs = {
            "hik": hik.copy(),
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

            hik_t = _rgb_uint8_to_tensor(obs["hik"], self._device)
            zed_t = _rgb_uint8_to_tensor(obs["zed"], self._device)
            state_t = torch.from_numpy(obs["state7"]).unsqueeze(0).to(self._device)

            batch = {
                "observation.images.OBS_IMAGE_1": hik_t,
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

            t_end = time.monotonic()
            total_ms = (t_end - t_total0) * 1000.0

            self._infer_call_count += 1
            self._infer_count_pub.publish(Int32(data=self._infer_call_count))

            if self._profile is not None and not self._profile.done:
                self._profile.record(self._infer_call_count, t_total0, t_end, total_ms)

            grip_vals = [f"{actions_np[i, 6]:+.3f}" for i in range(len(actions_np))]
            self.get_logger().info(
                f"[5cond] inference {total_ms:.0f}ms "
                f"(preproc={preproc_ms:.0f}ms predict={predict_ms:.0f}ms "
                f"postproc={postproc_ms:.0f}ms) shape={actions_np.shape}"
            )
            self.get_logger().info(
                f"[5cond] state7={np.round(obs['state7'], 1).tolist()}"
            )
            self.get_logger().info(
                f"[5cond] action_7={np.round(actions_np[0], 3).tolist()}"
            )
            self.get_logger().info(f"[5cond] suction_seq(abs): {grip_vals}")

            self._chunk_pub.publish(Float32MultiArray(data=actions_np.flatten().tolist()))

            if self._save_dir is not None:
                _save_input_images(
                    self._save_dir, self._infer_call_count, obs,
                    actions_np[0], self._save_resize, self
                )

        except Exception as exc:
            self.get_logger().error(f"[5cond] inference failed: {exc}")
        finally:
            self._inference_running = False

    # ── 종료 ──────────────────────────────────────────────────────────────────

    def destroy_node(self):
        self._shutting_down = True
        if getattr(self, "_profile", None) is not None:
            self._profile.shutdown()
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


def _resolve_base_model_path(override: str, adapter_base: str, node: Node) -> str:
    """로컬 dobot7d base(= 우리가 수정한 config의 base)만 허용. 우선순위:
      1) base_model_path 파라미터(override)가 유효한 로컬 base
      2) adapter_config의 base 경로(그대로 / billy→billye6 치환)

    ★ 학습 스크립트(train_5cond_*.sh)가 로컬 base 없으면 exit 4로 막았던 것과 동일하게,
      둘 다 실패하면 **HF hub(lerobot/smolvla_base) 등으로 폴백하지 않고 즉시 실패**한다.
      generic SmolVLA base에 LoRA 어댑터를 얹으면 base 가중치 불일치로 추론이 망가지므로,
      우리 config 수정 base를 못 쓰면 아예 추론을 진행하지 않는다.
    """
    def _is_local_base(p: str) -> bool:
        if not p:
            return False
        d = Path(p)
        # 학습 스크립트와 동일 기준: config.json + model.safetensors 둘 다 필요
        return d.is_dir() and (d / "config.json").is_file() and (d / SAFETENSORS_SINGLE_FILE).is_file()

    if _is_local_base(override):
        node.get_logger().info(f"[5cond] base from param: {override}")
        return override

    candidates = []
    if adapter_base:
        candidates.append(adapter_base)
        if "/media/billy/" in adapter_base:
            candidates.append(adapter_base.replace("/media/billy/", "/media/billye6/"))
    for path in candidates:
        if _is_local_base(path):
            if path != adapter_base:
                node.get_logger().warn(
                    f"[5cond] base path corrected: {adapter_base} -> {path}"
                )
            return path

    raise FileNotFoundError(
        "[5cond][FATAL] 로컬 dobot7d base(우리 수정 config)를 찾지 못했습니다 — 추론 중단. "
        "config.json + model.safetensors 둘 다 있는 로컬 base가 필요하며, "
        "lerobot/smolvla_base 등 HF hub 폴백은 금지합니다(base 가중치 불일치). "
        f"확인한 경로: param base_model_path={override!r}, adapter_config base={adapter_base!r}"
    )


def _rgb_uint8_to_tensor(arr: np.ndarray, device: torch.device) -> torch.Tensor:
    """(H, W, 3) uint8 RGB → (1, 3, H, W) float32 [0, 1] on device."""
    t = torch.from_numpy(arr.astype(np.float32) / 255.0)
    return t.permute(2, 0, 1).unsqueeze(0).to(device)


def _exp_tag(model_path: str) -> str:
    """model_path에서 5cond 실험 태그(exp3_vision_lora_6_10 등) 추출. 없으면 '5cond'."""
    for part in Path(model_path).parts:
        if part.startswith("exp"):
            return part
    return "5cond"


def _setup_save_dir(raw_dir, model_path: str, node: Node):
    """save_image_dir가 주어지면 <raw_dir>/<exp태그>_<타임스탬프>/ 생성 후 반환."""
    raw_dir = (raw_dir or "").strip()
    if not raw_dir or raw_dir.lower() in ("none", "off"):
        return None
    session = time.strftime("%Y%m%d_%H%M%S")
    out = Path(raw_dir).expanduser() / f"{_exp_tag(model_path)}_{session}"
    try:
        out.mkdir(parents=True, exist_ok=True)
        node.get_logger().info(f"[5cond] 입력 이미지 저장 활성화 → {out}")
        return out
    except Exception as exc:
        node.get_logger().error(f"[5cond] save_image_dir 생성 실패({exc}) → 저장 비활성화")
        return None


def _save_input_images(save_dir: Path, count: int, obs: dict, action0, size: int, node: Node):
    """모델 입력(이미 512 LANCZOS) HIK/ZED PNG + meta.csv 한 줄 저장."""
    try:
        import cv2

        for name, arr in (("hik", obs["hik"]), ("zed", obs["zed"])):
            cv2.imwrite(
                str(save_dir / f"{count:05d}_{name}_{arr.shape[0]}.png"),
                cv2.cvtColor(arr, cv2.COLOR_RGB2BGR),
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
            f"[5cond] 이미지 저장 실패: {exc}", throttle_duration_sec=5.0
        )


def main(args=None):
    rclpy.init(args=args)
    node = SmolVLABridge5CondNode()
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
