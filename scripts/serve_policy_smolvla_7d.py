#!/usr/bin/env python3
"""
SmolVLA 7D 추론 서버 (FastAPI)

E6-VLA_INFERENCE/scripts/serve_policy.py 와 동일한 역할:
  - SmolVLA 7D 모델을 로드하고 HTTP 서버로 서빙
  - 클라이언트(run_smolvla_client_7d.py)가 POST /act 로 관측을 보내고 액션을 받음

사용법 (Terminal 1):
  source ~/SmolVLA/.venv_SmolVLA310/bin/activate
  cd ~/SmolVLA/SmolVLA-INFERENCE
  bash scripts/run_server_7d_expert.sh
  또는
  python scripts/serve_policy_smolvla_7d.py --model-path MODEL_PATH --port 8001

아키텍처:
  [serve_policy_smolvla_7d.py] ← HTTP POST /act ← [run_smolvla_client_7d.py] → Dobot E6
"""
from __future__ import annotations

import argparse
import base64
import io
import logging
import sys
import time
from pathlib import Path
from typing import List

import numpy as np
import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from PIL import Image
from pydantic import BaseModel

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

STATE_DIM = 7
ACTION_DIM = 7

# ── 모델 경로 기본값 (7D Expert) ───────────────────────────────────────────────
DEFAULT_MODEL_PATH = (
    "/media/billye6/새 볼륨/Dobot/SmolVLA_outputs_orange_v2/"
    "smolvla_orange_v2_7d_chunk50_action10_100000steps/checkpoints/100000/pretrained_model"
)
EXPERT_7D_BASE_PATH = DEFAULT_MODEL_PATH
SAFETENSORS_SINGLE_FILE = "model.safetensors"
IMG_SIZE = (512, 512)

# ── FastAPI 앱 / 전역 정책 ────────────────────────────────────────────────────
app = FastAPI(title="SmolVLA 7D Policy Server")
_policy = None
_preprocessor = None
_postprocessor = None
_device = None
_resolved_model_path: str | None = None
_act_call_count = 0


class ActRequest(BaseModel):
    state: List[float]         # 7D
    image1_b64: str            # base64 PNG, (512,512,3) uint8 RGB — OBS_IMAGE_1 (HIK)
    image2_b64: str            # base64 PNG, (512,512,3) uint8 RGB — OBS_IMAGE_2 (ZED)
    task: str                  # 자연어 지시 문장


class ActResponse(BaseModel):
    actions: List[List[float]] # shape (n_action_steps, 7)


def _b64_to_tensor(b64str: str) -> torch.Tensor:
    """base64 PNG → (1, 3, H, W) float32 [0,1] tensor on _device."""
    data = base64.b64decode(b64str)
    img = Image.open(io.BytesIO(data)).convert("RGB")
    img = img.resize(IMG_SIZE, Image.LANCZOS)
    arr = np.asarray(img, dtype=np.float32) / 255.0  # (H,W,3)
    t = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)  # (1,3,H,W)
    return t.to(_device)


@app.get("/healthz")
def healthz():
    return {"status": "ok", "model_loaded": _policy is not None}


@app.post("/act", response_model=ActResponse)
def act(req: ActRequest):
    global _act_call_count
    if _policy is None or _preprocessor is None or _postprocessor is None:
        raise HTTPException(status_code=503, detail="Policy not loaded")

    state = np.asarray(req.state, dtype=np.float32)
    if state.shape != (STATE_DIM,):
        raise HTTPException(
            status_code=422,
            detail=f"state must be 7D, got {state.shape}",
        )

    t_total0 = time.monotonic()

    t0 = time.monotonic()
    batch = {
        "observation.images.OBS_IMAGE_1": _b64_to_tensor(req.image1_b64),
        "observation.images.OBS_IMAGE_2": _b64_to_tensor(req.image2_b64),
        "observation.state": torch.from_numpy(state).unsqueeze(0),
        "task": req.task,
    }
    decode_ms = (time.monotonic() - t0) * 1000.0

    t0 = time.monotonic()
    observation = _preprocessor(batch)
    preprocess_ms = (time.monotonic() - t0) * 1000.0

    with torch.no_grad():
        t0 = time.monotonic()
        action_chunk = _policy.predict_action_chunk(observation)
        if action_chunk.ndim == 2:
            action_chunk = action_chunk.unsqueeze(0)

        n_steps = int(_policy.config.n_action_steps)
        action_chunk = action_chunk[:, :n_steps, :]
        predict_ms = (time.monotonic() - t0) * 1000.0

        t0 = time.monotonic()
        chunk_size = action_chunk.shape[1]
        processed = []
        for i in range(chunk_size):
            processed.append(_postprocessor(action_chunk[:, i, :]))
        actions_t = torch.stack(processed, dim=1).squeeze(0)
        postproc_ms = (time.monotonic() - t0) * 1000.0

    total_ms = (time.monotonic() - t_total0) * 1000.0
    log.info(
        "[7D] /act timing decode=%.0fms preprocess=%.0fms predict=%.0fms "
        "postproc=%.0fms total=%.0fms",
        decode_ms, preprocess_ms, predict_ms, postproc_ms, total_ms,
    )

    actions_np = actions_t.detach().float().cpu().numpy()
    if actions_np.ndim == 1:
        actions_np = actions_np.reshape(1, -1)

    if actions_np.shape[-1] != ACTION_DIM:
        raise HTTPException(
            status_code=500,
            detail=f"Unexpected action shape: {actions_np.shape}",
        )

    if _act_call_count == 0 or _act_call_count % 50 == 0:
        log.info("[7D] inference input state shape=%s", state.shape)
        log.info("[7D] inference output action shape=%s", actions_np.shape)
        log.info("[7D] first action=%s", np.round(actions_np[0], 4).tolist())
        log.info("[7D] gripper step0=%.3f step9=%.3f",
                 float(actions_np[0, 6]), float(actions_np[min(9, len(actions_np) - 1), 6]))
    _act_call_count += 1

    return ActResponse(actions=actions_np.tolist())


def _resolve_base_model_path(base_path: str) -> str:
    """LoRA adapter config may reference another machine's mount path."""
    candidates = [base_path]
    if "/media/billy/" in base_path:
        candidates.append(base_path.replace("/media/billy/", "/media/billye6/"))
    candidates.append(EXPERT_7D_BASE_PATH)
    for path in candidates:
        if Path(path).is_dir() and (Path(path) / SAFETENSORS_SINGLE_FILE).is_file():
            if path != base_path:
                log.warning("[7D LoRA] base model path corrected: %s -> %s", base_path, path)
            return path
    raise FileNotFoundError(
        f"LoRA base model not found. Tried: {candidates}"
    )


def load_model(model_path: str) -> None:
    global _policy, _preprocessor, _postprocessor, _device, _resolved_model_path
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy  # pylint: disable=import-outside-toplevel
    from lerobot.processor.converters import (  # pylint: disable=import-outside-toplevel
        batch_to_transition,
        policy_action_to_transition,
        transition_to_batch,
        transition_to_policy_action,
    )
    from lerobot.processor import PolicyProcessorPipeline  # pylint: disable=import-outside-toplevel
    from lerobot.utils.constants import (  # pylint: disable=import-outside-toplevel
        POLICY_POSTPROCESSOR_DEFAULT_NAME,
        POLICY_PREPROCESSOR_DEFAULT_NAME,
    )

    _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _resolved_model_path = model_path
    log.info("디바이스: %s", _device)
    log.info("[7D] state_dim=%d", STATE_DIM)
    log.info("[7D] expected_action_dim=%d", ACTION_DIM)
    log.info("[7D] model_path=%s", model_path)
    log.info("모델 로딩: %s", model_path)

    model_dir = Path(model_path)
    full_weights = model_dir / SAFETENSORS_SINGLE_FILE
    adapter_weights = model_dir / "adapter_model.safetensors"

    if full_weights.is_file():
        _policy = SmolVLAPolicy.from_pretrained(model_path)
        log.info("[7D] loaded full checkpoint (Expert)")
    elif adapter_weights.is_file():
        from peft import PeftConfig, PeftModel  # pylint: disable=import-outside-toplevel

        peft_config = PeftConfig.from_pretrained(model_path)
        base_path = _resolve_base_model_path(peft_config.base_model_name_or_path)
        log.info("[7D LoRA] loading base model: %s", base_path)
        log.info("[7D LoRA] loading adapter: %s", model_path)
        base_policy = SmolVLAPolicy.from_pretrained(base_path)
        _policy = PeftModel.from_pretrained(
            base_policy, model_path, config=peft_config, is_trainable=False,
        )
        log.info("[7D LoRA] PEFT adapter loaded")
    else:
        raise FileNotFoundError(
            f"No model.safetensors or adapter_model.safetensors in {model_path}"
        )

    _policy.to(_device)
    _policy.eval()
    if hasattr(_policy, "reset"):
        _policy.reset()

    policy_cfg = _policy.config
    _preprocessor = PolicyProcessorPipeline.from_pretrained(
        model_path,
        config_filename=f"{POLICY_PREPROCESSOR_DEFAULT_NAME}.json",
        to_transition=batch_to_transition,
        to_output=transition_to_batch,
    )
    _postprocessor = PolicyProcessorPipeline.from_pretrained(
        model_path,
        config_filename=f"{POLICY_POSTPROCESSOR_DEFAULT_NAME}.json",
        to_transition=policy_action_to_transition,
        to_output=transition_to_policy_action,
    )
    log.info(
        "[7D] pre/post processor loaded (n_action_steps=%s)",
        getattr(policy_cfg, "n_action_steps", "?"),
    )
    log.info("SmolVLA 7D 로드 완료 (device=%s)", _device)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="SmolVLA 7D 추론 서버 (E6-VLA serve_policy.py 호환 역할)"
    )
    parser.add_argument(
        "--model-path", "--policy-dir", dest="model_path",
        default=DEFAULT_MODEL_PATH,
        help="SmolVLA pretrained_model 폴더 경로",
    )
    parser.add_argument("--port", type=int, default=8001)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--check-only", action="store_true",
                        help="모델 로드 확인 후 종료 (서버 미시작)")
    args, _ = parser.parse_known_args()

    model_path = Path(args.model_path).expanduser()
    if (model_path / "pretrained_model").is_dir():
        model_path = model_path / "pretrained_model"
    if not model_path.exists():
        print(f"[ERROR] 모델 경로 없음: {model_path}")
        sys.exit(1)

    load_model(str(model_path))

    if args.check_only:
        log.info("[OK] 모델 확인 완료 (--check-only 모드, 서버 미시작)")
        return

    import socket  # pylint: disable=import-outside-toplevel
    hostname = socket.gethostname()
    try:
        local_ip = socket.gethostbyname(hostname)
    except Exception:
        local_ip = "N/A"

    print("============================================")
    print(" SmolVLA 7D 정책 서버")
    print(f" model : {model_path}")
    print(f" host  : {args.host}  port: {args.port}")
    print(f" local IP: {local_ip}")
    print(f" device: {_device}")
    print(" endpoint: POST /act")
    print("============================================")

    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
