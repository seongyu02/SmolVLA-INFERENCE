#!/usr/bin/env python3
"""
SmolVLA 추론 서버 (FastAPI)

E6-VLA_INFERENCE/scripts/serve_policy.py 와 동일한 역할:
  - SmolVLA 모델을 로드하고 HTTP 서버로 서빙
  - 클라이언트(run_smolvla_client.py)가 POST /act 로 관측을 보내고 액션을 받음

사용법 (Terminal 1):
  source ~/SmolVLA/.venv_SmolVLA310/bin/activate
  cd ~/SmolVLA/SmolVLA-INFERENCE
  bash run_server.sh
  또는
  python scripts/serve_policy_smolvla.py --model-path MODEL_PATH --port 8000

아키텍처:
  [serve_policy_smolvla.py] ← HTTP POST /act ← [run_smolvla_client.py] → Dobot E6
"""
from __future__ import annotations

import argparse
import base64
import io
import logging
import sys
from pathlib import Path
from typing import List

import numpy as np
import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from PIL import Image
from pydantic import BaseModel
from transformers import AutoProcessor

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

# ── 모델 경로 기본값 ──────────────────────────────────────────────────────────
DEFAULT_MODEL_PATH = (
    "/media/billye6/새 볼륨/Dobot/SmolVLA_outputs_orange/"
    "smolvla_orange_chunk50_action10_100000steps/checkpoints/100000/pretrained_model"
)
IMG_SIZE = (512, 512)

# ── FastAPI 앱 / 전역 정책 ────────────────────────────────────────────────────
app = FastAPI(title="SmolVLA Policy Server")
_policy = None
_tokenizer = None
_device = None


class ActRequest(BaseModel):
    state: List[float]         # 13D
    image1_b64: str            # base64 PNG, (512,512,3) uint8 RGB — OBS_IMAGE_1 (HIK)
    image2_b64: str            # base64 PNG, (512,512,3) uint8 RGB — OBS_IMAGE_2 (ZED)
    task: str                  # 자연어 지시 문장


class ActResponse(BaseModel):
    actions: List[List[float]] # shape (n_action_steps, 13)


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
    if _policy is None or _tokenizer is None:
        raise HTTPException(status_code=503, detail="Policy not loaded")

    state = np.asarray(req.state, dtype=np.float32)
    if state.shape != (13,):
        raise HTTPException(status_code=422, detail=f"state must be 13D, got {state.shape}")

    tok = _tokenizer(
        [req.task], padding="longest", truncation=True,
        max_length=48, return_tensors="pt",
    )

    batch = {
        "observation.images.OBS_IMAGE_1": _b64_to_tensor(req.image1_b64),
        "observation.images.OBS_IMAGE_2": _b64_to_tensor(req.image2_b64),
        "observation.state": torch.from_numpy(state).unsqueeze(0).to(_device),
        "task": req.task,
        "observation.language.tokens": tok["input_ids"].to(_device),
        "observation.language.attention_mask": tok["attention_mask"].bool().to(_device),
    }

    with torch.no_grad():
        action_out = _policy.select_action(batch)

    if torch.is_tensor(action_out):
        actions_np = action_out.detach().float().cpu().numpy()
    else:
        actions_np = np.asarray(action_out, dtype=np.float32)

    # select_action returns (n_action_steps, 13) or (13,)
    if actions_np.ndim == 1:
        actions_np = actions_np.reshape(1, -1)

    if actions_np.shape[-1] != 13:
        raise HTTPException(status_code=500, detail=f"Unexpected action shape: {actions_np.shape}")

    return ActResponse(actions=actions_np.tolist())


def load_model(model_path: str) -> None:
    global _policy, _tokenizer, _device
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy  # pylint: disable=import-outside-toplevel

    _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info("디바이스: %s", _device)
    log.info("모델 로딩: %s", model_path)

    _tokenizer = AutoProcessor.from_pretrained(
        "HuggingFaceTB/SmolVLM2-500M-Video-Instruct"
    ).tokenizer
    _policy = SmolVLAPolicy.from_pretrained(model_path)
    _policy.to(_device)
    _policy.eval()
    if hasattr(_policy, "reset"):
        _policy.reset()
    log.info("SmolVLA 로드 완료 (device=%s)", _device)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="SmolVLA 추론 서버 (E6-VLA serve_policy.py 호환 역할)"
    )
    parser.add_argument(
        "--model-path", "--policy-dir", dest="model_path",
        default=DEFAULT_MODEL_PATH,
        help="SmolVLA pretrained_model 폴더 경로",
    )
    parser.add_argument("--port", type=int, default=8000)
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
    print(" SmolVLA 정책 서버")
    print(f" model : {model_path}")
    print(f" host  : {args.host}  port: {args.port}")
    print(f" local IP: {local_ip}")
    print(f" device: {_device}")
    print(" endpoint: POST /act")
    print("============================================")

    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
