#!/bin/bash
# ══════════════════════════════════════════════════════════════════════════════
#  SmolVLA 정책 서버 — Terminal 1 에서 실행
#
#  E6-VLA_INFERENCE/run_server_v26.sh 와 동일한 역할.
#  openpi WebSocket 서버 대신 SmolVLA FastAPI HTTP 서버를 port 8000 에서 시작.
#
#  사용법:
#    bash run_server.sh [모델경로]
#    bash run_server.sh   (기본 경로 사용)
#
#  예:
#    bash run_server.sh
#    bash run_server.sh "/media/billye6/새 볼륨/Dobot/.../pretrained_model"
# ══════════════════════════════════════════════════════════════════════════════

set -e

REPO="$(cd "$(dirname "$0")" && pwd)"
VENV_ROOT="$HOME/SmolVLA/.venv_SmolVLA310"
VENV_ACTIVATE="$VENV_ROOT/bin/activate"
VENV_PYTHON="$VENV_ROOT/bin/python"
VENV_SITE="$VENV_ROOT/lib/python3.10/site-packages"

DEFAULT_MODEL="/media/billye6/새 볼륨/Dobot/SmolVLA_outputs/smolvla_dobot_chunk10_20000steps/checkpoints/020000/pretrained_model"
MODEL_DIR="${1:-$DEFAULT_MODEL}"

if [ ! -d "$MODEL_DIR" ]; then
    echo "[오류] 모델 폴더 없음: $MODEL_DIR"
    exit 1
fi

if [ ! -x "$VENV_PYTHON" ]; then
    echo "[오류] venv python 없음: $VENV_PYTHON"
    exit 1
fi

if [ -f "$VENV_ACTIVATE" ]; then
    # shellcheck disable=SC1090
    source "$VENV_ACTIVATE"
fi

# ~/.local / ROS 가 venv보다 앞서면 uvicorn 등 import 실패 → venv site 우선
unset PYTHONPATH
export PYTHONNOUSERSITE=1
export PYTHONPATH="$VENV_SITE"
export MVCAM_COMMON_RUNENV=/opt/MVS/lib
export TORCHDYNAMO_DISABLE=1

echo "=============================="
echo " SmolVLA 정책 서버"
echo " model  : $MODEL_DIR"
echo " port   : 8000"
echo " device : GPU (cuda if available)"
echo " endpoint: POST http://localhost:8000/act"
echo "=============================="
echo ""

"$VENV_PYTHON" "$REPO/scripts/serve_policy_smolvla.py" \
    --model-path "$MODEL_DIR" \
    --port 8000 \
    --host "0.0.0.0"
