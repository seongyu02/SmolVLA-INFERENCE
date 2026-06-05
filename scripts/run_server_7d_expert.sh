#!/bin/bash
# ══════════════════════════════════════════════════════════════════════════════
#  SmolVLA 7D Expert 정책 서버 — Terminal 1 에서 실행
#
#  사용법:
#    bash scripts/run_server_7d_expert.sh [모델경로]
#    bash scripts/run_server_7d_expert.sh   (기본 Expert 7D 경로 사용)
#
#  예:
#    bash scripts/run_server_7d_expert.sh
#    bash scripts/run_server_7d_expert.sh "/media/billye6/새 볼륨/Dobot/.../pretrained_model"
# ══════════════════════════════════════════════════════════════════════════════

set -e

REPO="$(cd "$(dirname "$0")/.." && pwd)"
VENV_ROOT="$HOME/SmolVLA/.venv_SmolVLA310"
VENV_ACTIVATE="$VENV_ROOT/bin/activate"
VENV_PYTHON="$VENV_ROOT/bin/python"
VENV_SITE="$VENV_ROOT/lib/python3.10/site-packages"

DEFAULT_MODEL="/media/billye6/새 볼륨/Dobot/SmolVLA_outputs_orange_v2/smolvla_orange_v2_7d_chunk50_action10_100000steps/checkpoints/100000/pretrained_model"
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

unset PYTHONPATH
export PYTHONNOUSERSITE=1
export PYTHONPATH="$VENV_SITE"
export MVCAM_COMMON_RUNENV=/opt/MVS/lib
export TORCHDYNAMO_DISABLE=1

echo "=============================="
echo " SmolVLA 7D Expert 정책 서버"
echo " model  : $MODEL_DIR"
echo " port   : 8001"
echo " device : GPU (cuda if available)"
echo " endpoint: POST http://localhost:8001/act"
echo "=============================="
echo ""

"$VENV_PYTHON" "$REPO/scripts/serve_policy_smolvla_7d.py" \
    --model-path "$MODEL_DIR" \
    --port 8001 \
    --host "0.0.0.0"
