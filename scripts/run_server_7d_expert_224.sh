#!/bin/bash
# SmolVLA 7D Expert 224 inference server.
# Usage:
#   bash scripts/run_server_7d_expert_224.sh
#   bash scripts/run_server_7d_expert_224.sh "/path/to/checkpoint/pretrained_model"

set -e

REPO="$(cd "$(dirname "$0")/.." && pwd)"
if [ -x "$HOME/SmolVLA/.venv_SmolVLA310/bin/python" ]; then
    VENV_ROOT="$HOME/SmolVLA/.venv_SmolVLA310"
else
    VENV_ROOT="$HOME/SmolVLA/.venv_SmolVLA"
fi
VENV_ACTIVATE="$VENV_ROOT/bin/activate"
VENV_PYTHON="$VENV_ROOT/bin/python"

DEFAULT_MODEL="/media/billy/새 볼륨4/Dobot/SmolVLA_outputs_orange_v3/smolvla_orange_v3_224_7d_chunk50_action10_100000steps/checkpoints/100000/pretrained_model"
MODEL_DIR="${1:-$DEFAULT_MODEL}"

if [ ! -d "$MODEL_DIR" ]; then
    echo "[ERROR] model directory not found: $MODEL_DIR"
    echo "Pass the actual 224 checkpoint path as the first argument."
    exit 1
fi

if [ ! -x "$VENV_PYTHON" ]; then
    echo "[ERROR] venv python not found: $VENV_PYTHON"
    exit 1
fi

VENV_SITE="$("$VENV_PYTHON" -c 'import site; print(site.getsitepackages()[0])')"

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
echo " SmolVLA 7D Expert 224 inference server"
echo " model  : $MODEL_DIR"
echo " port   : 8003"
echo " device : GPU (cuda if available)"
echo " endpoint: POST http://localhost:8003/act"
echo "=============================="
echo ""

"$VENV_PYTHON" "$REPO/scripts/serve_policy_smolvla_7d_224.py" \
    --model-path "$MODEL_DIR" \
    --port 8003 \
    --host "0.0.0.0"
