#!/bin/bash
# SmolVLA 5cond 추론 자원 사용량 벤치마크 실행기.
#   1) venv_SmolVLA310(+CUDA) 로 측정 -> JSON
#   2) 시스템 python3(matplotlib) 로 막대그래프 PNG
#
# 사용:
#   ./run_infer_benchmark.sh                 # cond 1, 10회
#   ./run_infer_benchmark.sh --cond 3        # cond 3
#   ./run_infer_benchmark.sh --cond 1 --runs 20
#   ./run_infer_benchmark.sh --model_path /media/.../checkpoints/020000/pretrained_model
set -e

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV="$HOME/SmolVLA/.venv_SmolVLA310"
VENV_PYTHON="$VENV/bin/python3"
VENV_SITE="$VENV/lib/python3.10/site-packages"
VENV_CUDA_LIB="$VENV_SITE/nvidia/cu12/lib"

JSON="$HERE/infer_resource_result.json"
PNG="$HERE/infer_resource_bar.png"

# ── 1) 측정 (venv + CUDA) ────────────────────────────────────────────────────
echo "[run] measuring with venv_SmolVLA310 ..."
LD_LIBRARY_PATH="$VENV_CUDA_LIB:${LD_LIBRARY_PATH:-}" \
PYTHONPATH="$VENV_SITE:${PYTHONPATH:-}" \
TORCHDYNAMO_DISABLE=1 \
PYTHONNOUSERSITE=1 \
  "$VENV_PYTHON" "$HERE/infer_resource_benchmark.py" --out-json "$JSON" "$@"

# ── 2) 그래프 (시스템 python3 + matplotlib) ──────────────────────────────────
echo "[run] plotting with system python3 ..."
/usr/bin/python3 "$HERE/plot_infer_resource.py" --json "$JSON" --out "$PNG"

echo "[run] done."
echo "  JSON : $JSON"
echo "  PNG  : $PNG"
