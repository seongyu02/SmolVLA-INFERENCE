#!/bin/bash
# ?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧
#  SmolVLA 7D Expert 濡쒕큸 ?대씪?댁뼵????Terminal 2 (run_server_7d_expert.sh 癒쇱?)
#
# ?? ?⑥씪 臾몄옣 ?ㅽ뻾 (沅뚯옣) ?????????????????????????????????????????????????
#
#   bash scripts/run_client_7d_expert.sh
#   bash scripts/run_client_7d_expert.sh --config examples/e6/config_orange_7d_expert.yaml
#
# ?? FSM stage ?쒖뼱 (?좏깮) ?????????????????????????????????????????????????
#
#   bash scripts/run_client_7d_expert.sh --config examples/e6/config_orange_fsm_7d_expert.yaml
#
# ?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧

REPO="$(cd "$(dirname "$0")/.." && pwd)"
VENV_ROOT="$HOME/SmolVLA/.venv_SmolVLA310"
VENV_ACTIVATE="$VENV_ROOT/bin/activate"
VENV_PYTHON="$VENV_ROOT/bin/python"
VENV_SITE="$VENV_ROOT/lib/python3.10/site-packages"
SCRIPT="$REPO/examples/e6/run_smolvla_client_7d.py"
DEFAULT_CONFIG="$REPO/examples/e6/config_orange_7d_expert.yaml"

if [ ! -f "$SCRIPT" ]; then
  echo "[ERROR] SmolVLA 7D client script not found: $SCRIPT"
  exit 1
fi

if [ ! -x "$VENV_PYTHON" ]; then
  echo "[ERROR] venv python not found: $VENV_PYTHON"
  exit 1
fi

if [ -f "$VENV_ACTIVATE" ]; then
  # shellcheck disable=SC1090
  source "$VENV_ACTIVATE"
fi

unset PYTHONPATH
export PYTHONNOUSERSITE=1
export PYTHONPATH="$VENV_SITE:$REPO/hardware:$REPO/hardware/dobot"
export MVCAM_COMMON_RUNENV=/opt/MVS/lib

echo "=============================="
echo " SmolVLA 7D Expert robot client"
echo " server : 127.0.0.1:8001"
echo " robot  : 192.168.5.1"
echo " mode   : single-sentence (default)"
echo "=============================="
echo ""

HAS_CONFIG=0
for arg in "$@"; do
  if [ "$arg" = "--config" ]; then
    HAS_CONFIG=1
    break
  fi
done

EXTRA_ARGS=()
if [ "$HAS_CONFIG" -eq 0 ]; then
  EXTRA_ARGS=(--config "$DEFAULT_CONFIG")
fi

"$VENV_PYTHON" "$SCRIPT" \
    "${EXTRA_ARGS[@]}" \
    --server_host "127.0.0.1" \
    --server_port 8001 \
    --robot_ip "192.168.5.1" \
    --hz 16 \
    --steps_per_inference 10 \
    --max_delta_deg 3 \
    --movj_velocity 30 \
    --movj_accel 20 \
    "$@"
