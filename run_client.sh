#!/bin/bash
# ══════════════════════════════════════════════════════════════════════════════
#  SmolVLA 로봇 클라이언트 — Terminal 2 에서 실행 (run_server.sh 먼저 시작)
#
#  E6-VLA_INFERENCE/run_client.sh 와 동일한 2-터미널 구조.
#  SmolVLA는 학습과 같이 **단일 문장**으로 추론하는 것이 기본입니다.
#
# ── 단일 문장 실행 (권장) ─────────────────────────────────────────────────
#
#   bash run_client.sh --config examples/e6/config_orange.yaml
#
#   bash run_client.sh --prompt "pick up the orange box from the left side and place it on the right side"
#   bash run_client.sh --task "pick up the orange box from the left side and place it on the right side"
#
# ── FSM stage 제어 (선택, 모델 문장은 그대로) ─────────────────────────────
#
#   bash run_client.sh --config examples/e6/config_orange_fsm.yaml
#   bash run_client.sh --task_sequence "approach,pick,move_right,place_right" \\
#     --prompt "pick up the orange box from the left side and place it on the right side"
#
# ── threshold 조정 ─────────────────────────────────────────────────────────
#
#   --max_runtime_sec 120   단일 문장 모드 종료 시간(초). 0=무제한
#   --approach_z_done 100   approach 완료: TCP Z(mm) ≤ 이 값 (FSM 시)
#   --lift_z_done 200       pick/place 완료: TCP Z(mm) ≥ 이 값 (FSM 시)
#   --stage_done_steps 5    조건 연속 만족 스텝 수 (FSM 시)
#   --stage_timeout_sec 30  stage별 최대 시간(초) (FSM 시)
#   --hz 10                 제어 주파수
#   --movj_velocity 30      로봇 속도 0~100
#   --max_delta_deg 3       스텝당 최대 관절 이동(도)
#
# ── 디버그 ─────────────────────────────────────────────────────────────────
#
#   bash run_client.sh --dry_run --config examples/e6/config_orange.yaml
#   bash run_client.sh --dry_run --no_camera --no_zed --max_runtime_sec 10
#   bash run_client.sh --save_frames_dir ~/debug_frames
#
# ══════════════════════════════════════════════════════════════════════════════

REPO="$(cd "$(dirname "$0")" && pwd)"
VENV_ROOT="$HOME/SmolVLA/.venv_SmolVLA310"
VENV_ACTIVATE="$VENV_ROOT/bin/activate"
VENV_PYTHON="$VENV_ROOT/bin/python"
VENV_SITE="$VENV_ROOT/lib/python3.10/site-packages"
SCRIPT="$REPO/examples/e6/run_smolvla_client.py"

if [ ! -f "$SCRIPT" ]; then
  echo "[ERROR] SmolVLA client script not found: $SCRIPT"
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
echo " SmolVLA 로봇 클라이언트"
echo " server : 127.0.0.1:8000"
echo " robot  : 192.168.5.1"
echo " mode   : single-sentence (default)"
echo "=============================="
echo ""

"$VENV_PYTHON" "$SCRIPT" \
    --server_host "127.0.0.1" \
    --server_port 8000 \
    --robot_ip "192.168.5.1" \
    --hz 10 \
    --steps_per_inference 8 \
    --max_delta_deg 3 \
    --movj_velocity 30 \
    --movj_accel 20 \
    "$@"
