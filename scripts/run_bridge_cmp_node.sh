#!/bin/bash
# smolvla_bridge_cmp_node 래퍼 스크립트 (Where-collapse 비교실험 cond A/B/C/D)
# venv_SmolVLA310 Python (CUDA torch) + ROS2 경로를 결합하여 실행

VENV="$HOME/SmolVLA/.venv_SmolVLA310"
VENV_PYTHON="$VENV/bin/python3"
VENV_SITE="$VENV/lib/python3.10/site-packages"
VENV_CUDA_LIB="$VENV_SITE/nvidia/cu12/lib"

ROS2_HUMBLE="/opt/ros/humble/local/lib/python3.10/dist-packages"
ROS2_WS="$HOME/SmolVLA/SmolVLA-INFERENCE/ros2/install/e6_vla_ros/lib/python3.10/site-packages"

NODE_BIN="$HOME/SmolVLA/SmolVLA-INFERENCE/ros2/install/e6_vla_ros/lib/e6_vla_ros/smolvla_bridge_cmp_node"

export LD_LIBRARY_PATH="$VENV_CUDA_LIB:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$VENV_SITE:$ROS2_WS:$ROS2_HUMBLE:${PYTHONPATH:-}"
export TORCHDYNAMO_DISABLE=1
export PYTHONNOUSERSITE=1

exec "$VENV_PYTHON" "$NODE_BIN" "$@"
