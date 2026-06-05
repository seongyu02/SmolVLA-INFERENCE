#!/usr/bin/env python3
"""
SmolVLA Dobot E6 추론 클라이언트 — HTTP 정책 서버 연결 버전

run_e6_client.py와 동일한 구조/FSM/제어 로직.
openpi WebsocketClientPolicy 대신 HTTP POST /act 로 SmolVLA 서버에 연결.

아키텍처:
  [serve_policy_smolvla.py] ← HTTP POST /act ← [이 스크립트] → Dobot E6

관측 계약 (SmolVLA):
  observation.state                : (13,) float32 — [j1..j6 deg, tx,ty,tz,rx,ry,rz, suction]
  observation.images.OBS_IMAGE_1   : (512, 512, 3) uint8 RGB — HIK 탑뷰
  observation.images.OBS_IMAGE_2   : (512, 512, 3) uint8 RGB — ZED 좌측 (없으면 zeros)
  task                             : str — 자연어 지시 문장

액션 계약 (SmolVLA 13D — convert_dobot_to_lerobot_v21.py 기준):
  action[0:6]  Δ관절각 (deg)   — 현재 관절에 누산 → MovJ
  action[6:12] Δ TCP (mm/deg) — 참고용 (현재 joint 제어 사용)
  action[12]   흡착 절대값 (0/1, 연속값 → hysteresis 처리)

서버 실행 예 (Terminal 1):
  cd ~/SmolVLA/SmolVLA-INFERENCE && bash run_server.sh

클라이언트 실행 예 (Terminal 2):
  cd ~/SmolVLA/SmolVLA-INFERENCE && bash run_client.sh \\
    --task_sequence "approach,pick,move_right,place_right"
"""
from __future__ import annotations

import argparse
import base64
import io
import json
import os
import sys
import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import requests

# ── hardware 경로 ──────────────────────────────────────────────────────────────
_REPO = Path(__file__).resolve().parents[2]
_HARDWARE = _REPO / "hardware"
_DOBOT_SDK = _HARDWARE / "dobot"
for _p in (_DOBOT_SDK, _HARDWARE):
    if _p.is_dir() and str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

_camera_capture_mod = None
try:
    import camera_capture as _camera_capture_mod  # type: ignore[import]
except ImportError:
    _camera_capture_mod = None


# ── 이미지 상수 ────────────────────────────────────────────────────────────────
IMG_SIZE = (512, 512)


# ── 기본값 ─────────────────────────────────────────────────────────────────────
DEFAULT_TASK_TEXT = "pick up the orange box from the left side and place it on the right side"
DEFAULT_MODEL_PATH = (
    "/media/billye6/새 볼륨/Dobot/SmolVLA_outputs_orange/"
    "smolvla_orange_chunk50_action10_100000steps/checkpoints/100000/pretrained_model"
)
INIT_POSES: dict[str, list[float]] = {
    "e6_v1": [90.128, 42.907, 59.355, -11.702, -87.582, 177.813],
    "ver1": [0.0, 0.0, 90.0, 0.0, 90.0, 0.0],
    "ver2": [-0.16, -43.88, 79.66, -2.49, 54.22, -0.15],
}


# ── stage 이름 파싱 (run_e6_client.py 동일) ────────────────────────────────────
def _stage_from_prompt(prompt: str) -> str:
    p = prompt.lower().strip()
    for tag in ("approach", "pick", "move", "place", "return"):
        if p.startswith(tag) or f"[{tag}]" in p:
            return tag
    return "unknown"


# ── stage 완료 판정 (run_e6_client.py와 완전 동일) ────────────────────────────
def _stage_complete(
    stage: str,
    tool_z: Optional[float],
    gripper: int,
    approach_z: float,
    lift_z: float,
    home_z: float,
) -> bool:
    if tool_z is None:
        return False
    if stage == "approach":
        return tool_z <= approach_z
    if stage == "pick":
        return gripper == 1 and tool_z >= lift_z
    if stage in ("place_left", "place_right", "place_middle", "place"):
        return gripper == 0 and tool_z >= lift_z
    if stage == "return":
        return tool_z >= home_z
    return False


# ── ZED 프레임 읽기 ────────────────────────────────────────────────────────────
def _resize_like_training(rgb: np.ndarray) -> np.ndarray:
    """Match dataset conversion: RGB image -> 512x512 with PIL LANCZOS."""
    arr = np.asarray(rgb, dtype=np.uint8)
    if arr.ndim == 2:
        arr = np.stack([arr] * 3, axis=-1)
    if arr.shape[-1] == 4:
        arr = arr[:, :, :3]

    image_mod = _pil_Image()
    pil = image_mod.fromarray(arr).convert("RGB")
    try:
        resample = image_mod.Resampling.LANCZOS
    except AttributeError:
        resample = image_mod.LANCZOS
    return np.asarray(pil.resize(IMG_SIZE, resample), dtype=np.uint8)


def _read_zed_frame(zed, zed_mat) -> np.ndarray:
    """ZED -> 512x512 RGB, matching dataset conversion."""
    if zed is None or zed_mat is None:
        return np.zeros((*IMG_SIZE, 3), dtype=np.uint8)
    try:
        import pyzed.sl as sl  # type: ignore
        if zed.grab() == sl.ERROR_CODE.SUCCESS:
            zed.retrieve_image(zed_mat, sl.VIEW.LEFT)
            frame = zed_mat.get_data()[:, :, :3][:, :, ::-1].copy()  # BGRA→RGB
            return _resize_like_training(frame)
    except Exception as exc:
        print(f"  [ZED] 읽기 실패: {exc}")
    return np.zeros((*IMG_SIZE, 3), dtype=np.uint8)


# ── 이미지 전처리 ──────────────────────────────────────────────────────────────
def _preprocess_hik(frame_rgb: np.ndarray) -> np.ndarray:
    """HIK -> 512x512 RGB, matching dataset conversion."""
    return _resize_like_training(frame_rgb)


def _to_b64png(rgb: np.ndarray) -> str:
    """(H,W,3) uint8 RGB → base64 PNG 문자열."""
    pil = _pil_Image().fromarray(rgb.astype(np.uint8))
    buf = io.BytesIO()
    pil.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


_PIL_Image = None
def _pil_Image():
    global _PIL_Image
    if _PIL_Image is None:
        from PIL import Image  # pylint: disable=import-outside-toplevel
        _PIL_Image = Image
    return _PIL_Image


# ── HTTP 정책 클라이언트 (WebsocketClientPolicy 대체) ─────────────────────────
class SmolVLAHttpPolicy:
    def __init__(self, host: str, port: int):
        self.url = f"http://{host}:{port}"
        self._session = requests.Session()

    def wait_for_server(self, timeout_sec: float = 60.0) -> None:
        """서버 ready 될 때까지 대기 (run_e6_client WebsocketClientPolicy 초기화와 동일 역할)."""
        import time as _time  # pylint: disable=import-outside-toplevel
        t0 = _time.monotonic()
        while _time.monotonic() - t0 < timeout_sec:
            try:
                resp = self._session.get(f"{self.url}/healthz", timeout=2)
                if resp.status_code == 200 and resp.json().get("model_loaded"):
                    return
            except Exception:
                pass
            _time.sleep(1.0)
        raise RuntimeError(f"SmolVLA server not ready at {self.url} after {timeout_sec}s")

    def get_server_metadata(self) -> dict:
        try:
            resp = self._session.get(f"{self.url}/healthz", timeout=3)
            return resp.json()
        except Exception:
            return {}

    def infer(self, obs: dict) -> dict:
        """관측 → 액션 청크.

        입력 obs 키:
          state                                : (13,) float32
          observation/exterior_image_1_left    : (H,W,3) uint8 RGB
          observation/exterior_image_2_left    : (H,W,3) uint8 RGB
          prompt                               : str (호환용, task 도 수락)
        반환 dict:
          actions : (n_action_steps, 13) np.ndarray
        """
        state = np.asarray(obs.get("observation/state", obs.get("state", np.zeros(13))),
                           dtype=np.float32)

        img1 = obs.get("observation/exterior_image_1_left",
                       np.zeros((*IMG_SIZE, 3), dtype=np.uint8))
        img2 = obs.get("observation/exterior_image_2_left",
                       np.zeros((*IMG_SIZE, 3), dtype=np.uint8))

        if img1.shape[:2] != IMG_SIZE:
            img1 = _resize_like_training(img1)
        if img2.shape[:2] != IMG_SIZE:
            img2 = _resize_like_training(img2)

        task_text: str = obs.get("task", obs.get("prompt", DEFAULT_TASK_TEXT))

        payload = {
            "state": state.tolist(),
            "image1_b64": _to_b64png(img1),
            "image2_b64": _to_b64png(img2),
            "task": task_text,
        }
        resp = self._session.post(f"{self.url}/act", json=payload, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        actions = np.asarray(data["actions"], dtype=np.float32)
        return {"actions": actions}


# ── config YAML 지원 ──────────────────────────────────────────────────────────
def _load_config(path: str) -> dict:
    import yaml  # pylint: disable=import-outside-toplevel
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _apply_config(cfg: dict, args: argparse.Namespace) -> None:
    _BOOL = {"dry_run", "no_camera", "no_zed", "show_actions", "no_init_pose",
             "auto_cycle_return_approach", "vacuum_check_enabled",
             "hold_on_bad_camera", "safety_hold_pose"}
    # YAML task → 모델 입력 prompt (단일 문장)
    if cfg.get("task") is not None:
        setattr(args, "prompt", str(cfg["task"]))
    if "skip_init_pose" in cfg:
        setattr(args, "no_init_pose", bool(cfg["skip_init_pose"]))
    for k, v in cfg.items():
        if k in ("task", "skip_init_pose"):
            continue
        attr = k.replace("-", "_")
        if attr not in vars(args):
            continue
        if attr in _BOOL:
            setattr(args, attr, bool(v))
        else:
            setattr(args, attr, v)


# ── 메인 ──────────────────────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(
        description="SmolVLA Dobot E6 클라이언트 — HTTP 정책 서버 연결 (run_e6_client.py 호환)"
    )

    # ── config ───────────────────────────────────────────────────────────────
    parser.add_argument("--config", default=None, help="YAML config 파일 경로")

    # ── 서버 ─────────────────────────────────────────────────────────────────
    parser.add_argument("--server_host", default="127.0.0.1")
    parser.add_argument("--server_port", type=int, default=8000)

    # ── 로봇 ─────────────────────────────────────────────────────────────────
    parser.add_argument("--robot_ip", default="192.168.5.1")
    parser.add_argument("--dry_run", action="store_true", help="로봇 전송 없이 추론만")
    parser.add_argument("--no_camera", action="store_true", help="카메라 미사용 (더미 이미지)")
    parser.add_argument("--no_zed", action="store_true", help="ZED 비활성화")
    parser.add_argument("--show_actions", action="store_true", help="매 스텝 액션 출력")

    # ── 프롬프트 (단일 문장 — SmolVLA 학습과 동일) ───────────────────────────
    parser.add_argument(
        "--prompt", "--task",
        dest="prompt",
        default=DEFAULT_TASK_TEXT,
        help="모델에 입력할 고정 자연어 지시 (--task 와 동일)",
    )

    # ── 제어 루프 ─────────────────────────────────────────────────────────────
    parser.add_argument("--hz", type=float, default=10.0, help="제어 주파수")
    parser.add_argument("--steps_per_inference", type=int, default=None,
                        help="청크에서 몇 스텝 실행 후 재추론. None=full chunk")
    parser.add_argument(
        "--max_runtime_sec", type=float, default=0.0,
        help="최대 실행 시간(초). 0=무제한(Ctrl+C). 단일 문장 모드 종료 조건",
    )
    parser.add_argument("--max_staleness_ms", type=float, default=5000.0)

    # ── 액션 안전 ─────────────────────────────────────────────────────────────
    parser.add_argument("--max_delta_deg", type=float, default=3.0,
                        help="스텝당 최대 관절 이동(도). 0=무제한")
    parser.add_argument("--min_tool_z", type=float, default=80.0,
                        help="안전: TCP Z(mm) 이하이면 루프 중단")
    parser.add_argument("--safety_hold_pose", action="store_true")
    parser.add_argument("--movj_velocity", type=int, default=30, help="MovJ 속도 0~100")
    parser.add_argument("--movj_accel", type=int, default=20, help="MovJ 가속 0~100")

    # ── 그리퍼(흡착) ──────────────────────────────────────────────────────────
    parser.add_argument("--grip_open_threshold", type=float, default=0.45)
    parser.add_argument("--grip_close_threshold", type=float, default=0.55)
    parser.add_argument("--grip_close_latch_steps", type=int, default=0)
    parser.add_argument("--vacuum_check_z", type=float, default=85.0)
    parser.add_argument("--vacuum_check_enabled", action="store_true", default=False)

    # ── 카메라 안전 ───────────────────────────────────────────────────────────
    parser.add_argument("--hold_on_bad_camera", action="store_true", default=True)
    parser.add_argument("--no_hold_on_bad_camera", action="store_false",
                        dest="hold_on_bad_camera")
    parser.add_argument("--camera_black_mean", type=float, default=8.0)
    parser.add_argument("--bad_camera_consecutive", type=int, default=10)

    # ── 초기 자세 ─────────────────────────────────────────────────────────────
    parser.add_argument("--no_init_pose", action="store_true", help="초기 자세 스킵")
    parser.add_argument("--init_pose_version", choices=["ver1", "ver2", "e6_v1"],
                        default="e6_v1")

    # ── 태스크 시퀀스 (run_e6_client.py 동일) ─────────────────────────────────
    parser.add_argument(
        "--task_sequence", default=None,
        help=(
            "(선택) 로봇 FSM stage 시퀀스. 모델 문장은 바꾸지 않음.\n"
            "미지정 시 단일 문장 추론만 반복. 예: approach,pick,move_right,place_right"
        ),
    )
    parser.add_argument("--approach_z_done", type=float, default=100.0)
    parser.add_argument("--lift_z_done", type=float, default=200.0)
    parser.add_argument("--home_z_done", type=float, default=300.0)
    parser.add_argument("--stage_done_steps", type=int, default=5)
    parser.add_argument("--stage_timeout_sec", type=float, default=30.0)

    # ── 로깅 ─────────────────────────────────────────────────────────────────
    parser.add_argument("--save_frames_dir", default=None)

    # two-pass parse: config → default → CLI override
    pre, _ = parser.parse_known_args()
    if pre.config:
        cfg = _load_config(pre.config)
        _apply_config(cfg, pre)
        parser.set_defaults(**{k: v for k, v in vars(pre).items() if k != "config"})
    args = parser.parse_args()

    # ── 1) HTTP 정책 서버 연결 ────────────────────────────────────────────────
    policy = SmolVLAHttpPolicy(host=args.server_host, port=args.server_port)
    print(f"[1/3] 정책 서버 연결 중: http://{args.server_host}:{args.server_port}")
    policy.wait_for_server(timeout_sec=120.0)
    print(f"      연결 완료. 서버 정보: {policy.get_server_metadata()}")

    # ── 2) 로봇 & 카메라 연결 ────────────────────────────────────────────────
    dashboard = feed = None
    if not args.dry_run:
        try:
            from dobot_api import DobotApiDashboard, DobotApiFeedBack  # noqa: PLC0415
            print(f"[2/3] 로봇 연결: {args.robot_ip}")
            dashboard = DobotApiDashboard(args.robot_ip, 29999)
            feed = DobotApiFeedBack(args.robot_ip, 30005)
            dashboard.EnableRobot()
            time.sleep(0.5)
            print("      EnableRobot 완료")
        except Exception as exc:
            print(f"[WARN] 로봇 연결 실패 ({exc}). dry_run 모드로 계속.")
            dashboard = feed = None

    camera = None
    zed = zed_mat = None
    if not args.no_camera:
        if _camera_capture_mod is not None:
            camera = _camera_capture_mod.CameraCapture()
            print(f"[2/3] HIK 카메라 초기화 완료")
        else:
            print("[WARN] camera_capture 모듈 없음 → 더미 이미지")
        if not args.no_zed:
            try:
                import pyzed.sl as sl  # type: ignore
                _zed = sl.Camera()
                _init = sl.InitParameters()
                _init.depth_mode = sl.DEPTH_MODE.NONE
                _init.camera_resolution = sl.RESOLUTION.HD1080
                _init.camera_fps = 30
                if _zed.open(_init) == sl.ERROR_CODE.SUCCESS:
                    zed = _zed
                    zed_mat = sl.Mat()
                    print(f"[2/3] ZED 카메라 초기화 완료")
                else:
                    print("[WARN] ZED 오픈 실패 → 더미")
            except Exception as exc:
                print(f"[WARN] ZED 초기화 실패 ({exc}) → 더미")

    # ── 2.5) 초기 자세 (run_e6_client.py와 동일) ─────────────────────────────
    if not args.no_init_pose and dashboard is not None:
        pose = INIT_POSES[args.init_pose_version]
        print(f"[2.5/3] 초기 자세 이동 ({args.init_pose_version}): {pose}")
        try:
            j1, j2, j3, j4, j5, j6 = pose
            dashboard.MovJ(j1, j2, j3, j4, j5, j6, 1,
                           v=args.movj_velocity, a=args.movj_accel)
            time.sleep(1.5)
        except Exception as exc:
            print(f"[WARN] 초기 자세 이동 실패: {exc}")

    # ── 3) 추론 루프 ─────────────────────────────────────────────────────────
    print(f"[3/3] 추론 루프 시작 (Ctrl+C 종료)")
    print(f"       task  : {args.prompt!r}")
    if args.task_sequence:
        print(f"       FSM   : {args.task_sequence!r} (모델 문장 고정)")
    else:
        print("       mode  : single-sentence (no task_sequence)")

    dt = 1.0 / args.hz
    step = 0
    current_chunk: np.ndarray | None = None
    chunk_index = 0
    chunk_infer_t0: float | None = None
    last_tool_on = 0
    grip_latch_remaining = 0
    bad_camera_streak = 0
    save_frame_count = 0
    save_frames_max = 60
    loop_tool_z: float | None = None
    vacuum_di_state: int = -1
    vacuum_fail_logged: bool = False

    stage_name = _stage_from_prompt(args.prompt)
    loop_start_mono = time.monotonic()
    stage_start_mono = loop_start_mono
    task_result = "RUNNING"

    # ── 태스크 시퀀스 초기화 ──────────────────────────────────────────────────
    _task_seq: list[str] | None = None
    _seq_idx: int = 0
    stage_done_streak: int = 0
    if args.task_sequence:
        _task_seq = [s.strip() for s in args.task_sequence.split(",") if s.strip()]
        stage_name = _task_seq[0]
        print(f"[SEQ] 태스크 시퀀스: {_task_seq}")

    steps_per_inference = args.steps_per_inference

    if args.save_frames_dir:
        os.makedirs(args.save_frames_dir, exist_ok=True)

    try:
        while True:
            t0 = time.monotonic()
            elapsed_runtime = t0 - loop_start_mono
            stage_elapsed = t0 - stage_start_mono

            # ── 최대 실행 시간 ────────────────────────────────────────────────
            if args.max_runtime_sec > 0 and elapsed_runtime > args.max_runtime_sec:
                task_result = "FAIL_TIMEOUT"
                print(f"[TASK_DONE] {task_result} runtime>{args.max_runtime_sec}s")
                break

            # ── 센서 기반 stage 완료 판정 (run_e6_client.py 동일) ─────────────
            if _task_seq is not None:
                if _stage_complete(stage_name, loop_tool_z, last_tool_on,
                                   args.approach_z_done, args.lift_z_done, args.home_z_done):
                    stage_done_streak += 1
                else:
                    stage_done_streak = 0

                stage_timed_out = (args.stage_timeout_sec > 0
                                   and stage_elapsed > args.stage_timeout_sec)
                stage_ok = stage_done_streak >= args.stage_done_steps

                if stage_ok or stage_timed_out:
                    reason = "done" if stage_ok else "timeout"
                    z_str = f"{loop_tool_z:.1f}mm" if loop_tool_z is not None else "N/A"
                    _seq_idx += 1
                    if _seq_idx >= len(_task_seq):
                        task_result = "SUCCESS"
                        print(f"[TASK_DONE] {task_result} ({reason}) z={z_str} step={step}")
                        break
                    next_stage = _task_seq[_seq_idx]
                    print(f"[STAGE_SWITCH] {stage_name}→{next_stage} ({reason}) z={z_str}")
                    stage_name = next_stage
                    stage_start_mono = time.monotonic()
                    stage_elapsed = 0.0
                    stage_done_streak = 0
                    current_chunk = None
                    chunk_index = 0
                    continue

            # ── 재추론 필요 여부 ──────────────────────────────────────────────
            chunk_len = current_chunk.shape[0] if current_chunk is not None else 0
            spi = steps_per_inference if steps_per_inference is not None else chunk_len
            need_infer = (current_chunk is None or chunk_index >= spi
                          or chunk_index >= chunk_len)

            infer_time_ms: float | None = None
            if need_infer:
                # ── 로봇 상태 읽기 ────────────────────────────────────────────
                current_joints_deg6 = np.zeros(6, dtype=np.float32)
                current_tcp6 = np.zeros(6, dtype=np.float32)
                current_gripper = float(last_tool_on)

                if dashboard is not None:
                    try:
                        res = dashboard.GetToolDO(1)
                        if res:
                            parts = res.split(",")
                            if len(parts) >= 3:
                                current_gripper = float(int(parts[2]))
                            elif parts[0].strip().isdigit():
                                current_gripper = float(int(parts[0].strip()))
                    except Exception:
                        pass

                if feed is not None:
                    try:
                        fb = feed.feedBackData()
                        if fb is not None:
                            current_joints_deg6 = np.asarray(
                                fb["QActual"][0], dtype=np.float32
                            ).ravel()[:6]
                            current_tcp6 = np.asarray(
                                fb["ToolVectorActual"][0], dtype=np.float32
                            ).ravel()[:6]
                    except Exception as exc:
                        print(f"  피드백 읽기 실패: {exc}")

                state_13 = np.concatenate(
                    [current_joints_deg6, current_tcp6, [current_gripper]],
                    dtype=np.float32,
                )

                # ── 이미지 수집 ───────────────────────────────────────────────
                if camera is not None:
                    frame = camera.get_frame()
                    if frame is not None:
                        obs_img = _preprocess_hik(np.asarray(frame, dtype=np.uint8))
                    else:
                        obs_img = np.zeros((*IMG_SIZE, 3), dtype=np.uint8)
                else:
                    obs_img = np.zeros((*IMG_SIZE, 3), dtype=np.uint8)

                zed_img = _read_zed_frame(zed, zed_mat)

                # ── bad camera 안전 홀드 ──────────────────────────────────────
                camera_hold = False
                if args.hold_on_bad_camera and float(obs_img.mean()) < args.camera_black_mean:
                    camera_hold = True
                    bad_camera_streak += 1
                else:
                    bad_camera_streak = 0
                if bad_camera_streak > args.bad_camera_consecutive:
                    task_result = "FAIL_SAFETY"
                    print(f"[TASK_DONE] {task_result} bad_camera>{args.bad_camera_consecutive}")
                    break

                # ── 프레임 저장 (디버깅) ──────────────────────────────────────
                if (args.save_frames_dir and save_frame_count < save_frames_max
                        and step % 20 == 0):
                    try:
                        path = os.path.join(
                            args.save_frames_dir,
                            f"frame_{save_frame_count:03d}_step{step}.png",
                        )
                        cv2.imwrite(path, cv2.cvtColor(obs_img, cv2.COLOR_RGB2BGR))
                        save_frame_count += 1
                    except Exception:
                        pass

                # ── 서버에 추론 요청 ─────────────────────────────────────────
                obs = {
                    "observation/exterior_image_1_left": obs_img,
                    "observation/exterior_image_2_left": zed_img,
                    "observation/state": state_13,
                    "prompt": args.prompt,          # 호환용
                    "task": args.prompt,            # SmolVLA 키
                }
                if not camera_hold:
                    t_infer0 = time.monotonic()
                    result = policy.infer(obs)
                    infer_time_ms = (time.monotonic() - t_infer0) * 1000.0
                    if step == 0 or step % 10 == 0:
                        print(
                            f"  [추론] step={step} {infer_time_ms:.1f}ms "
                            f"stage={stage_name!r}"
                        )
                    actions = np.asarray(result["actions"], dtype=np.float32)
                    if step == 0:
                        print(f"  [ACTION_SHAPE] {actions.shape}")
                    spi = (steps_per_inference if steps_per_inference is not None
                           else actions.shape[0])
                    current_chunk = actions[:spi]
                    chunk_index = 0
                    chunk_len = current_chunk.shape[0]
                    chunk_infer_t0 = time.monotonic()
                else:
                    time.sleep(dt)
                    continue

            # ── 청크 staleness ─────────────────────────────────────────────────
            if current_chunk is not None and chunk_infer_t0 is not None:
                stale_ms = (time.monotonic() - chunk_infer_t0) * 1000.0
                if stale_ms > args.max_staleness_ms:
                    print(f"  [STALE_DROP] {stale_ms:.0f}ms → chunk 폐기")
                    current_chunk = None
                    chunk_index = 0
                    time.sleep(dt)
                    continue

            if current_chunk is None or chunk_index >= chunk_len:
                time.sleep(dt)
                continue

            a = current_chunk[chunk_index]

            # ── 현재 피드백 ────────────────────────────────────────────────────
            current_joints_deg: np.ndarray | None = None
            current_tool_z: float | None = None
            if feed is not None:
                try:
                    fb = feed.feedBackData()
                    if fb is not None:
                        current_joints_deg = np.asarray(
                            fb["QActual"][0], dtype=np.float32
                        ).ravel()[:6]
                        tv = np.asarray(fb["ToolVectorActual"][0], dtype=np.float32).ravel()
                        current_tool_z = float(tv[2])
                        loop_tool_z = current_tool_z
                except Exception as exc:
                    print(f"  피드백 읽기 실패: {exc}")

            # ── min_tool_z 안전 ────────────────────────────────────────────────
            if current_tool_z is not None and current_tool_z < args.min_tool_z:
                if args.safety_hold_pose:
                    chunk_index += 1
                    time.sleep(dt)
                    continue
                task_result = "FAIL_SAFETY"
                print(f"[TASK_DONE] {task_result} tool_z={current_tool_z:.1f} < {args.min_tool_z}")
                break

            # ── 진공 흡착 확인 (run_e6_client.py 동일) ────────────────────────
            if (args.vacuum_check_enabled and dashboard is not None
                    and current_tool_z is not None):
                if current_tool_z <= args.vacuum_check_z and last_tool_on == 1:
                    try:
                        di_res = dashboard.ToolDI(1)
                        if di_res is not None:
                            parts = di_res.split(",")
                            vacuum_di_state = int(parts[2]) if len(parts) >= 3 else int(parts[0])
                    except Exception:
                        pass
                    if vacuum_di_state == 0 and not vacuum_fail_logged:
                        print(
                            f"  [PICK_FAIL] ToolDI(1)=0 → 흡착 없음 "
                            f"z={current_tool_z:.1f}mm step={step}"
                        )
                        vacuum_fail_logged = True

            # ── 액션 파싱 (SmolVLA 13D → delta joints + suction) ─────────────
            # action[0:6]  = Δ관절각 (deg)  — make_action_array 기준
            # action[6:12] = Δ TCP (mm/deg) — 참고용
            # action[12]   = 흡착 절대값
            delta_joints = a[0:6].copy()
            raw_suction = float(a[12])

            # max_delta_deg 클리핑
            if args.max_delta_deg > 0:
                delta_joints = np.clip(delta_joints,
                                       -args.max_delta_deg, args.max_delta_deg)

            if args.show_actions:
                print(
                    f"  [ACTION] step={step:4d} | "
                    f"Δj=[{', '.join(f'{v:+.3f}' for v in delta_joints)}] | "
                    f"suction_raw={raw_suction:.3f}"
                )

            # ── 그리퍼(흡착) 처리 (run_e6_client.py hysteresis 동일) ──────────
            desired_tool: int | None = None
            if raw_suction >= args.grip_close_threshold:
                desired_tool = 1
                if args.grip_close_latch_steps > 0:
                    grip_latch_remaining = args.grip_close_latch_steps
            elif raw_suction <= args.grip_open_threshold:
                if grip_latch_remaining > 0:
                    grip_latch_remaining -= 1
                    desired_tool = 1
                else:
                    desired_tool = 0
            if desired_tool is not None and desired_tool != last_tool_on:
                print(
                    f"  [GRIP] {last_tool_on}→{desired_tool} "
                    f"raw={raw_suction:.3f} step={step}"
                )
                if dashboard is not None:
                    dashboard.ToolDO(1, desired_tool)
                last_tool_on = desired_tool
                vacuum_fail_logged = False

            # ── 로봇 이동 (관절 delta 누산 → MovJ) ───────────────────────────
            if not args.dry_run and current_joints_deg is not None and dashboard is not None:
                target_joints = current_joints_deg + delta_joints
                j1, j2, j3, j4, j5, j6 = [float(v) for v in target_joints[:6]]
                try:
                    dashboard.MovJ(j1, j2, j3, j4, j5, j6, 1,
                                   v=args.movj_velocity, a=args.movj_accel)
                except Exception as exc:
                    print(f"  [WARN] MovJ 실패: {exc}")

            chunk_index += 1
            step += 1

            elapsed = time.monotonic() - t0
            sleep_t = dt - elapsed
            if sleep_t > 0:
                time.sleep(sleep_t)

    except KeyboardInterrupt:
        print("\n[TASK_DONE] Ctrl+C 종료")
    finally:
        if dashboard is not None:
            try:
                dashboard.ToolDO(1, 0)
            except Exception:
                pass
        try:
            if camera is not None and hasattr(camera, "close"):
                camera.close()
        except Exception:
            pass
        print(f"[결과] {task_result}")


# ── 실행 예시 ──────────────────────────────────────────────────────────────────
# Terminal 1:
#   cd ~/SmolVLA/SmolVLA-INFERENCE && bash run_server.sh
#
# Terminal 2 (단일 문장, 권장):
#   cd ~/SmolVLA/SmolVLA-INFERENCE && bash run_client.sh --config examples/e6/config_orange.yaml
#
# FSM 옵션 (모델 문장은 동일):
#   bash run_client.sh --config examples/e6/config_orange_fsm.yaml
if __name__ == "__main__":
    main()
