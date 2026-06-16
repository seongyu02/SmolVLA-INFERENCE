#!/usr/bin/env python3
"""
SmolVLA Dobot E6 7D 異붾줎 ?대씪?댁뼵????HTTP ?뺤콉 ?쒕쾭 ?곌껐 踰꾩쟾

run_e6_client.py? ?숈씪??援ъ“/FSM/?쒖뼱 濡쒖쭅.
openpi WebsocketClientPolicy ???HTTP POST /act 濡?SmolVLA 7D ?쒕쾭???곌껐.

?꾪궎?띿쿂:
  [serve_policy_smolvla_7d_224.py] ??HTTP POST /act ??[???ㅽ겕由쏀듃] ??Dobot E6

愿痢?怨꾩빟 (SmolVLA 7D):
  observation.state                : (7,) float32 ??[j1..j6 deg, gripper_state]
  observation.images.OBS_IMAGE_1   : (224, 224, 3) uint8 RGB ??HIK ?묐럭
  observation.images.OBS_IMAGE_2   : (224, 224, 3) uint8 RGB ??ZED 醫뚯륫 (?놁쑝硫?zeros)
  task                             : str ???먯뿰??吏??臾몄옣

?≪뀡 怨꾩빟 (SmolVLA 7D):
  action[0:6]  ?愿?덇컖 (deg)   ???꾩옱 愿?덉뿉 ?꾩궛 ??MovJ
  action[6]    ?≪갑 ?덈?媛?(0/1, ?곗냽媛???hysteresis 泥섎━)

?쒕쾭 ?ㅽ뻾 ??(Terminal 1):
  cd ~/SmolVLA/SmolVLA-INFERENCE && bash scripts/run_server_7d_expert_224.sh

?대씪?댁뼵???ㅽ뻾 ??(Terminal 2):
  cd ~/SmolVLA/SmolVLA-INFERENCE && bash scripts/run_client_7d_expert_224.sh
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

# ?? hardware 寃쎈줈 ??????????????????????????????????????????????????????????????
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


# ?? ?대?吏 ?곸닔 ????????????????????????????????????????????????????????????????
IMG_SIZE = (224, 224)


# ?? 湲곕낯媛??????????????????????????????????????????????????????????????????????
DEFAULT_TASK_TEXT = "pick up the orange box from the left side and place it on the right side"
DEFAULT_MODEL_PATH = (
    "/media/billy/??蹂쇰ⅷ4/Dobot/SmolVLA_outputs_orange_v3/"
    "smolvla_orange_v3_224_7d_chunk50_action10_100000steps/checkpoints/100000/pretrained_model"
)
STATE_DIM = 7
INIT_POSES: dict[str, list[float]] = {
    "e6_v1": [90.128, 42.907, 59.355, -11.702, -87.582, 177.813],
    "ver1": [0.0, 0.0, 90.0, 0.0, 90.0, 0.0],
    "ver2": [-0.16, -43.88, 79.66, -2.49, 54.22, -0.15],
}


# ?? stage ?대쫫 ?뚯떛 (run_e6_client.py ?숈씪) ????????????????????????????????????
def _stage_from_prompt(prompt: str) -> str:
    p = prompt.lower().strip()
    for tag in ("approach", "pick", "move", "place", "return"):
        if p.startswith(tag) or f"[{tag}]" in p:
            return tag
    return "unknown"


# ?? stage ?꾨즺 ?먯젙 (run_e6_client.py? ?꾩쟾 ?숈씪) ????????????????????????????
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


# ?? ZED ?꾨젅???쎄린 ????????????????????????????????????????????????????????????
def _resize_like_training(rgb: np.ndarray) -> np.ndarray:
    """Return the 224x224 RGB image expected by the 224 dataset/model."""
    arr = np.asarray(rgb, dtype=np.uint8)
    if arr.ndim == 2:
        arr = np.stack([arr] * 3, axis=-1)
    if arr.shape[-1] == 4:
        arr = arr[:, :, :3]
    if arr.shape[:2] == IMG_SIZE and arr.shape[-1] == 3:
        return arr

    image_mod = _pil_Image()
    pil = image_mod.fromarray(arr).convert("RGB")
    try:
        resample = image_mod.Resampling.LANCZOS
    except AttributeError:
        resample = image_mod.LANCZOS
    return np.asarray(pil.resize(IMG_SIZE, resample), dtype=np.uint8)


_last_zed_frame: np.ndarray | None = None


def _preprocess_zed_like_training(rgb: np.ndarray) -> np.ndarray:
    """ZED left view -> 224x224 RGB using the raw-data collection crop pipeline."""
    import cv2

    arr = np.asarray(rgb, dtype=np.uint8)
    if arr.ndim == 2:
        arr = np.stack([arr] * 3, axis=-1)
    if arr.shape[-1] == 4:
        arr = arr[:, :, :3]
    if arr.shape[:2] == IMG_SIZE and arr.shape[-1] == 3:
        return arr

    # Raw ZED collection path:
    # HD1080 -> 640x480 -> crop[120:480, 150:510] -> 224x224.
    arr = cv2.resize(arr, (640, 480))
    arr = arr[120:480, 150:510]
    arr = cv2.resize(arr, IMG_SIZE)
    return arr.astype(np.uint8)


def _read_zed_frame(zed, zed_mat) -> np.ndarray:
    """ZED -> 224x224 RGB, matching the raw-data collection crop pipeline."""
    global _last_zed_frame
    if zed is None or zed_mat is None:
        return (
            _last_zed_frame
            if _last_zed_frame is not None
            else np.zeros((*IMG_SIZE, 3), dtype=np.uint8)
        )
    try:
        import pyzed.sl as sl  # type: ignore
        if zed.grab() == sl.ERROR_CODE.SUCCESS:
            zed.retrieve_image(zed_mat, sl.VIEW.LEFT)
            frame = zed_mat.get_data()[:, :, :3][:, :, ::-1].copy()  # BGRA?뭃GB
            _last_zed_frame = _preprocess_zed_like_training(frame)
            return _last_zed_frame
    except Exception as exc:
        print(f"  [ZED] ?쎄린 ?ㅽ뙣: {exc}")
    return (
        _last_zed_frame
        if _last_zed_frame is not None
        else np.zeros((*IMG_SIZE, 3), dtype=np.uint8)
    )


# ?? ?대?吏 ?꾩쿂由???????????????????????????????????????????????????????????????
def _preprocess_hik(frame_rgb: np.ndarray) -> np.ndarray:
    """HIK -> 224x224 RGB. CameraCapture already applies the training crop pipeline."""
    return _resize_like_training(frame_rgb)


def _to_b64png(rgb: np.ndarray) -> str:
    """(H,W,3) uint8 RGB ??base64 PNG 臾몄옄??"""
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


# ?? HTTP ?뺤콉 ?대씪?댁뼵??(WebsocketClientPolicy ?泥? ?????????????????????????
class SmolVLAHttpPolicy:
    def __init__(self, host: str, port: int):
        self.url = f"http://{host}:{port}"
        self._session = requests.Session()

    def wait_for_server(self, timeout_sec: float = 60.0) -> None:
        """?쒕쾭 ready ???뚭퉴吏 ?湲?(run_e6_client WebsocketClientPolicy 珥덇린?붿? ?숈씪 ??븷)."""
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
        """愿痢????≪뀡 泥?겕.

        ?낅젰 obs ??
          state                                : (7,) float32
          observation/exterior_image_1_left    : (H,W,3) uint8 RGB
          observation/exterior_image_2_left    : (H,W,3) uint8 RGB
          prompt                               : str (?명솚?? task ???섎씫)
        諛섑솚 dict:
          actions : (n_action_steps, 7) np.ndarray
        """
        state = np.asarray(obs.get("observation/state", obs.get("state", np.zeros(STATE_DIM))),
                           dtype=np.float32)

        img1 = obs.get("observation/exterior_image_1_left",
                       np.zeros((*IMG_SIZE, 3), dtype=np.uint8))
        img2 = obs.get("observation/exterior_image_2_left",
                       np.zeros((*IMG_SIZE, 3), dtype=np.uint8))

        if img1.shape[:2] != IMG_SIZE:
            img1 = _resize_like_training(img1)
        if img2.shape[:2] != IMG_SIZE:
            img2 = _preprocess_zed_like_training(img2)

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


# ?? config YAML 吏????????????????????????????????????????????????????????????
def _load_config(path: str) -> dict:
    import yaml  # pylint: disable=import-outside-toplevel
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _apply_config(cfg: dict, args: argparse.Namespace) -> None:
    _BOOL = {"dry_run", "no_camera", "no_zed", "show_actions", "no_init_pose",
             "auto_cycle_return_approach", "vacuum_check_enabled",
             "hold_on_bad_camera", "safety_hold_pose"}
    # YAML task ??紐⑤뜽 ?낅젰 prompt (?⑥씪 臾몄옣)
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


# ?? 硫붿씤 ??????????????????????????????????????????????????????????????????????
def main() -> None:
    parser = argparse.ArgumentParser(
        description="SmolVLA Dobot E6 ?대씪?댁뼵????HTTP ?뺤콉 ?쒕쾭 ?곌껐 (run_e6_client.py ?명솚)"
    )

    # ?? config ???????????????????????????????????????????????????????????????
    parser.add_argument("--config", default=None, help="YAML config ?뚯씪 寃쎈줈")

    # ?? ?쒕쾭 ?????????????????????????????????????????????????????????????????
    parser.add_argument("--server_host", default="127.0.0.1")
    parser.add_argument("--server_port", type=int, default=8003)

    # ?? 濡쒕큸 ?????????????????????????????????????????????????????????????????
    parser.add_argument("--robot_ip", default="192.168.5.1")
    parser.add_argument("--dry_run", action="store_true", help="Run inference without sending robot commands")
    parser.add_argument("--no_camera", action="store_true", help="移대찓??誘몄궗??(?붾? ?대?吏)")
    parser.add_argument("--no_zed", action="store_true", help="ZED 鍮꾪솢?깊솕")
    parser.add_argument("--show_actions", action="store_true", help="留??ㅽ뀦 ?≪뀡 異쒕젰")

    # ?? ?꾨＼?꾪듃 (?⑥씪 臾몄옣 ??SmolVLA ?숈뒿怨??숈씪) ???????????????????????????
    parser.add_argument(
        "--prompt", "--task",
        dest="prompt",
        default=DEFAULT_TASK_TEXT,
        help="紐⑤뜽???낅젰??怨좎젙 ?먯뿰??吏??(--task ? ?숈씪)",
    )

    # ?? ?쒖뼱 猷⑦봽 ?????????????????????????????????????????????????????????????
    parser.add_argument("--hz", type=float, default=16.0, help="Control frequency")
    parser.add_argument("--steps_per_inference", type=int, default=10,
                        help="泥?겕?먯꽌 紐??ㅽ뀦 ?ㅽ뻾 ???ъ텛濡? None=full chunk")
    parser.add_argument(
        "--max_runtime_sec", type=float, default=0.0,
        help="理쒕? ?ㅽ뻾 ?쒓컙(珥?. 0=臾댁젣??Ctrl+C). ?⑥씪 臾몄옣 紐⑤뱶 醫낅즺 議곌굔",
    )
    parser.add_argument("--max_staleness_ms", type=float, default=5000.0)

    # ?? ?≪뀡 ?덉쟾 ?????????????????????????????????????????????????????????????
    parser.add_argument("--max_delta_deg", type=float, default=3.0,
                        help="Maximum joint delta per step in degrees. 0 disables clipping.")
    parser.add_argument("--min_tool_z", type=float, default=80.0,
                        help="?덉쟾: TCP Z(mm) ?댄븯?대㈃ 猷⑦봽 以묐떒")
    parser.add_argument("--safety_hold_pose", action="store_true")
    parser.add_argument("--movj_velocity", type=int, default=30, help="MovJ ?띾룄 0~100")
    parser.add_argument("--movj_accel", type=int, default=20, help="MovJ 媛??0~100")

    # ?? 洹몃━???≪갑) ??????????????????????????????????????????????????????????
    parser.add_argument("--grip_open_threshold", type=float, default=0.45)
    parser.add_argument("--grip_close_threshold", type=float, default=0.55)
    parser.add_argument("--grip_close_latch_steps", type=int, default=0)
    parser.add_argument("--vacuum_check_z", type=float, default=85.0)
    parser.add_argument("--vacuum_check_enabled", action="store_true", default=False)

    # ?? 移대찓???덉쟾 ???????????????????????????????????????????????????????????
    parser.add_argument("--hold_on_bad_camera", action="store_true", default=True)
    parser.add_argument("--no_hold_on_bad_camera", action="store_false",
                        dest="hold_on_bad_camera")
    parser.add_argument("--camera_black_mean", type=float, default=8.0)
    parser.add_argument("--bad_camera_consecutive", type=int, default=10)

    # ?? 珥덇린 ?먯꽭 ?????????????????????????????????????????????????????????????
    parser.add_argument("--no_init_pose", action="store_true", help="珥덇린 ?먯꽭 ?ㅽ궢")
    parser.add_argument("--init_pose_version", choices=["ver1", "ver2", "e6_v1"],
                        default="e6_v1")

    # ?? ?쒖뒪???쒗??(run_e6_client.py ?숈씪) ?????????????????????????????????
    parser.add_argument(
        "--task_sequence", default=None,
        help=(
            "(?좏깮) 濡쒕큸 FSM stage ?쒗?? 紐⑤뜽 臾몄옣? 諛붽씀吏 ?딆쓬.\n"
            "誘몄??????⑥씪 臾몄옣 異붾줎留?諛섎났. ?? approach,pick,move_right,place_right"
        ),
    )
    parser.add_argument("--approach_z_done", type=float, default=100.0)
    parser.add_argument("--lift_z_done", type=float, default=200.0)
    parser.add_argument("--home_z_done", type=float, default=300.0)
    parser.add_argument("--stage_done_steps", type=int, default=5)
    parser.add_argument("--stage_timeout_sec", type=float, default=30.0)

    # ?? 濡쒓퉭 ?????????????????????????????????????????????????????????????????
    parser.add_argument("--save_frames_dir", default=None)

    # two-pass parse: config ??default ??CLI override
    pre, _ = parser.parse_known_args()
    if pre.config:
        cfg = _load_config(pre.config)
        _apply_config(cfg, pre)
        parser.set_defaults(**{k: v for k, v in vars(pre).items() if k != "config"})
    args = parser.parse_args()

    if args.config:
        cfg = _load_config(args.config)
        if cfg.get("model_path"):
            print(f"[7D] model_path={cfg['model_path']}")

    # ?? 1) HTTP ?뺤콉 ?쒕쾭 ?곌껐 ????????????????????????????????????????????????
    policy = SmolVLAHttpPolicy(host=args.server_host, port=args.server_port)
    print(f"[1/3] ?뺤콉 ?쒕쾭 ?곌껐 以? http://{args.server_host}:{args.server_port}")
    policy.wait_for_server(timeout_sec=120.0)
    print(f"      ?곌껐 ?꾨즺. ?쒕쾭 ?뺣낫: {policy.get_server_metadata()}")

    # ?? 2) 濡쒕큸 & 移대찓???곌껐 ????????????????????????????????????????????????
    dashboard = feed = None
    if not args.dry_run:
        try:
            from dobot_api import DobotApiDashboard, DobotApiFeedBack  # noqa: PLC0415
            print(f"[2/3] 濡쒕큸 ?곌껐: {args.robot_ip}")
            dashboard = DobotApiDashboard(args.robot_ip, 29999)
            feed = DobotApiFeedBack(args.robot_ip, 30005)
            dashboard.EnableRobot()
            time.sleep(0.5)
            print("      EnableRobot ?꾨즺")
        except Exception as exc:
            print(f"[WARN] 濡쒕큸 ?곌껐 ?ㅽ뙣 ({exc}). dry_run 紐⑤뱶濡?怨꾩냽.")
            dashboard = feed = None

    camera = None
    zed = zed_mat = None
    if not args.no_camera:
        if _camera_capture_mod is not None:
            camera = _camera_capture_mod.CameraCapture()
            print(f"[2/3] HIK 移대찓??珥덇린???꾨즺")
        else:
            print("[WARN] camera_capture 紐⑤뱢 ?놁쓬 ???붾? ?대?吏")
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
                    print(f"[2/3] ZED 移대찓??珥덇린???꾨즺")
                else:
                    print("[WARN] ZED ?ㅽ뵂 ?ㅽ뙣 ???붾?")
            except Exception as exc:
                print(f"[WARN] ZED 珥덇린???ㅽ뙣 ({exc}) ???붾?")

    # ?? 2.5) 珥덇린 ?먯꽭 (run_e6_client.py? ?숈씪) ?????????????????????????????
    if not args.no_init_pose and dashboard is not None:
        pose = INIT_POSES[args.init_pose_version]
        print(f"[2.5/3] 珥덇린 ?먯꽭 ?대룞 ({args.init_pose_version}): {pose}")
        try:
            j1, j2, j3, j4, j5, j6 = pose
            dashboard.MovJ(j1, j2, j3, j4, j5, j6, 1,
                           v=args.movj_velocity, a=args.movj_accel)
            time.sleep(1.5)
        except Exception as exc:
            print(f"[WARN] 珥덇린 ?먯꽭 ?대룞 ?ㅽ뙣: {exc}")

    # ?? 3) 異붾줎 猷⑦봽 ?????????????????????????????????????????????????????????
    print(f"[3/3] 異붾줎 猷⑦봽 ?쒖옉 (Ctrl+C 醫낅즺)")
    print(f"       task  : {args.prompt!r}")
    if args.task_sequence:
        print(f"       FSM   : {args.task_sequence!r} (紐⑤뜽 臾몄옣 怨좎젙)")
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

    # ?? ?쒖뒪???쒗??珥덇린????????????????????????????????????????????????????
    _task_seq: list[str] | None = None
    _seq_idx: int = 0
    stage_done_streak: int = 0
    if args.task_sequence:
        _task_seq = [s.strip() for s in args.task_sequence.split(",") if s.strip()]
        stage_name = _task_seq[0]
        print(f"[SEQ] ?쒖뒪???쒗?? {_task_seq}")

    steps_per_inference = args.steps_per_inference

    if args.save_frames_dir:
        os.makedirs(args.save_frames_dir, exist_ok=True)

    try:
        while True:
            t0 = time.monotonic()
            elapsed_runtime = t0 - loop_start_mono
            stage_elapsed = t0 - stage_start_mono

            # ?? 理쒕? ?ㅽ뻾 ?쒓컙 ????????????????????????????????????????????????
            if args.max_runtime_sec > 0 and elapsed_runtime > args.max_runtime_sec:
                task_result = "FAIL_TIMEOUT"
                print(f"[TASK_DONE] {task_result} runtime>{args.max_runtime_sec}s")
                break

            # ?? ?쇱꽌 湲곕컲 stage ?꾨즺 ?먯젙 (run_e6_client.py ?숈씪) ?????????????
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
                    print(f"[STAGE_SWITCH] {stage_name}->{next_stage} ({reason}) z={z_str}")
                    stage_name = next_stage
                    stage_start_mono = time.monotonic()
                    stage_elapsed = 0.0
                    stage_done_streak = 0
                    current_chunk = None
                    chunk_index = 0
                    continue

            # ?? ?ъ텛濡??꾩슂 ?щ? ??????????????????????????????????????????????
            chunk_len = current_chunk.shape[0] if current_chunk is not None else 0
            spi = steps_per_inference if steps_per_inference is not None else chunk_len
            need_infer = (current_chunk is None or chunk_index >= spi
                          or chunk_index >= chunk_len)

            infer_time_ms: float | None = None
            if need_infer:
                # ?? 濡쒕큸 ?곹깭 ?쎄린 ????????????????????????????????????????????
                current_joints_deg6 = np.zeros(6, dtype=np.float32)
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
                    except Exception as exc:
                        print(f"  ?쇰뱶諛??쎄린 ?ㅽ뙣: {exc}")

                obs_state_7d = np.concatenate(
                    [current_joints_deg6, [current_gripper]],
                    dtype=np.float32,
                )
                if step == 0 or step % 10 == 0:
                    print(f"  [7D] generated state7={obs_state_7d.tolist()}")

                # ?? ?대?吏 ?섏쭛 ???????????????????????????????????????????????
                if camera is not None:
                    frame = camera.get_frame()
                    if frame is not None:
                        obs_img = _preprocess_hik(np.asarray(frame, dtype=np.uint8))
                    else:
                        obs_img = np.zeros((*IMG_SIZE, 3), dtype=np.uint8)
                else:
                    obs_img = np.zeros((*IMG_SIZE, 3), dtype=np.uint8)

                zed_img = _read_zed_frame(zed, zed_mat)

                # ?? bad camera ?덉쟾 ?????????????????????????????????????????
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

                # ?? ?꾨젅?????(?붾쾭源? ??????????????????????????????????????
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

                # ?? ?쒕쾭??異붾줎 ?붿껌 ?????????????????????????????????????????
                obs = {
                    "observation/exterior_image_1_left": obs_img,
                    "observation/exterior_image_2_left": zed_img,
                    "observation/state": obs_state_7d,
                    "prompt": args.prompt,
                    "task": args.prompt,
                }
                if not camera_hold:
                    t_infer0 = time.monotonic()
                    result = policy.infer(obs)
                    infer_time_ms = (time.monotonic() - t_infer0) * 1000.0
                    if step == 0 or step % 10 == 0:
                        print(
                            f"  [異붾줎] step={step} {infer_time_ms:.1f}ms "
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

            # ?? 泥?겕 staleness ?????????????????????????????????????????????????
            if current_chunk is not None and chunk_infer_t0 is not None:
                stale_ms = (time.monotonic() - chunk_infer_t0) * 1000.0
                if stale_ms > args.max_staleness_ms:
                    print(f"  [STALE_DROP] {stale_ms:.0f}ms ??chunk ?먭린")
                    current_chunk = None
                    chunk_index = 0
                    time.sleep(dt)
                    continue

            if current_chunk is None or chunk_index >= chunk_len:
                time.sleep(dt)
                continue

            a = current_chunk[chunk_index]

            # ?? ?꾩옱 ?쇰뱶諛?????????????????????????????????????????????????????
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
                    print(f"  ?쇰뱶諛??쎄린 ?ㅽ뙣: {exc}")

            # ?? min_tool_z ?덉쟾 ????????????????????????????????????????????????
            if current_tool_z is not None and current_tool_z < args.min_tool_z:
                if args.safety_hold_pose:
                    chunk_index += 1
                    time.sleep(dt)
                    continue
                task_result = "FAIL_SAFETY"
                print(f"[TASK_DONE] {task_result} tool_z={current_tool_z:.1f} < {args.min_tool_z}")
                break

            # ?? 吏꾧났 ?≪갑 ?뺤씤 (run_e6_client.py ?숈씪) ????????????????????????
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
                            f"  [PICK_FAIL] ToolDI(1)=0 ???≪갑 ?놁쓬 "
                            f"z={current_tool_z:.1f}mm step={step}"
                        )
                        vacuum_fail_logged = True

            # ?? ?≪뀡 ?뚯떛 (SmolVLA 7D ??delta joints + gripper) ??????????????
            joint_delta = a[:6].copy()
            gripper_cmd = float(a[6])
            delta_joints = joint_delta
            raw_suction = gripper_cmd
            if step == 0 or args.show_actions:
                print(f"  [7D] action_7={a.tolist()}")
                print(f"  [7D] joint_delta={joint_delta.tolist()}")
                print(f"  [7D] gripper_cmd={gripper_cmd}")

            # max_delta_deg limit
            if args.max_delta_deg > 0:
                delta_joints = np.clip(delta_joints,
                                       -args.max_delta_deg, args.max_delta_deg)

            if args.show_actions:
                print(
                    f"  [ACTION] step={step:4d} | "
                    f"?j=[{', '.join(f'{v:+.3f}' for v in delta_joints)}] | "
                    f"suction_raw={raw_suction:.3f}"
                )

            # ?? 洹몃━???≪갑) 泥섎━ (run_e6_client.py hysteresis ?숈씪) ??????????
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
                    f"  [GRIP] {last_tool_on}->{desired_tool} "
                    f"raw={raw_suction:.3f} step={step}"
                )
                if dashboard is not None:
                    dashboard.ToolDO(1, desired_tool)
                last_tool_on = desired_tool
                vacuum_fail_logged = False

            # ?? 濡쒕큸 ?대룞 (愿??delta ?꾩궛 ??MovJ) ???????????????????????????
            if not args.dry_run and current_joints_deg is not None and dashboard is not None:
                target_joints = current_joints_deg + delta_joints
                j1, j2, j3, j4, j5, j6 = [float(v) for v in target_joints[:6]]
                try:
                    dashboard.MovJ(j1, j2, j3, j4, j5, j6, 1,
                                   v=args.movj_velocity, a=args.movj_accel)
                except Exception as exc:
                    print(f"  [WARN] MovJ ?ㅽ뙣: {exc}")

            chunk_index += 1
            step += 1

            elapsed = time.monotonic() - t0
            sleep_t = dt - elapsed
            if sleep_t > 0:
                time.sleep(sleep_t)

    except KeyboardInterrupt:
        print("\n[TASK_DONE] Ctrl+C 醫낅즺")
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
        print(f"[寃곌낵] {task_result}")


# ?? ?ㅽ뻾 ?덉떆 ??????????????????????????????????????????????????????????????????
# Terminal 1:
#   cd ~/SmolVLA/SmolVLA-INFERENCE && bash scripts/run_server_7d_expert_224.sh
#
# Terminal 2 (?⑥씪 臾몄옣, 沅뚯옣):
#   cd ~/SmolVLA/SmolVLA-INFERENCE && bash scripts/run_client_7d_expert_224.sh
#
# FSM ?듭뀡 (紐⑤뜽 臾몄옣? ?숈씪):
#   bash scripts/run_client_7d_expert_224.sh --config examples/e6/config_orange_7d_expert_224.yaml
if __name__ == "__main__":
    main()

