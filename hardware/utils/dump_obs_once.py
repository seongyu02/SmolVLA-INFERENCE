#!/usr/bin/env python3
import argparse
import json
import os
import time
from pathlib import Path

import numpy as np


def _maybe_add_dobot_api_to_path():
    """
    Mirror run_robot_inference.py behavior:
    add Dobot-Arm-DataCollect dir to sys.path so `import dobot_api` works.
    """
    here = Path(__file__).resolve().parent
    dobot_dir = here / "dobot_ws" / "src" / "Dobot-Arm-DataCollect"
    if dobot_dir.is_dir():
        import sys

        sys.path.insert(0, str(dobot_dir))
        return str(dobot_dir)
    return None


def _maybe_import_camera_capture():
    here = Path(__file__).resolve().parent
    cam_py = here / "camera_capture.py"
    if not cam_py.is_file():
        return None
    import importlib.util

    spec = importlib.util.spec_from_file_location("camera_capture", str(cam_py))
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def main():
    ap = argparse.ArgumentParser(description="Dump one raw observation (image/state/gripper/prompt) to npz.")
    ap.add_argument("--out_npz", required=True, help="Output .npz path (e.g. /tmp/obs_one.npz)")
    ap.add_argument("--robot_ip", type=str, default="192.168.5.1")
    ap.add_argument("--prompt", type=str, default="pick and place")
    ap.add_argument("--no_camera", action="store_true", help="Use black dummy image")
    ap.add_argument("--hz", type=float, default=20.0, help="Only used for metadata")
    ap.add_argument("--wait_s", type=float, default=0.0, help="Sleep before capture (seconds)")
    ap.add_argument("--use_hikrobot", action="store_true", default=True, help="Prefer HIKRobot backend")
    ap.add_argument("--no_use_hikrobot", action="store_false", dest="use_hikrobot")
    args = ap.parse_args()

    out_npz = Path(os.path.expanduser(args.out_npz)).resolve()
    out_npz.parent.mkdir(parents=True, exist_ok=True)
    out_meta = out_npz.with_suffix(".meta.json")

    # Dobot feedback (joint deg) + dashboard ToolDO(gripper)
    dobot_path = _maybe_add_dobot_api_to_path()
    feed = None
    dashboard = None
    gripper_on = 0
    joint_7 = np.zeros(7, dtype=np.float32)
    tool_vec = None
    try:
        # NOTE: reuse same dobot API package path expectations as run_robot_inference.py
        from dobot_api import DobotApiDashboard, DobotApiFeedBack

        dashboard = DobotApiDashboard(args.robot_ip, 29999)
        feed = DobotApiFeedBack(args.robot_ip, 30005)

        # Best-effort gripper read
        try:
            res = dashboard.GetToolDO(1)
            if res and len(res) > 0:
                parts = res.split(",")
                if len(parts) >= 3:
                    gripper_on = int(parts[2])
                elif len(parts) >= 1 and parts[0].strip().isdigit():
                    gripper_on = int(parts[0].strip())
        except Exception:
            pass

        try:
            data = feed.feedBackData()
            q_deg = np.array(data["QActual"][0], dtype=np.float32)
            q_rad = np.deg2rad(q_deg)
            joint_7[:6] = q_rad
            joint_7[6] = 0.0
            tv = data.get("ToolVectorActual", None)
            if tv is not None:
                tool_vec = np.array(tv[0], dtype=np.float32)
        except Exception:
            pass
    except Exception as e:
        hint = f" (sys.path added: {dobot_path})" if dobot_path else ""
        print(f"[WARN] Dobot API init failed (dump will use zeros): {e}{hint}")
        feed = None
        dashboard = None

    # Camera frame (raw uint8 HWC)
    H, W = 224, 224
    cam_name = None
    obs_img = np.zeros((H, W, 3), dtype=np.uint8)
    if args.wait_s > 0:
        time.sleep(args.wait_s)

    camera = None
    if not args.no_camera:
        cam_mod = _maybe_import_camera_capture()
        if cam_mod is None:
            print("[WARN] camera_capture.py not found. Using dummy image.")
        else:
            try:
                camera = cam_mod.CameraCapture(use_hikrobot=bool(args.use_hikrobot))
                cam_name = getattr(camera, "_name", None)
                frame = camera.get_frame()
                if frame is not None and frame.shape[:2] == (H, W):
                    obs_img = np.asarray(frame, dtype=np.uint8)
                else:
                    print("[WARN] camera frame invalid. Using dummy image.")
            except Exception as e:
                print(f"[WARN] camera init/capture failed. Using dummy image. err={e}")
                camera = None

    if camera is not None:
        try:
            camera.close()
        except Exception:
            pass

    exterior = obs_img
    wrist = obs_img.copy()
    gripper = np.array([float(gripper_on)], dtype=np.float32)
    prompt = np.array(args.prompt, dtype=object)
    ts = np.array([time.time()], dtype=np.float64)

    meta = {
        "schema": "openpi_obs_npz_v1",
        "created_unix_s": float(ts[0]),
        "robot_ip": args.robot_ip,
        "prompt": args.prompt,
        "hz": float(args.hz),
        "image": {
            "shape": list(exterior.shape),
            "dtype": str(exterior.dtype),
            "layout": "HWC",
            "color": "RGB",
            "notes": "This is the raw image array passed to policy.infer (224x224).",
            "camera_backend": cam_name,
        },
        "state": {
            "joint_position": {
                "shape": list(joint_7.shape),
                "dtype": str(joint_7.dtype),
                "units": "rad",
                "order": "q1..q6, dummy0",
                "notes": "Matches run_robot_inference.py: joint_7[:6]=deg2rad(QActual), joint_7[6]=0",
            },
            "gripper_position": {
                "shape": [1],
                "dtype": "float32",
                "units": "0/1",
                "source": "Dashboard GetToolDO(1)",
            },
            "tool_vector_actual": None if tool_vec is None else tool_vec.astype(np.float32).tolist(),
            "tool_vector_units": "mm,deg",
        },
    }

    np.savez_compressed(
        out_npz,
        exterior_image=exterior,
        wrist_image=wrist,
        joint_position=joint_7,
        gripper=gripper,
        prompt=prompt,
        timestamp=ts,
        meta_json=np.array(json.dumps(meta, ensure_ascii=False), dtype=object),
    )
    out_meta.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[OK] wrote {out_npz}")
    print(f"[OK] wrote {out_meta}")
    print(f"     exterior_image: shape={exterior.shape}, dtype={exterior.dtype}, color=RGB")
    print(f"     joint_position: {joint_7[:6].round(4).tolist()} (rad), gripper={int(gripper_on)}")


if __name__ == "__main__":
    main()

