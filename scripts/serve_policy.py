#!/usr/bin/env python3
"""SmolVLA 모델 로드 확인 래퍼 (run_server.sh에서 호출).

openpi WebSocket 서버를 띄우지 않고 SmolVLA 모델 경로/로드 여부만 확인합니다.
실제 추론은 run_client.sh → run_smolvla_client.py 단독으로 동작합니다.
"""
from __future__ import annotations

import argparse
from pathlib import Path


def _resolve_model_path(raw_path: str) -> Path:
    path = Path(raw_path).expanduser()
    if (path / "pretrained_model").is_dir():
        return path / "pretrained_model"
    return path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="SmolVLA 모델 경로 확인 (서버 미시작)"
    )
    parser.add_argument("--port", type=int, default=8000,
                        help="호환성 유지용 (미사용)")
    parser.add_argument("--model-path", "--policy-dir", dest="model_path",
                        default="",
                        help="SmolVLA pretrained_model 경로 또는 상위 디렉터리")
    parser.add_argument("--check-only", action="store_true",
                        help="모델 로드 후 종료")
    args, unknown = parser.parse_known_args()

    model_path_raw = args.model_path
    if not model_path_raw:
        for i, tok in enumerate(unknown):
            if tok in ("--policy.dir", "--policy_dir") and i + 1 < len(unknown):
                model_path_raw = unknown[i + 1]
                break

    if not model_path_raw:
        raise SystemExit("[ERROR] 모델 경로 없음. --model-path <PATH> 사용")

    model_path = _resolve_model_path(model_path_raw)
    if not model_path.exists():
        raise SystemExit(f"[ERROR] 모델 경로 없음: {model_path}")

    print("============================================")
    print(" SmolVLA 정책 래퍼")
    print(f" model path : {model_path}")
    print(f" port       : {args.port} (호환성 유지용)")
    print("============================================")

    if args.check_only:
        import torch  # pylint: disable=import-outside-toplevel
        from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy  # pylint: disable=import-outside-toplevel

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        policy = SmolVLAPolicy.from_pretrained(str(model_path))
        policy.to(device)
        policy.eval()
        print(f"[OK] SmolVLA 로드 완료 on {device}")
        return

    print("[INFO] SmolVLA 모드에서는 WebSocket 서버를 시작하지 않습니다.")
    print("[INFO] 추론은 run_client.sh 단독으로 실행하세요.")


if __name__ == "__main__":
    main()
