#!/usr/bin/env python3
"""
SmolVLA 5cond 추론 자원 사용량 벤치마크 (GPU / CPU / RAM %).

smolvla_bridge_5cond_node.py 와 동일한 방식으로 모델을 1회 로드하고,
동일한 추론 파이프라인(preprocessor -> predict_action_chunk -> postprocessor)을
N회(기본 10회) 반복하면서 매 추론 구간의 자원 사용량을 샘플링한다.

  - GPU 사용률 : tegrastats 의 GR3D_FREQ (Jetson GPU utilization %)
  - CPU 사용률 : psutil.cpu_percent (시스템 전체, 0~100%)
  - RAM 사용률 : psutil.virtual_memory().percent (시스템 전체, 0~100%)

각 추론 구간 [t_start, t_end] 안에 들어온 샘플을 평균 -> 추론 1회 값.
그 값을 N회에 대해 평균/표준편차 내어 JSON 으로 저장한다(그래프는 plot 스크립트가 그림).

반드시 run_infer_benchmark.sh (= venv_SmolVLA310 + CUDA 환경) 으로 실행할 것.
"""

import argparse
import json
import os
import re
import subprocess
import threading
import time
from pathlib import Path

import numpy as np
import psutil
import torch

# ── 노드와 동일한 상수/경로 ──────────────────────────────────────────────────
STATE_DIM = 7
SAFETENSORS_SINGLE_FILE = "model.safetensors"
CKPT_ROOT = "/media/billye6/새 볼륨1/SmolVLA/smolvla_5cond"
DEFAULT_BASE = "/media/billye6/새 볼륨1/SmolVLA/SmolVLA_base_dobot7d_local/pretrained_model"
COND_DIRS = {
    "1": "exp1_expert_only",
    "2": "exp2_vision_lora_0_10",
    "3": "exp3_vision_lora_6_10",
    "4": "exp4_nolr_vision_lora_0_10",
    "5": "exp5_nolr_vision_lora_6_10",
}
PROMPT_DIRECTIONAL = "pick up the orange box from the left side and place it on the right side"
PROMPT_NOLR = "pick up the orange box and place it on the other side"


# ── tegrastats(GPU) 리더 ─────────────────────────────────────────────────────
class TegraReader(threading.Thread):
    """tegrastats 를 백그라운드로 돌리며 최신 GR3D_FREQ(%) 를 self.gpu 에 갱신."""

    _GR3D = re.compile(r"GR3D_FREQ\s+(\d+)%")

    def __init__(self, interval_ms: int = 100):
        super().__init__(daemon=True)
        self.gpu = 0.0
        self._stop = False
        self.proc = subprocess.Popen(
            ["tegrastats", "--interval", str(interval_ms)],
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True, bufsize=1,
        )

    def run(self):
        for line in self.proc.stdout:
            if self._stop:
                break
            m = self._GR3D.search(line)
            if m:
                self.gpu = float(m.group(1))

    def stop(self):
        self._stop = True
        try:
            self.proc.terminate()
        except Exception:
            pass


# ── 자원 샘플러 (GPU/CPU/RAM 을 일정 주기로 기록) ────────────────────────────
class Sampler(threading.Thread):
    def __init__(self, tegra: TegraReader, interval: float = 0.05):
        super().__init__(daemon=True)
        self.tegra = tegra
        self.interval = interval
        self.samples = []  # (t, gpu, cpu, ram)
        self.lock = threading.Lock()
        self._stop = False

    def run(self):
        psutil.cpu_percent(None)  # cpu_percent prime (첫 호출은 0 반환)
        time.sleep(self.interval)
        while not self._stop:
            t = time.monotonic()
            cpu = psutil.cpu_percent(None)
            ram = psutil.virtual_memory().percent
            gpu = self.tegra.gpu
            with self.lock:
                self.samples.append((t, gpu, cpu, ram))
            time.sleep(self.interval)

    def stop(self):
        self._stop = True

    def window_mean(self, t0: float, t1: float):
        """[t0, t1] 안의 샘플 평균. 샘플이 없으면 t1 에 가장 가까운 샘플 1개."""
        with self.lock:
            arr = list(self.samples)
        inside = [s for s in arr if t0 <= s[0] <= t1]
        if not inside:
            if not arr:
                return None
            inside = [min(arr, key=lambda s: abs(s[0] - t1))]
        a = np.array([[s[1], s[2], s[3]] for s in inside], dtype=np.float64)
        return a.mean(axis=0), len(inside)  # [gpu,cpu,ram], n_samples


# ── 노드의 _rgb_uint8_to_tensor 동일 ─────────────────────────────────────────
def rgb_uint8_to_tensor(arr: np.ndarray, device) -> torch.Tensor:
    t = torch.from_numpy(arr.astype(np.float32) / 255.0)
    return t.permute(2, 0, 1).unsqueeze(0).to(device)


def resolve_model_path(cond: str, model_path: str) -> str:
    mp = (model_path or "").strip()
    if mp:
        p = Path(mp)
        return str(p / "pretrained_model") if (p / "pretrained_model").is_dir() else str(p)
    sub = COND_DIRS.get(cond)
    if not sub:
        raise SystemExit(f"cond={cond!r} 은 1~5 가 아니며 --model_path 도 없습니다.")
    return os.path.join(CKPT_ROOT, sub, "checkpoints", "020000", "pretrained_model")


def resolve_base(override: str, adapter_base: str) -> str:
    def ok(p):
        d = Path(p)
        return bool(p) and d.is_dir() and (d / "config.json").is_file() and (d / SAFETENSORS_SINGLE_FILE).is_file()
    if ok(override):
        return override
    for cand in (adapter_base, (adapter_base or "").replace("/media/billy/", "/media/billye6/")):
        if ok(cand):
            return cand
    raise FileNotFoundError("로컬 dobot7d base 를 찾지 못했습니다 (--base_model_path 지정 필요).")


def load_model(model_path: str, base_override: str, device):
    """smolvla_bridge_5cond_node._load_model 과 동일한 로직."""
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
    from lerobot.processor import PolicyProcessorPipeline
    from lerobot.processor.converters import (
        batch_to_transition, policy_action_to_transition,
        transition_to_batch, transition_to_policy_action,
    )
    from lerobot.utils.constants import (
        POLICY_POSTPROCESSOR_DEFAULT_NAME, POLICY_PREPROCESSOR_DEFAULT_NAME,
    )

    model_dir = Path(model_path)
    full_w = model_dir / SAFETENSORS_SINGLE_FILE
    adapter_w = model_dir / "adapter_model.safetensors"

    if full_w.is_file():
        model = SmolVLAPolicy.from_pretrained(model_path)
        print(f"[bench] loaded merged/full checkpoint: {model_path}")
    elif adapter_w.is_file():
        from peft import PeftConfig, PeftModel
        from lerobot.configs.policies import PreTrainedConfig
        peft_cfg = PeftConfig.from_pretrained(model_path)
        base_path = resolve_base(base_override, peft_cfg.base_model_name_or_path)
        cmp_cfg = PreTrainedConfig.from_pretrained(model_path)
        print(f"[bench] base={base_path}")
        print(f"[bench] adapter={model_path} (chunk={cmp_cfg.chunk_size} n_action_steps={cmp_cfg.n_action_steps})")
        base_model = SmolVLAPolicy.from_pretrained(base_path, config=cmp_cfg, strict=False)
        model = PeftModel.from_pretrained(base_model, model_path, config=peft_cfg, is_trainable=False)
    else:
        raise FileNotFoundError(f"No {SAFETENSORS_SINGLE_FILE}/adapter in {model_path}")

    model.to(device)
    model.eval()
    if hasattr(model, "reset"):
        model.reset()

    preprocessor = PolicyProcessorPipeline.from_pretrained(
        model_path, config_filename=f"{POLICY_PREPROCESSOR_DEFAULT_NAME}.json",
        to_transition=batch_to_transition, to_output=transition_to_batch,
    )
    postprocessor = PolicyProcessorPipeline.from_pretrained(
        model_path, config_filename=f"{POLICY_POSTPROCESSOR_DEFAULT_NAME}.json",
        to_transition=policy_action_to_transition, to_output=transition_to_policy_action,
    )
    n_action_steps = int(model.config.n_action_steps)
    return model, preprocessor, postprocessor, n_action_steps


def run_once(model, preprocessor, postprocessor, n_action_steps, prompt, device):
    """노드 _run_infer 의 연산부와 동일 (publish 제외). 더미 512x512 입력 사용."""
    hik = np.random.randint(0, 256, (512, 512, 3), dtype=np.uint8)
    zed = np.random.randint(0, 256, (512, 512, 3), dtype=np.uint8)
    state7 = np.random.randn(STATE_DIM).astype(np.float32)

    batch = {
        "observation.images.OBS_IMAGE_1": rgb_uint8_to_tensor(hik, device),
        "observation.images.OBS_IMAGE_2": rgb_uint8_to_tensor(zed, device),
        "observation.state": torch.from_numpy(state7).unsqueeze(0).to(device),
        "task": prompt,
    }
    with torch.no_grad():
        observation = preprocessor(batch)
        action_chunk = model.predict_action_chunk(observation)
        if action_chunk.ndim == 2:
            action_chunk = action_chunk.unsqueeze(0)
        action_chunk = action_chunk[:, :n_action_steps, :]
        steps = action_chunk.shape[1]
        processed = [postprocessor(action_chunk[:, i, :]) for i in range(steps)]
        actions_t = torch.stack(processed, dim=1).squeeze(0)
    _ = actions_t.detach().float().cpu().numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cond", default="1", help="1~5 (model_path 없을 때 사용)")
    ap.add_argument("--model_path", default="", help="체크포인트 직접 지정 (cond 무시)")
    ap.add_argument("--base_model_path", default=DEFAULT_BASE)
    ap.add_argument("--prompt", default="", help="비우면 cond 기본 프롬프트")
    ap.add_argument("--runs", type=int, default=10)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--sample-interval", type=float, default=0.05, help="샘플링 주기(초)")
    ap.add_argument("--tegra-interval", type=int, default=100, help="tegrastats 주기(ms)")
    ap.add_argument("--gap", type=float, default=0.0, help="추론 사이 대기(초). 0=back-to-back")
    ap.add_argument("--out-json", default=os.path.join(os.path.dirname(__file__), "infer_resource_result.json"))
    args = ap.parse_args()

    model_path = resolve_model_path(args.cond, args.model_path)
    prompt = args.prompt or (PROMPT_NOLR if args.cond in ("4", "5") else PROMPT_DIRECTIONAL)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[bench] device={device} cond={args.cond} runs={args.runs} warmup={args.warmup}")
    print(f"[bench] model_path={model_path}")
    print(f"[bench] prompt={prompt!r}")

    t_load0 = time.monotonic()
    model, pre, post, n_steps = load_model(model_path, args.base_model_path, device)
    print(f"[bench] model ready ({time.monotonic() - t_load0:.1f}s) n_action_steps={n_steps}")

    # 샘플러 시작
    tegra = TegraReader(args.tegra_interval)
    tegra.start()
    sampler = Sampler(tegra, args.sample_interval)
    sampler.start()
    time.sleep(0.5)  # 샘플러/리더 안정화

    # 워밍업 (첫 추론은 cudnn/flow-matching 초기화로 느림 -> 측정 제외)
    print(f"[bench] warmup x{args.warmup} ...")
    for _ in range(args.warmup):
        run_once(model, pre, post, n_steps, prompt, device)
    if device.type == "cuda":
        torch.cuda.synchronize()

    # 측정
    print(f"[bench] measuring x{args.runs} ...")
    windows, latencies = [], []
    for i in range(args.runs):
        t0 = time.monotonic()
        run_once(model, pre, post, n_steps, prompt, device)
        if device.type == "cuda":
            torch.cuda.synchronize()
        t1 = time.monotonic()
        windows.append((t0, t1))
        latencies.append((t1 - t0) * 1000.0)
        if args.gap > 0:
            time.sleep(args.gap)

    sampler.stop()
    tegra.stop()

    # 구간별 평균
    per_run = []  # {gpu,cpu,ram,latency_ms,n_samples}
    for (t0, t1), lat in zip(windows, latencies):
        res = sampler.window_mean(t0, t1)
        if res is None:
            continue
        vals, n = res
        per_run.append({
            "gpu": float(vals[0]), "cpu": float(vals[1]), "ram": float(vals[2]),
            "latency_ms": float(lat), "n_samples": int(n),
        })

    def stat(key):
        a = np.array([r[key] for r in per_run], dtype=np.float64)
        return float(a.mean()), float(a.std())

    means = {k: stat(k)[0] for k in ("gpu", "cpu", "ram")}
    stds = {k: stat(k)[1] for k in ("gpu", "cpu", "ram")}
    lat_mean, lat_std = stat("latency_ms")

    total_ram_mb = psutil.virtual_memory().total / (1024 ** 2)
    result = {
        "meta": {
            "device": torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu",
            "cond": args.cond, "model_path": model_path, "prompt": prompt,
            "runs": args.runs, "warmup": args.warmup,
            "sample_interval_s": args.sample_interval, "tegra_interval_ms": args.tegra_interval,
            "n_action_steps": n_steps, "total_ram_mb": round(total_ram_mb, 1),
            "n_cpu": psutil.cpu_count(),
        },
        "means": means, "stds": stds,
        "latency_ms": {"mean": lat_mean, "std": lat_std},
        "per_run": per_run,
    }
    with open(args.out_json, "w") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    # 콘솔 표
    print("\n" + "=" * 60)
    print(f"{'run':>4} {'GPU%':>8} {'CPU%':>8} {'RAM%':>8} {'lat(ms)':>9} {'#smp':>5}")
    print("-" * 60)
    for i, r in enumerate(per_run, 1):
        print(f"{i:>4} {r['gpu']:>8.1f} {r['cpu']:>8.1f} {r['ram']:>8.1f} {r['latency_ms']:>9.1f} {r['n_samples']:>5}")
    print("-" * 60)
    print(f"{'mean':>4} {means['gpu']:>8.1f} {means['cpu']:>8.1f} {means['ram']:>8.1f} {lat_mean:>9.1f}")
    print(f"{'std':>4} {stds['gpu']:>8.1f} {stds['cpu']:>8.1f} {stds['ram']:>8.1f} {lat_std:>9.1f}")
    print("=" * 60)
    print(f"[bench] JSON saved -> {args.out_json}")


if __name__ == "__main__":
    main()
