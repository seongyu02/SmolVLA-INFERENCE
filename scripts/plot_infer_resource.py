#!/usr/bin/env python3
"""
infer_resource_benchmark.py 가 만든 JSON 을 읽어 막대그래프(PNG) 로 그린다.
시스템 python3(matplotlib 설치됨)으로 실행. (venv 에는 matplotlib 없음)

  GPU / CPU / RAM 평균 사용률(%) 막대 + 표준편차 에러바 + 값 라벨.
"""
import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")  # 헤드리스
import matplotlib.pyplot as plt


def main():
    ap = argparse.ArgumentParser()
    here = os.path.dirname(__file__)
    ap.add_argument("--json", default=os.path.join(here, "infer_resource_result.json"))
    ap.add_argument("--out", default=os.path.join(here, "infer_resource_bar.png"))
    args = ap.parse_args()

    with open(args.json) as f:
        d = json.load(f)

    m, s = d["means"], d["stds"]
    meta = d["meta"]
    labels = ["GPU", "CPU", "RAM"]
    keys = ["gpu", "cpu", "ram"]
    vals = [m[k] for k in keys]
    errs = [s[k] for k in keys]
    colors = ["#4C9BE8", "#E8A23C", "#5FB37A"]

    fig, ax = plt.subplots(figsize=(7, 5))
    bars = ax.bar(labels, vals, yerr=errs, capsize=8, color=colors,
                  edgecolor="black", linewidth=0.8, alpha=0.9)

    for bar, v, e in zip(bars, vals, errs):
        ax.text(bar.get_x() + bar.get_width() / 2, v + e + 1.5,
                f"{v:.1f}%", ha="center", va="bottom", fontsize=12, fontweight="bold")

    ax.set_ylabel("Usage (%)", fontsize=12)
    ax.set_ylim(0, 105)
    ax.yaxis.grid(True, linestyle="--", alpha=0.4)
    ax.set_axisbelow(True)

    lat = d.get("latency_ms", {})
    title = (f"SmolVLA inference resource usage  "
             f"(cond {meta.get('cond')}, mean of {meta.get('runs')} runs)")
    sub = (f"{meta.get('device','?')}  |  "
           f"latency {lat.get('mean',0):.0f}±{lat.get('std',0):.0f} ms  |  "
           f"RAM total {meta.get('total_ram_mb',0)/1024:.0f} GB, {meta.get('n_cpu','?')} CPU")
    ax.set_title(title + "\n" + sub, fontsize=11)

    fig.tight_layout()
    fig.savefig(args.out, dpi=150)
    print(f"[plot] saved -> {args.out}")


if __name__ == "__main__":
    main()
