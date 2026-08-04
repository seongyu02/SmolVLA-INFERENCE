# SmolVLA 추론 자원(GPU/CPU/RAM) 측정 & 그래프 — 사용 가이드

추론할 때 GPU/CPU/RAM 사용량을 재서 **10회 평균 막대그래프**로 뽑는 방법.
(Jetson AGX Orin, JetPack/CUDA, `.venv_SmolVLA310` 기준)

---

## 1. 무슨 프로그램으로 측정하나

| 항목 | 측정 도구 | 정확히 뭘 읽나 | 의미 |
|------|----------|---------------|------|
| **GPU** | `tegrastats` (NVIDIA Jetson **기본 내장** CLI) | 출력의 `GR3D_FREQ NN%` 필드 | GPU **연산 점유율**(얼마나 바쁜지). ※메모리 아님 |
| **CPU** | `psutil` (Python 라이브러리) | `psutil.cpu_percent()` | 8코어 **전체 사용률** 0~100% |
| **RAM** | `psutil` | `psutil.virtual_memory().percent` | 시스템 **메모리 사용량** 0~100% |
| **그래프** | `matplotlib` (Python 라이브러리) | `ax.bar(... yerr=표준편차)` | 막대그래프 + 에러바 PNG |

- **GPU**: `tegrastats`는 Jetson에 원래 깔려 있음(`/usr/bin/tegrastats`). 일반 데스크톱 GPU의
  `nvidia-smi` utilization이 Jetson(통합 GPU)에선 잘 안 나와서, Jetson 표준인 `tegrastats`의
  `GR3D_FREQ`(3D 엔진 부하)를 읽음.
- **CPU/RAM**: `psutil`은 측정용 venv(`.venv_SmolVLA310`)에 이미 설치돼 있음.
- **그래프**: `matplotlib`은 **시스템 python3**(`/usr/bin/python3`)에 설치돼 있음.
  (측정용 venv엔 matplotlib이 없어서, 측정과 그래프 단계를 일부러 분리함 — 아래 4번 참고)

> Orin은 CPU/GPU가 RAM을 공유하는 **통합 메모리**라 별도 VRAM이 없음.
> 그래서 "GPU 메모리"는 따로 안 재고, **RAM 막대 = 메모리 사용량**으로 봄.

---

## 2. 어떻게 평균을 내나 (측정 원리)

1. 백그라운드 스레드가 추론 내내 **~50ms마다** (GPU%, CPU%, RAM%)를 한 줄씩 기록.
   - GPU값은 `tegrastats`를 100ms 주기 서브프로세스로 돌려 최신 `GR3D_FREQ`를 계속 갱신.
2. 추론 **1회**가 일어날 때 그 구간 `[시작시각, 끝시각]` 안에 들어온 샘플들을 평균 → "추론 1회 사용량".
   - 추론 1회 ≈ 1.3초이므로 한 번에 약 25개 샘플이 잡혀서 평균이 안정적.
3. 이렇게 모은 **10회**를 다시 평균 → 막대 높이, 표준편차 → 에러바.
4. **모델 로딩 시간은 제외**됨(측정은 추론 함수 안에서만 일어나고, 로딩은 그 전에 끝남).
   첫 추론의 cold-start(cuDNN 오토튜닝 등)는 1회성으로 느려서, `profile_warmup`만큼 버리고
   그다음 N회만 측정함 → "진짜 정상상태 추론"만 평균.

---

## 3. 실행 방법 (2가지)

### 방법 A — 실제 노드에서 라이브 측정 (권장: 진짜 배포 상태)
카메라·로봇 다 켜진 실제 추론 중에 잼. ROS2 launch에 플래그만 추가.

```bash
source /opt/ros/humble/setup.bash
source ~/SmolVLA/SmolVLA-INFERENCE/ros2/install/setup.bash

ros2 launch e6_vla_ros smolvla_5cond.launch.py \
    cond:=5 \
    profile:=true \
    profile_warmup:=2
```

- `profile:=true` : 측정 ON (이거 없으면 측정 안 함)
- `profile_warmup:=2` : 앞 2회(cold-start) 버리고 그다음 10회 측정
- `profile_runs:=10` : 측정 횟수(기본 10)
- `profile_out:=/path/x.json` : 저장 경로(기본 auto)
- `cond:=1~5` : 조건 선택

로그에 `[profile] 1/10 ...` → `[profile] DONE 10회 평균 → GPU=.. CPU=.. RAM=..` 뜨면 완료.
결과: `scripts/infer_resource_live.json` + `scripts/infer_resource_live.png` (자동 생성)

### 방법 B — 독립 벤치마크 (로봇/카메라 없이 모델만)
모델 forward만 깨끗하게 비교하고 싶을 때. 더미 512×512 입력 사용.

```bash
cd ~/SmolVLA/SmolVLA-INFERENCE/scripts
./run_infer_benchmark.sh --cond 5            # cond 5, 10회
./run_infer_benchmark.sh --cond 5 --runs 20  # 20회
```
결과: `scripts/infer_resource_result.json` + `scripts/infer_resource_bar.png`

> A vs B 차이: 라이브(A)는 카메라 노드 등 다른 프로세스 부하까지 잡혀 **CPU가 더 높게** 나옴.
> GPU/RAM은 모델 forward가 본체라 둘이 거의 같음.

---

## 4. 파일 구성

| 파일 | 역할 | 실행 환경 |
|------|------|----------|
| `infer_resource_benchmark.py` | 방법 B 측정 본체(모델 로드+더미추론+샘플링) → JSON | venv(`.venv_SmolVLA310`) + CUDA |
| `plot_infer_resource.py` | JSON → 막대그래프 PNG (matplotlib) | **시스템** python3 |
| `run_infer_benchmark.sh` | 방법 B 한 방 실행기 (측정→그래프) | (둘 다 알아서 호출) |
| `smolvla_bridge_5cond_node.py` 안 `_ResourceProfiler` | 방법 A 측정(라이브 노드 내장) | venv + CUDA |

**왜 측정/그래프를 나눴나:** 측정은 torch+CUDA가 있는 venv에서 해야 하는데 그 venv엔 matplotlib이
없음. 반대로 시스템 python엔 matplotlib이 있지만 venv의 numpy(2.x)와 충돌함. 그래서
측정은 venv → JSON 저장, 그래프는 시스템 python으로 JSON→PNG. (라이브 노드도 그래프는
**깨끗한 env로 시스템 python**을 따로 띄워서 그림.)

---

## 5. 다른 사람한테 시킬 때 체크리스트

1. Jetson인지 확인 (`tegrastats` 있어야 함 → `which tegrastats`).
2. 측정 venv에 `psutil` 있는지 (`.venv_SmolVLA310`엔 이미 있음).
3. 시스템 python3에 `matplotlib` 있는지 (`/usr/bin/python3 -c "import matplotlib"`).
4. 방법 A면 ROS2 source 후 위 launch 명령, 방법 B면 `run_infer_benchmark.sh`.
5. 결과 PNG/JSON는 `scripts/` 폴더에 생성. 같은 이름에 덮어쓰니, 보관하려면 미리 복사.
