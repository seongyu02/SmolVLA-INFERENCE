#!/usr/bin/env python3
"""
voice_command_node — 마이크 음성 또는 텍스트 입력 → ROS2 명령 토픽 변환

발행 토픽:
  /e6/task/voice_command        std_msgs/String  — task_node로 전달할 prompt
  /e6/supervisor/voice_override std_msgs/String  — executor로 전달할 STOP 명령

구독 토픽:
  /e6/voice/text_input          std_msgs/String  — STT 없이 직접 텍스트 주입 (use_mic=false 또는 보조 입력)

파라미터:
  use_mic              (bool,  default true)   마이크 캡처 활성화
  model_size           (str,   default "base") faster-whisper 모델 크기 (tiny/base/small/medium)
  language             (str,   default "ko")   STT 언어
  sample_rate          (int,   default 16000)  마이크 샘플링 레이트
  vad_min_amplitude    (float, default 0.02)   음성 감지 최소 RMS 진폭
  silence_duration_sec (float, default 1.5)    이 시간 이상 침묵이면 발화 종료 판정
  device_index         (int,   default -1)     마이크 장치 인덱스 (-1=기본값)
"""
from __future__ import annotations

import queue
import threading

import numpy as np
import rclpy
from rclpy.node import Node
from std_msgs.msg import String

# ── STOP 키워드 ──────────────────────────────────────────────────────────────
_STOP_WORDS = frozenset([
    "멈춰", "멈춰요", "멈추", "멈춤", "정지", "그만", "중지", "스탑",
    "stop", "halt", "freeze", "emergency",
])

# ── 명령 분류 + 한국어 → 영어 prompt 변환 ────────────────────────────────────

_PICK_WORDS  = ["집어", "잡아", "들어", "픽", "pick"]
_LEFT_WORDS  = ["왼쪽", "왼", "left", "레프트"]
_RIGHT_WORDS = ["오른쪽", "오른", "right", "라이트"]

# 한국어 단순 매핑 (phase prompt가 아닌 task-level 명령)
_KO_TO_PROMPT: list[tuple[list[str], list[str], str]] = [
    # (pick_hint, side_hint, prompt)
    (_PICK_WORDS, _LEFT_WORDS,  "pick up the orange box from the left side and place it on the right side"),
    (_PICK_WORDS, _RIGHT_WORDS, "pick up the orange box from the right side and place it on the left side"),
    (_PICK_WORDS, [],           "pick up the orange box"),
]


def _classify(text: str) -> tuple[str, str | None]:
    """
    Returns:
        ("stop", None)         — STOP 명령
        ("task", prompt_str)   — 작업 명령 (prompt 변환 완료)
        ("unknown", None)      — 인식 불가
    """
    t = text.strip().lower()

    # STOP 판단
    for w in _STOP_WORDS:
        if w in t:
            return ("stop", None)

    # 영어 prompt 직접 passthrough (pick/place/move/approach 로 시작)
    for prefix in ("pick", "place", "move", "approach", "lift", "transport", "release"):
        if t.startswith(prefix):
            return ("task", text.strip())

    # 한국어 → 영어 변환
    for pick_hints, side_hints, prompt in _KO_TO_PROMPT:
        has_pick = any(w in t for w in pick_hints)
        if not has_pick:
            continue
        if not side_hints:
            return ("task", prompt)
        has_side = any(w in t for w in side_hints)
        if has_side:
            return ("task", prompt)

    return ("unknown", None)


# ── ROS2 노드 ────────────────────────────────────────────────────────────────

class VoiceCommandNode(Node):

    def __init__(self):
        super().__init__("voice_command_node")

        self.declare_parameter("use_mic",              True)
        self.declare_parameter("model_size",           "base")
        self.declare_parameter("language",             "ko")
        self.declare_parameter("sample_rate",          16000)
        self.declare_parameter("vad_min_amplitude",    0.02)
        self.declare_parameter("silence_duration_sec", 1.5)
        self.declare_parameter("device_index",         -1)

        use_mic       = self.get_parameter("use_mic").value
        model_size    = self.get_parameter("model_size").value
        self._lang    = self.get_parameter("language").value
        self._sr      = self.get_parameter("sample_rate").value
        self._vad     = self.get_parameter("vad_min_amplitude").value
        self._sil_dur = self.get_parameter("silence_duration_sec").value
        dev_idx       = self.get_parameter("device_index").value

        # 발행
        self._task_pub = self.create_publisher(String, "/e6/task/voice_command",        10)
        self._stop_pub = self.create_publisher(String, "/e6/supervisor/voice_override", 10)

        # 텍스트 직접 주입 (마이크 없이 테스트할 때)
        self.create_subscription(String, "/e6/voice/text_input", self._cb_text, 10)

        if use_mic:
            self._start_mic(model_size, dev_idx)
        else:
            self.get_logger().info(
                "voice_command_node 시작 (마이크 비활성) — "
                "'ros2 topic pub /e6/voice/text_input std_msgs/msg/String "
                "\"data: 왼쪽 박스 집어줘\"' 로 명령 입력"
            )

    # ── 마이크 초기화 ─────────────────────────────────────────────────────────

    def _start_mic(self, model_size: str, dev_idx: int):
        try:
            import sounddevice as sd  # noqa: PLC0415
        except ImportError:
            self.get_logger().error("sounddevice 패키지 없음 — pip3 install sounddevice")
            return
        try:
            from faster_whisper import WhisperModel  # noqa: PLC0415
        except ImportError:
            self.get_logger().error("faster-whisper 패키지 없음 — pip3 install faster-whisper")
            return

        self.get_logger().info(f"Whisper 모델 로딩 ({model_size}) …")
        self._model = WhisperModel(model_size, device="cpu", compute_type="int8")
        self.get_logger().info("Whisper 모델 로딩 완료")

        self._audio_q: queue.Queue = queue.Queue()
        device = dev_idx if dev_idx >= 0 else None

        blocksize = int(self._sr * 0.1)  # 100ms 청크
        self._stream = sd.InputStream(
            samplerate=self._sr,
            channels=1,
            dtype="float32",
            blocksize=blocksize,
            device=device,
            callback=self._audio_cb,
        )
        self._stream.start()

        t = threading.Thread(target=self._mic_loop, daemon=True)
        t.start()

        self.get_logger().info(
            f"마이크 음성 감지 시작 — "
            f"vad_threshold={self._vad:.3f} silence={self._sil_dur:.1f}s"
        )

    def _audio_cb(self, indata, frames, time_info, status):  # noqa: ARG002
        self._audio_q.put(indata.copy())

    # ── 음성 감지 루프 (백그라운드 스레드) ───────────────────────────────────

    def _mic_loop(self):
        chunk_sec    = 0.1
        sil_chunks   = int(self._sil_dur / chunk_sec)

        recording    = False
        buf: list    = []
        silent_cnt   = 0

        while rclpy.ok():
            try:
                chunk = self._audio_q.get(timeout=0.5)
            except queue.Empty:
                continue

            rms      = float(np.sqrt(np.mean(chunk ** 2)))
            is_voice = rms > self._vad

            if is_voice:
                if not recording:
                    recording  = True
                    buf        = []
                    silent_cnt = 0
                    self.get_logger().info("음성 감지 시작 …")
                buf.append(chunk)
                silent_cnt = 0
            elif recording:
                buf.append(chunk)
                silent_cnt += 1
                if silent_cnt >= sil_chunks:
                    recording = False
                    audio = np.concatenate(buf, axis=0).flatten()
                    self._transcribe(audio)
                    buf = []

    # ── STT ──────────────────────────────────────────────────────────────────

    def _transcribe(self, audio: np.ndarray):
        self.get_logger().info("STT 처리 중 …")
        try:
            segments, _ = self._model.transcribe(
                audio,
                language=self._lang,
                beam_size=5,
                vad_filter=True,
            )
            text = " ".join(s.text for s in segments).strip()
        except Exception as exc:
            self.get_logger().error(f"STT 오류: {exc}")
            return

        if not text:
            return

        self.get_logger().info(f"STT 결과: '{text}'")
        self._process(text)

    # ── 텍스트 직접 주입 콜백 ────────────────────────────────────────────────

    def _cb_text(self, msg: String):
        self.get_logger().info(f"텍스트 입력: '{msg.data}'")
        self._process(msg.data)

    # ── 분류 후 발행 ─────────────────────────────────────────────────────────

    def _process(self, text: str):
        kind, prompt = _classify(text)

        if kind == "stop":
            self.get_logger().warn(f"[STOP] '{text}' → /e6/supervisor/voice_override STOP 발행")
            self._stop_pub.publish(String(data="STOP"))

        elif kind == "task" and prompt:
            self.get_logger().info(f"[TASK] '{text}' → '{prompt}'")
            self._task_pub.publish(String(data=prompt))

        else:
            self.get_logger().warn(f"[UNKNOWN] '{text}' — 인식된 명령 없음 (무시)")


def main(args=None):
    rclpy.init(args=args)
    node = VoiceCommandNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
