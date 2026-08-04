#!/usr/bin/env python3
"""
camera_state_node — HIKRobot 카메라 + ZED 카메라 + Dobot feedBack 읽기

발행 토픽:
  /e6/camera/image        sensor_msgs/Image          18Hz  224x224 RGB  (HIK)
  /e6/camera/zed_image    sensor_msgs/Image          18Hz  224x224 RGB  (ZED left)
  /e6/camera/image_512    sensor_msgs/Image          18Hz  512x512 RGB  (HIK, SmolVLA용, pub_smolvla_images=true 시)
  /e6/camera/zed_image_512 sensor_msgs/Image         18Hz  512x512 RGB  (ZED, SmolVLA용, pub_smolvla_images=true 시)
  /e6/smolvla5/image_hik_512 sensor_msgs/Image       18Hz  512x512 RGB  (HIK→OBS_IMAGE_1, 5cond용, pub_5cond_images=true 시)
  /e6/smolvla5/image_zed_512 sensor_msgs/Image       18Hz  512x512 RGB  (ZED→OBS_IMAGE_2, 5cond용, pub_5cond_images=true 시)
  /e6/robot/state         std_msgs/Float32MultiArray 18Hz  [j1..j6 deg, gripper 0~1]
  /e6/robot/tcp           std_msgs/Float32MultiArray 18Hz  [tx,ty,tz,rx,ry,rz mm/deg]
  /e6/robot/tcp_z         std_msgs/Float32           18Hz  TCP Z (mm)

파라미터:
  robot_ip            (str,  default "192.168.5.1")
  dry_run             (bool, default False)  — 로봇 없이 더미 데이터
  no_camera           (bool, default False)  — 카메라 없이 검정 이미지
  pub_smolvla_images  (bool, default False)  — 512x512 SmolVLA용 이미지 추가 발행 (crop+rotate, dobot7d base 변환용)
  pub_5cond_images    (bool, default False)  — 5cond용 512 이미지 발행. 224 crop 프레임을
                       PIL LANCZOS로 512 업스케일 → Convert_SmolVLADataset_Orange_512.py 와
                       동일(학습=infer 일치). HIK→/e6/smolvla5/image_hik_512(OBS_IMAGE_1),
                       ZED→/e6/smolvla5/image_zed_512(OBS_IMAGE_2).
"""
from __future__ import annotations

import sys
import os
import time
from pathlib import Path

import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Float32MultiArray, Float32

# ── 경로 설정 ────────────────────────────────────────────────────────────────
def _find_repo_root() -> Path:
    for parent in Path(__file__).resolve().parents:
        if (parent / "hardware" / "dobot" / "dobot_api.py").exists():
            return parent
    raise RuntimeError("repo root (hardware/dobot/dobot_api.py) not found")

_REPO = _find_repo_root()
_HARDWARE = _REPO / "hardware"
_DOBOT_SDK = _HARDWARE / "dobot"
for _p in [str(_HARDWARE), str(_DOBOT_SDK)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)


def _numpy_to_image_msg(frame: np.ndarray) -> Image:
    msg = Image()
    msg.height = frame.shape[0]
    msg.width = frame.shape[1]
    msg.encoding = "rgb8"
    msg.is_bigendian = False
    msg.step = frame.shape[1] * 3
    msg.data = frame.tobytes()
    return msg


def _resize_like_training(frame: np.ndarray, size: int = 512) -> np.ndarray:
    """Resize the collected RGB frame to match the training conversion."""
    frame = np.asarray(frame, dtype=np.uint8)
    try:
        from PIL import Image as PILImage  # type: ignore
        pil = PILImage.fromarray(frame, mode="RGB")
        return np.asarray(
            pil.resize((size, size), PILImage.Resampling.LANCZOS),
            dtype=np.uint8,
        )
    except Exception:
        import cv2  # type: ignore
        return cv2.resize(frame, (size, size), interpolation=cv2.INTER_LANCZOS4).astype(np.uint8)


def _hik_direct512(camera) -> np.ndarray:
    """cond5_direct512용 HIK: 640×480 raw → 가로 480 중앙크롭(480×480 정사각) → LANCZOS 512.

    640×480을 바로 512×512로 늘이면 가로세로 비율이 찌그러지므로, 짧은 변(480) 기준
    중앙 정사각으로 먼저 잘라낸 뒤 512로 키운다(크롭이라기보다 비율 보정).
    """
    if camera is None:
        return np.zeros((512, 512, 3), dtype=np.uint8)
    try:
        raw = np.asarray(camera.get_raw640(), dtype=np.uint8)  # (480, 640, 3) RGB
        h, w = raw.shape[:2]
        side = min(h, w)                 # 480
        x0 = (w - side) // 2             # (640-480)//2 = 80
        y0 = (h - side) // 2             # 0
        sq = raw[y0:y0 + side, x0:x0 + side]   # 480×480
        return _resize_like_training(sq, 512)
    except Exception:
        return np.zeros((512, 512, 3), dtype=np.uint8)


class CameraStateNode(Node):

    def __init__(self):
        super().__init__("camera_state_node")

        # 파라미터
        self.declare_parameter("robot_ip", "192.168.5.1")
        self.declare_parameter("dry_run", False)
        self.declare_parameter("no_camera", False)
        self.declare_parameter("camera_black_mean", 8.0)
        self.declare_parameter("pub_smolvla_images", False)
        # 5cond(Orange-512) 추론용. 224 crop 프레임을 PIL LANCZOS로 512 업스케일해
        # /e6/smolvla5/image_hik_512(HIK→OBS_IMAGE_1), /e6/smolvla5/image_zed_512(ZED→OBS_IMAGE_2)로
        # 발행. Convert_SmolVLADataset_Orange_512.py(224→512 LANCZOS)와 동일 → 학습/추론 일치.
        self.declare_parameter("pub_5cond_images", False)
        # 5cond 이미지 생성 방식(pub_5cond_images=True일 때만 의미). 기본 False=학습 크롭 그대로.
        #   False: 224 학습 크롭(ZED [120:480,150:510], HIK [16:240,55:279]) → LANCZOS 512 (학습과 동일)
        #   True : 학습 크롭 없이 raw → 512 직접.
        #          ZED: 640×480 → 512 직접 리사이즈(no crop)
        #          HIK: 640×480 → 가로 480 중앙크롭(480×480 정사각) → 512  (640→512 직접 시 비율왜곡 방지)
        self.declare_parameter("cond5_direct512", False)
        # True면 크롭 없이 raw 전체를 바로 512×512로 리사이즈해서 발행 (/e6/camera/image,
        # /e6/camera/zed_image). 소비 노드는 msg 크기로 reshape하므로 512가 그대로 들어가고
        # 모델 resize_imgs_with_padding(512)는 no-op. False(기본)면 기존 크롭→224 그대로.
        self.declare_parameter("no_crop", False)

        robot_ip = self.get_parameter("robot_ip").value
        self._dry_run = self.get_parameter("dry_run").value
        self._no_camera = self.get_parameter("no_camera").value
        self._camera_black_mean = self.get_parameter("camera_black_mean").value
        self._pub_smolvla = self.get_parameter("pub_smolvla_images").value
        self._pub_5cond = self.get_parameter("pub_5cond_images").value
        self._cond5_direct512 = self.get_parameter("cond5_direct512").value
        self._last_zed_raw640 = None  # cond5_direct512용 ZED 640×480 raw 보관
        if self._pub_5cond and self._cond5_direct512:
            self.get_logger().warn(
                "cond5_direct512=True → 5cond 이미지를 학습 크롭 없이 raw→512로 발행 "
                "(ZED 640×480→512 직접, HIK 480 정사각→512). 학습 분포와 다름(실험용)."
            )
        self._no_crop = self.get_parameter("no_crop").value
        if self._no_crop:
            self.get_logger().warn(
                "no_crop=True → 크롭 없이 raw 전체를 512×512로 리사이즈 발행 "
                "(224 크롭 파이프라인 우회)"
            )

        # 퍼블리셔
        self._img_pub     = self.create_publisher(Image,             "/e6/camera/image",      10)
        self._zed_pub     = self.create_publisher(Image,             "/e6/camera/zed_image",  10)
        self._state_pub   = self.create_publisher(Float32MultiArray, "/e6/robot/state",       10)
        self._tcp_pub     = self.create_publisher(Float32MultiArray, "/e6/robot/tcp",         10)
        self._tcpz_pub    = self.create_publisher(Float32,           "/e6/robot/tcp_z",       10)
        if self._pub_smolvla:
            self._img512_pub = self.create_publisher(Image, "/e6/camera/image_512",     10)
            self._zed512_pub = self.create_publisher(Image, "/e6/camera/zed_image_512", 10)
        if self._pub_5cond:
            self._hik5_pub = self.create_publisher(Image, "/e6/smolvla5/image_hik_512", 10)
            self._zed5_pub = self.create_publisher(Image, "/e6/smolvla5/image_zed_512", 10)

        # 하드웨어 초기화
        self._feed = None
        self._camera = None
        self._zed = None
        self._zed_mat = None
        self._last_gripper = 0.0
        self._last_zed_frame: np.ndarray | None = None  # grab 실패 시 직전 유효 프레임 재사용

        if not self._dry_run:
            self._init_robot(robot_ip)
        if not self._no_camera:
            self._init_camera()
            self._init_zed()

        # executor_supervisor_node가 발행하는 명령 그리퍼 상태 구독
        # DigitalOutputs 비트 마스크 대신 명령 상태를 신뢰할 수 있는 출처로 사용
        self.create_subscription(Float32, "/e6/gripper/commanded",
                                 lambda msg: setattr(self, "_last_gripper", msg.data), 10)

        # 18Hz 타이머
        self.create_timer(1/18, self._tick)
        self.get_logger().info(
            f"camera_state_node 시작 — robot={'연결됨' if self._feed else 'dry_run'} "
            f"camera={'연결됨' if self._camera else 'dummy'}"
        )

    # ── 초기화 ──────────────────────────────────────────────────────────────

    def _init_robot(self, robot_ip: str):
        try:
            from dobot_api import DobotApiFeedBack  # type: ignore
            self._feed = DobotApiFeedBack(robot_ip, 30005)
            self.get_logger().info(f"Dobot feedBack 연결: {robot_ip}:30005")
        except Exception as exc:
            self.get_logger().warn(f"Dobot 연결 실패 ({exc}) → dry_run 모드")
            self._feed = None

    def _init_camera(self):
        try:
            import camera_capture  # type: ignore
            self._camera = camera_capture.CameraCapture()
            self.get_logger().info(f"HIK 카메라 초기화: {self._camera._name}")
        except Exception as exc:
            self.get_logger().warn(f"HIK 카메라 초기화 실패 ({exc}) → 더미 이미지")
            self._camera = None

    def _init_zed(self):
        try:
            import pyzed.sl as sl  # type: ignore
            zed = sl.Camera()
            init_params = sl.InitParameters()
            init_params.depth_mode = sl.DEPTH_MODE.NONE
            init_params.camera_resolution = sl.RESOLUTION.HD1080  # 학습 수집과 동일 해상도
            init_params.camera_fps = 30
            status = zed.open(init_params)
            if status != sl.ERROR_CODE.SUCCESS:
                self.get_logger().warn(f"ZED 카메라 오픈 실패: {status} → 더미 이미지")
                return
            self._zed = zed
            self._zed_mat = sl.Mat()
            self.get_logger().info(
                f"ZED 카메라 초기화: SN={zed.get_camera_information().serial_number}"
            )
        except Exception as exc:
            self.get_logger().warn(f"ZED 카메라 초기화 실패 ({exc}) → 더미 이미지")
            self._zed = None
            self._zed_mat = None

    # ── 18Hz 타이머 ─────────────────────────────────────────────────────────

    def _tick(self):
        now = self.get_clock().now().to_msg()

        # 1) HIK 이미지
        frame = self._read_frame()
        img_msg = _numpy_to_image_msg(frame)
        img_msg.header.stamp = now
        self._img_pub.publish(img_msg)

        # 2) ZED 이미지
        zed_frame = self._read_zed_frame()
        zed_msg = _numpy_to_image_msg(zed_frame)
        zed_msg.header.stamp = now
        self._zed_pub.publish(zed_msg)

        # 2) 로봇 상태
        deg6, tcp6, gripper = self._read_robot_state()
        state = np.array([*deg6, gripper], dtype=np.float32)
        self._state_pub.publish(Float32MultiArray(data=state.tolist()))

        # 3) TCP (6D) + TCP Z
        self._tcp_pub.publish(Float32MultiArray(data=tcp6.tolist()))
        self._tcpz_pub.publish(Float32(data=float(tcp6[2])))

        # 4) SmolVLA용 512x512 이미지 (pub_smolvla_images=true 시)
        if self._pub_smolvla:
            img512 = self._read_frame_512()
            msg512 = _numpy_to_image_msg(img512)
            msg512.header.stamp = now
            self._img512_pub.publish(msg512)

            zed512 = self._read_zed_frame_512()
            zed512_msg = _numpy_to_image_msg(zed512)
            zed512_msg.header.stamp = now
            self._zed512_pub.publish(zed512_msg)

        # 5) 5cond(Orange-512)용 이미지 (pub_5cond_images=true 시)
        if self._pub_5cond:
            if self._cond5_direct512:
                # [direct512 모드] 학습 크롭 없이 raw→512 (실험용)
                #   HIK: 640×480 → 가로 480 중앙크롭(480 정사각) → LANCZOS 512 (비율왜곡 방지)
                #   ZED: 640×480 raw → LANCZOS 512 직접
                hik5 = _hik_direct512(self._camera)
                zed5 = (
                    _resize_like_training(self._last_zed_raw640, 512)
                    if self._last_zed_raw640 is not None
                    else np.zeros((512, 512, 3), dtype=np.uint8)
                )
            else:
                # [기본] 224 학습 crop 프레임 → LANCZOS 512 (Convert_...512.py와 동일)
                hik5 = _resize_like_training(frame, 512)       # HIK → OBS_IMAGE_1
                zed5 = _resize_like_training(zed_frame, 512)   # ZED → OBS_IMAGE_2

            hik5_msg = _numpy_to_image_msg(hik5)
            hik5_msg.header.stamp = now
            self._hik5_pub.publish(hik5_msg)

            zed5_msg = _numpy_to_image_msg(zed5)
            zed5_msg.header.stamp = now
            self._zed5_pub.publish(zed5_msg)

    # ── 이미지 읽기 ─────────────────────────────────────────────────────────

    def _read_frame(self) -> np.ndarray:
        H = W = 224
        if self._camera is None:
            return np.zeros((H, W, 3), dtype=np.uint8)
        try:
            if self._no_crop:
                import cv2  # type: ignore
                raw = self._camera.get_raw640()  # (480, 640, 3), 크롭 전 raw
                return cv2.resize(raw, (512, 512), interpolation=cv2.INTER_AREA).astype(np.uint8)
            frame = self._camera.get_frame()
            if frame is not None:
                return np.asarray(frame, dtype=np.uint8)
        except Exception as exc:
            self.get_logger().warn(f"HIK 카메라 읽기 실패: {exc}", throttle_duration_sec=5.0)
        return np.zeros((512, 512, 3) if self._no_crop else (H, W, 3), dtype=np.uint8)

    def _read_zed_frame(self) -> np.ndarray:
        H = W = 224
        if self._zed is None or self._zed_mat is None:
            return self._last_zed_frame if self._last_zed_frame is not None else np.zeros((H, W, 3), dtype=np.uint8)
        try:
            import cv2  # type: ignore
            import pyzed.sl as sl  # type: ignore
            if self._zed.grab() == sl.ERROR_CODE.SUCCESS:
                self._zed.retrieve_image(self._zed_mat, sl.VIEW.LEFT)
                frame = self._zed_mat.get_data()[:, :, :3]  # BGRA → drop alpha
                frame = frame[:, :, ::-1].copy()            # BGR → RGB
                if self._no_crop:
                    # 크롭 없이 raw 전체를 바로 512×512로 리사이즈
                    self._last_zed_frame = cv2.resize(
                        frame, (512, 512), interpolation=cv2.INTER_AREA
                    ).astype(np.uint8)
                    return self._last_zed_frame
                # 학습 수집(robot_server.py)과 동일한 전처리
                # HD1080 → 640×480 → crop[120:480, 150:510] (360×360) → 224×224
                frame = cv2.resize(frame, (640, 480))
                if self._pub_5cond and self._cond5_direct512:
                    # cond5_direct512용: crop 전 640×480 raw 보관 (기본 동작엔 영향 없음)
                    self._last_zed_raw640 = frame.copy()
                frame = frame[120:480, 150:510]
                frame = cv2.resize(frame, (W, H))
                self._last_zed_frame = frame.astype(np.uint8)
                return self._last_zed_frame
        except Exception as exc:
            self.get_logger().warn(f"ZED 카메라 읽기 실패: {exc}", throttle_duration_sec=5.0)
        # grab 실패 시 직전 유효 프레임 재사용 (zeros 대신)
        return self._last_zed_frame if self._last_zed_frame is not None else np.zeros((H, W, 3), dtype=np.uint8)

    # ── 로봇 상태 읽기 ──────────────────────────────────────────────────────

    def _read_robot_state(self) -> tuple[np.ndarray, np.ndarray, float]:
        """(deg6, tcp6, gripper 0~1) 반환. 실패 시 이전값 유지."""
        deg6 = np.zeros(6, dtype=np.float32)
        tcp6 = np.zeros(6, dtype=np.float32)

        if self._feed is None:
            return deg6, tcp6, self._last_gripper

        try:
            fb = self._feed.feedBackData()
            if fb is not None and len(fb) > 0:
                deg6 = np.asarray(fb["QActual"][0], dtype=np.float32)[:6]
                tcp6 = np.asarray(fb["ToolVectorActual"][0], dtype=np.float32)[:6]
        except Exception as exc:
            self.get_logger().warn(f"feedBackData 실패: {exc}", throttle_duration_sec=5.0)

        return deg6, tcp6, self._last_gripper

    # ── SmolVLA용 512x512 이미지 ──────────────────────────────────────────────
    # HIK crop: [0:480, 94:574] → 480×480 → 512×512
    # ZED crop: [0:480, 80:560] → 480×480 → 512×512

    def _read_frame_512(self) -> np.ndarray:
        # training 변환(convert_dobot_to_lerobot)과 동일: raw640 → crop[0:480,94:574] → 512×512
        # 카메라가 90° CCW로 장착되어 있으므로 90° CW 보정 적용
        if self._camera is None:
            return np.zeros((512, 512, 3), dtype=np.uint8)
        try:
            import cv2  # type: ignore
            raw = self._camera.get_raw640()  # (480, 640, 3)
            crop = raw[0:480, 94:574]        # 480×480
            img = cv2.resize(crop, (512, 512), interpolation=cv2.INTER_AREA)
            img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
            return img.astype(np.uint8)
        except Exception as exc:
            self.get_logger().warn(f"HIK 512 읽기 실패: {exc}", throttle_duration_sec=5.0)
        return np.zeros((512, 512, 3), dtype=np.uint8)

    def _read_zed_frame_512(self) -> np.ndarray:
        # training 변환과 동일: ZED HD1080 → 640×480 → crop[0:480,80:560] → 512×512
        # HIK와 같은 마운트라면 ZED도 동일하게 90° CW 보정
        if self._zed is None or self._zed_mat is None:
            return np.zeros((512, 512, 3), dtype=np.uint8)
        try:
            import cv2  # type: ignore
            import pyzed.sl as sl  # type: ignore
            if self._zed.grab() == sl.ERROR_CODE.SUCCESS:
                self._zed.retrieve_image(self._zed_mat, sl.VIEW.LEFT)
                frame = self._zed_mat.get_data()[:, :, :3][:, :, ::-1].copy()
                frame = cv2.resize(frame, (640, 480))
                crop = frame[0:480, 80:560]
                img = cv2.resize(crop, (512, 512), interpolation=cv2.INTER_AREA)
                img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
                return img.astype(np.uint8)
        except Exception as exc:
            self.get_logger().warn(f"ZED 512 읽기 실패: {exc}", throttle_duration_sec=5.0)
        return np.zeros((512, 512, 3), dtype=np.uint8)

    def destroy_node(self):
        if self._zed is not None:
            try:
                self._zed.close()
            except Exception:
                pass
            self._zed = None
            self._zed_mat = None
        if self._camera is not None and hasattr(self._camera, "close"):
            try:
                self._camera.close()
            except Exception:
                pass
            self._camera = None
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = CameraStateNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        try:
            rclpy.shutdown()
        except Exception:
            pass


if __name__ == "__main__":
    main()
