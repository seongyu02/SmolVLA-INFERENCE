#!/usr/bin/env python3
"""
20Hz 카메라 캡처: HIKRobot(MVS) 시도 후 OpenCV USB 폴백.
Jetson: MVCAM_COMMON_RUNENV 설정 (예: export MVCAM_COMMON_RUNENV=/opt/MVS/lib).
"""
import os
import sys
import time

IMG_H, IMG_W = 224, 224

# Jetson/Linux MVS 기본 경로
if sys.platform != "win32" and not os.environ.get("MVCAM_COMMON_RUNENV"):
    for _p in ("/opt/MVS/lib", "/opt/MVS/lib64"):
        if os.path.isdir(_p):
            os.environ["MVCAM_COMMON_RUNENV"] = _p
            break

DOBOT_SCRIPT_DIR = os.path.join(os.path.dirname(__file__), "dobot_ws", "src", "Dobot-Arm-DataCollect")


class CameraCapture:
    """단일 카메라 → 224x224 RGB. HIKRobot 우선 시도, 실패 시 OpenCV."""

    def __init__(self, use_hikrobot=True):
        self._cam = None
        self._use_cv2 = False
        self._last_frame = None
        self._name = "none"

        if use_hikrobot and self._init_hikrobot():
            self._name = "hikrobot"
            return
        if self._init_opencv():
            self._name = "opencv"
            return
        self._cam = None

    def _init_hikrobot(self):
        try:
            if DOBOT_SCRIPT_DIR not in sys.path:
                sys.path.insert(0, DOBOT_SCRIPT_DIR)
            from ctypes import byref, sizeof, memset, cast, POINTER, c_ubyte
            from MvImport.MvCameraControl_class import (
                MvCamera,
                MV_CC_DEVICE_INFO_LIST,
                MV_CC_DEVICE_INFO,
                MV_GIGE_DEVICE,
                MV_USB_DEVICE,
                MV_TRIGGER_MODE_OFF,
                MV_FRAME_OUT_INFO_EX,
                PixelType_Gvsp_Mono8,
                PixelType_Gvsp_RGB8_Packed,
                PixelType_Gvsp_BGR8_Packed,
                PixelType_Gvsp_BayerRG8,
                PixelType_Gvsp_BayerGR8,
                PixelType_Gvsp_BayerGB8,
                PixelType_Gvsp_BayerBG8,
            )
            import numpy as np
            import cv2

            # 기존 카메라 점유 프로세스는 run_robot_inference.py에서 정리됨

            ret = MvCamera.MV_CC_Initialize()
            if ret != 0:
                print(f"  [camera_capture] HIKRobot MV_CC_Initialize 실패: 0x{ret:x}")
                return False
            deviceList = MV_CC_DEVICE_INFO_LIST()
            ret = MvCamera.MV_CC_EnumDevices(MV_GIGE_DEVICE | MV_USB_DEVICE, deviceList)
            if ret != 0:
                print(f"  [camera_capture] HIKRobot MV_CC_EnumDevices 실패: 0x{ret:x}")
                return False
            if deviceList.nDeviceNum == 0:
                print(f"  [camera_capture] HIKRobot 카메라 없음 (nDeviceNum=0)")
                return False
            print(f"  [camera_capture] HIKRobot 카메라 발견: {deviceList.nDeviceNum}개")

            self._cam = MvCamera()
            stDeviceInfo = cast(deviceList.pDeviceInfo[0], POINTER(MV_CC_DEVICE_INFO)).contents
            ret = self._cam.MV_CC_CreateHandle(stDeviceInfo)
            if ret != 0:
                print(f"  [camera_capture] HIKRobot MV_CC_CreateHandle 실패: 0x{ret:x}")
                self._cam = None
                return False
            ret = self._cam.MV_CC_OpenDevice()
            if ret != 0:
                print(f"  [camera_capture] HIKRobot MV_CC_OpenDevice 실패: 0x{ret:x}")
                self._cam = None
                return False
            # 원본 camera_calibration_hikrobot.py와 동일한 순서로 설정
            self._cam.MV_CC_SetEnumValue("TriggerMode", MV_TRIGGER_MODE_OFF)
            try:
                self._cam.MV_CC_SetBoolValue("AcquisitionFrameRateEnable", True)
                self._cam.MV_CC_SetFloatValue("AcquisitionFrameRate", 10.0)  # 원본과 동일: 10.0
            except Exception:
                pass
            # 수동 노출/게인 고정 (Auto OFF 후 수동값 설정 - 학습 데이터와 동일한 밝기 유지)
            try:
                self._cam.MV_CC_SetEnumValue("ExposureAuto", 0)   # 0 = Off (수동)
                self._cam.MV_CC_SetEnumValue("GainAuto", 0)       # 0 = Off (수동)
            except Exception:
                pass
            try:
                self._cam.MV_CC_SetFloatValue("ExposureTime", 20000.0)  # 20ms
                self._cam.MV_CC_SetFloatValue("Gain", 10.0)             # 10 dB
            except Exception:
                pass
            # 스트리밍 시작
            ret = self._cam.MV_CC_StartGrabbing()
            if ret != 0:
                print(f"  [camera_capture] HIKRobot MV_CC_StartGrabbing 실패: 0x{ret:x}")
                try:
                    self._cam.MV_CC_CloseDevice()
                except Exception:
                    pass
                self._cam = None
                return False
            # 스트리밍 시작 후 노출 수렴 대기 (원본 camera_calibration_hikrobot.py는 대기 없지만, 프레임 읽기 실패 시 대기 필요)
            import time
            time.sleep(2.0)  # 노출 수렴 대기
            print(f"  [camera_capture] HIKRobot 초기화 성공 (스트리밍 시작 후 2초 대기 완료)")
            return True
        except Exception as e:
            print(f"  [camera_capture] HIKRobot 초기화 예외: {e}")
            import traceback
            traceback.print_exc()
            self._cam = None
            return False

    def _init_opencv(self):
        try:
            import cv2
            self._cam = cv2.VideoCapture(0)
            if self._cam.isOpened():
                self._cam.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                self._cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                self._cam.set(cv2.CAP_PROP_FPS, 20)
                self._use_cv2 = True
                return True
            self._cam.release()
            self._cam = None
        except Exception:
            pass
        return False

    def _decode_hik_to_640x480(self):
        """HIK SDK에서 프레임을 읽어 640×480 RGB로 디코딩. 실패 시 None 반환."""
        from ctypes import byref, sizeof, memset, c_ubyte
        from MvImport.MvCameraControl_class import (
            MV_FRAME_OUT_INFO_EX,
            PixelType_Gvsp_Mono8,
            PixelType_Gvsp_RGB8_Packed,
            PixelType_Gvsp_BGR8_Packed,
            PixelType_Gvsp_BayerRG8, PixelType_Gvsp_BayerGR8,
            PixelType_Gvsp_BayerGB8, PixelType_Gvsp_BayerBG8,
        )
        import cv2
        import numpy as np

        buf_size = 2448 * 2048 * 3
        pData = (c_ubyte * buf_size)()
        stFrameInfo = MV_FRAME_OUT_INFO_EX()
        memset(byref(stFrameInfo), 0, sizeof(stFrameInfo))
        ret = None
        for attempt in range(5):
            timeout_ms = 2000 + attempt * 500
            ret = self._cam.MV_CC_GetOneFrameTimeout(pData, buf_size, stFrameInfo, timeout_ms)
            if ret == 0:
                break
            if attempt < 4:
                time.sleep(0.2)

        if ret != 0:
            if not hasattr(self, "_frame_error_printed"):
                print(f"  [camera_capture] 프레임 읽기 실패 (5회 시도): 0x{ret:x}")
                self._frame_error_printed = True
            return None

        image_data = np.frombuffer(pData, dtype=np.uint8, count=stFrameInfo.nFrameLen)
        h, w = stFrameInfo.nHeight, stFrameInfo.nWidth
        pt = stFrameInfo.enPixelType

        if not hasattr(self, "_debug_pixel_type_printed"):
            try:
                print(f"  [camera_capture] pixelType=0x{int(pt):x}, h={h}, w={w}")
            except Exception:
                pass
            self._debug_pixel_type_printed = True

        if pt == PixelType_Gvsp_RGB8_Packed:
            img = image_data.reshape(h, w, 3).copy()
        elif pt == PixelType_Gvsp_BGR8_Packed:
            img = image_data.reshape(h, w, 3)[:, :, ::-1].copy()
        elif pt == PixelType_Gvsp_Mono8:
            img = cv2.cvtColor(image_data.reshape(h, w), cv2.COLOR_GRAY2RGB)
        elif pt == PixelType_Gvsp_BayerRG8:
            img = cv2.cvtColor(image_data.reshape(h, w), cv2.COLOR_BayerRG2RGB)
        elif pt == PixelType_Gvsp_BayerGR8:
            img = cv2.cvtColor(image_data.reshape(h, w), cv2.COLOR_BayerGB2RGB)
        elif pt == PixelType_Gvsp_BayerGB8:
            img = cv2.cvtColor(image_data.reshape(h, w), cv2.COLOR_BayerGB2RGB)
        elif pt == PixelType_Gvsp_BayerBG8:
            img = cv2.cvtColor(image_data.reshape(h, w), cv2.COLOR_BayerBG2RGB)
        else:
            if image_data.size == h * w:
                img = cv2.cvtColor(image_data.reshape(h, w), cv2.COLOR_GRAY2RGB)
            else:
                img = image_data.reshape(h, w, -1)
                if img.shape[-1] == 3:
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                else:
                    img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

        return cv2.resize(img, (640, 480), interpolation=cv2.INTER_LINEAR)

    def get_frame(self):
        """(224, 224, 3) uint8 RGB. 실패 시 이전 프레임 또는 검은 화면."""
        import numpy as np
        try:
            if self._cam is None:
                return np.zeros((IMG_H, IMG_W, 3), dtype=np.uint8)

            if self._use_cv2:
                import cv2
                ok, frame = self._cam.read()
                if not ok or frame is None:
                    if self._last_frame is not None:
                        return self._last_frame
                    return np.zeros((IMG_H, IMG_W, 3), dtype=np.uint8)
                frame = np.asarray(frame)
                if frame.ndim == 2:
                    frame = np.stack([frame] * 3, axis=-1)
                elif frame.shape[-1] == 3:
                    frame = frame[:, :, ::-1].copy()  # BGR → RGB
                import cv2
                frame = cv2.resize(frame, (IMG_W, IMG_H), interpolation=cv2.INTER_LINEAR)
                frame = np.asarray(frame, dtype=np.uint8)
                self._last_frame = frame
                return frame

            # HIKRobot
            import cv2
            img = self._decode_hik_to_640x480()
            if img is None:
                if self._last_frame is not None:
                    return self._last_frame
                return np.zeros((IMG_H, IMG_W, 3), dtype=np.uint8)

            # 학습 수집(robot_server.py)과 동일한 전처리
            # Native → 640×480 → 320×240 → crop[16:240, 55:279] → 224×224
            img = cv2.resize(img, (320, 240), interpolation=cv2.INTER_LINEAR)
            img = img[16:240, 55:279]
            img = np.asarray(img, dtype=np.uint8)
            self._last_frame = img
            return img
        except Exception:
            if hasattr(self, "_last_frame") and self._last_frame is not None:
                return self._last_frame
            return np.zeros((IMG_H, IMG_W, 3), dtype=np.uint8)

    def get_raw640(self):
        """(480, 640, 3) uint8 RGB. 학습 변환(convert_dobot_to_lerobot)과 동일한 640×480 중간 프레임."""
        import numpy as np
        try:
            if self._cam is None:
                return np.zeros((480, 640, 3), dtype=np.uint8)

            if self._use_cv2:
                import cv2
                ok, frame = self._cam.read()
                if not ok or frame is None:
                    return np.zeros((480, 640, 3), dtype=np.uint8)
                frame = np.asarray(frame)
                if frame.ndim == 2:
                    frame = np.stack([frame] * 3, axis=-1)
                elif frame.shape[-1] == 3:
                    frame = frame[:, :, ::-1].copy()  # BGR → RGB
                import cv2
                return np.asarray(cv2.resize(frame, (640, 480), interpolation=cv2.INTER_LINEAR), dtype=np.uint8)

            # HIKRobot
            img = self._decode_hik_to_640x480()
            if img is None:
                return np.zeros((480, 640, 3), dtype=np.uint8)
            return np.asarray(img, dtype=np.uint8)
        except Exception:
            return np.zeros((480, 640, 3), dtype=np.uint8)

    def close(self):
        if self._cam is None:
            return
        try:
            if self._use_cv2:
                self._cam.release()
            else:
                self._cam.MV_CC_StopGrabbing()
                self._cam.MV_CC_CloseDevice()
        except Exception:
            pass
        self._cam = None
