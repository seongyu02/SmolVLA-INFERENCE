#!/usr/bin/env python3
"""
Dobot 로봇 통신 확인: Dashboard(29999), Motion(30004), Feedback(30005) 연결 테스트.

사용:
  cd ~/move-one
  python3 test_robot_connection.py [--robot_ip 192.168.5.1]
"""
import argparse
import os
import sys

DOBOT_SCRIPT_DIR = os.path.join(os.path.dirname(__file__), "dobot_ws", "src", "Dobot-Arm-DataCollect")
if os.path.isdir(DOBOT_SCRIPT_DIR):
    sys.path.insert(0, DOBOT_SCRIPT_DIR)


def main():
    parser = argparse.ArgumentParser(description="Dobot 통신 확인")
    parser.add_argument("--robot_ip", type=str, default="192.168.5.1", help="Dobot IP")
    args = parser.parse_args()

    print("=" * 60)
    print("  Dobot 로봇 통신 확인")
    print("=" * 60)
    print(f"  IP: {args.robot_ip}")
    print()

    try:
        from dobot_api import DobotApiDashboard, DobotApiFeedBack
    except ImportError as e:
        print(f"오류: dobot_api 로드 실패: {e}")
        print(f"  경로 확인: {DOBOT_SCRIPT_DIR}")
        sys.exit(1)

    ok_count = 0

    # 1) Dashboard (29999)
    try:
        dashboard = DobotApiDashboard(args.robot_ip, 29999)
        ret = dashboard.RobotMode()
        print(f"  [29999] Dashboard: OK (RobotMode 응답)")
        ok_count += 1
    except Exception as e:
        print(f"  [29999] Dashboard: 실패 - {e}")

    # 2) Motion (30004)
    try:
        move_api = DobotApiDashboard(args.robot_ip, 30004)
        # Motion 포트는 연결만 확인 (실제 이동 명령은 보내지 않음)
        print(f"  [30004] Motion: 연결됨 (OK)")
        ok_count += 1
    except Exception as e:
        print(f"  [30004] Motion: 실패 - {e}")

    # 3) Feedback (30005)
    try:
        feed = DobotApiFeedBack(args.robot_ip, 30005)
        data = feed.feedBackData()
        if data is not None:
            print(f"  [30005] Feedback: OK (피드백 수신, QActual 등 사용 가능)")
            ok_count += 1
        else:
            print(f"  [30005] Feedback: 연결됐으나 데이터 없음")
    except Exception as e:
        print(f"  [30005] Feedback: 실패 - {e}")

    print()
    if ok_count == 3:
        print("  결과: 3/3 포트 통신 정상. 로봇과 통신 잘 되고 있습니다.")
    else:
        print(f"  결과: {ok_count}/3 포트만 성공. 로봇 전원·네트워크·IP 확인하세요.")
    print("=" * 60)


if __name__ == "__main__":
    main()
