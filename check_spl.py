import pyrealsense2 as rs
import numpy as np
import cv2
import os
from datetime import datetime
import time

def pixel_to_robot_coordinates(pixel_x, pixel_y):
    robot_x = (1 / 15) * pixel_y - (113 / 15)
    robot_y = (-19 / 286) * pixel_x + (12697 / 286)
    return robot_x, robot_y

# RealSense 설정
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
pipeline.start(config)
time.sleep(2)

print("📸 스페이스바를 누르면 1장 촬영 및 검출 수행 (ESC 종료)")

try:
    while True:
        key = cv2.waitKey(1)
        
        if key == 27:  # ESC
            print("🛑 종료됨")
            break

        elif key == 32:  # Spacebar
            frames = pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            if not color_frame:
                print("❌ 컬러 프레임 없음")
                continue

            color_image = np.asanyarray(color_frame.get_data())
            result_img = color_image.copy()

            # HSV 변환 및 마스크
            hsv = cv2.cvtColor(color_image, cv2.COLOR_BGR2HSV)
            lower_black = np.array([0, 0, 0])
            upper_black = np.array([180, 255, 80])
            mask = cv2.inRange(hsv, lower_black, upper_black)
            mask = cv2.erode(mask, np.ones((5, 5), np.uint8), iterations=2)
            mask = cv2.medianBlur(mask, 5)

            # 컨투어
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            print(f"\n📷 캡처됨 - contour 수: {len(contours)}")

            for i, cnt in enumerate(contours):
                area = cv2.contourArea(cnt)
                rect = cv2.minAreaRect(cnt)
                (x, y), (w, h), angle = rect
                center_x, center_y = int(x), int(y)
                box = cv2.boxPoints(rect)
                box = np.intp(box)

                color = (0, 0, 255)
                reason = ""

                if area < 4000:
                    reason = "면적 작음"
                elif area > 10000:
                    reason = "면적 큼"
                else:
                    color = (0, 255, 0)
                    if w < h:
                        angle += 90
                    robot_x, robot_y = pixel_to_robot_coordinates(center_x, center_y)
                    reason = "✔️ 검출됨"
                    print(f"[✔️] ID:{i}, 중심=({center_x},{center_y}), 회전각={angle:.1f}°, robot=({robot_x:.1f},{robot_y:.1f})")

                if reason != "✔️ 검출됨":
                    print(f"[❌] ID:{i}, 이유: {reason}")

                cv2.drawContours(result_img, [box], 0, color, 2)
                cv2.circle(result_img, (center_x, center_y), 4, (255, 0, 0), -1)
                cv2.putText(result_img, f"ID:{i}", (center_x - 20, center_y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            # 결과 출력    
            cv2.imshow("Mask", mask)
            cv2.imshow("Result", result_img)

finally:
    pipeline.stop()
    cv2.destroyAllWindows()
