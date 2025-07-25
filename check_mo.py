import cv2
import numpy as np

def pixel_to_robot_coordinates(pixel_x, pixel_y):
    robot_x = (1 / 15) * pixel_y - (113 / 15)
    robot_y = (-19 / 286) * pixel_x + (12697 / 286)
    return robot_x, robot_y

# 이미지 경로
img_path = r"C:\Users\SANG\Desktop\image\captured_images\1.jpg"
image = cv2.imread(img_path)
if image is None:
    print("❌ 이미지 로드 실패!")
    exit()

# === ROI 설정: 네가 지정한 영역 + 여유 10픽셀
roi_x, roi_y, roi_w, roi_h = 173, 11, 455, 412
roi = image[roi_y:roi_y+roi_h, roi_x:roi_x+roi_w]

# === HSV 변환 및 마스크 처리 ===
hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
lower_black = np.array([0, 0, 0])
upper_black = np.array([180, 255, 50])
mask = cv2.inRange(hsv, lower_black, upper_black)
mask = cv2.medianBlur(mask, 5)

# === 외곽선 검출 ===
contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

result_img = image.copy()
detected_objects = []
print(f"🔍 ROI 내 contour 개수: {len(contours)}")

for i, cnt in enumerate(contours):
    area = cv2.contourArea(cnt)
    rect = cv2.minAreaRect(cnt)
    (x, y), (w, h), angle = rect

    # ROI 기준 → 전체 이미지 좌표계로 보정
    center_x = int(x + roi_x)
    center_y = int(y + roi_y)
    box = cv2.boxPoints(rect)
    box += np.array([roi_x, roi_y])
    box = np.intp(box)

    # 필터링
    color = (0, 0, 255)
    reason = ""
    if area < 1500:
        reason = "면적이 너무 작음"
    elif area > 7000:
        reason = "면적이 너무 큼"
    else:
        color = (0, 255, 0)
        if w < h:
            angle += 90
        robot_x, robot_y = pixel_to_robot_coordinates(center_x, center_y)
        detected_objects.append({
            'id': i,
            'pixel_x': float(center_x),
            'pixel_y': float(center_y),
            'angle': float(angle),
            'robot_x': float(robot_x),
            'robot_y': float(robot_y)
        })
        reason = "✔️ 검출됨"

    # 시각화
    cv2.drawContours(result_img, [box], 0, color, 2)
    cv2.circle(result_img, (center_x, center_y), 4, (0, 0, 255), -1)
    cv2.putText(result_img, f"ID:{i}", (center_x - 20, center_y - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    if reason == "✔️ 검출됨":
        print(f"[✔️] ID:{i}, 중심=({center_x}, {center_y}), 회전각={angle:.1f}°, robot=({robot_x:.1f}, {robot_y:.1f})")
    else:
        print(f"[❌] ID:{i}, 이유: {reason}")

# === 결과 시각화 ===
cv2.imshow("🔍 ROI 마스크", mask)
cv2.imshow("📦 결과", result_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
