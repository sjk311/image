import cv2
import numpy as np

# ✅ 픽셀 → 로봇 좌표 변환 함수
def pixel_to_robot_coordinates(pixel_x, pixel_y):
    robot_x = (1 / 15) * pixel_y - (113 / 15)
    robot_y = (-19 / 286) * pixel_x + (12697 / 286)
    return robot_x, robot_y

# ✅ 이미지 로드
img_path = r"C:\Users\SANG\Desktop\image\captured_images\image_20250722_165139.jpg"
image = cv2.imread(img_path)

if image is None:
    print("❌ 이미지 로드 실패! 경로 확인 필요")
    exit()
print(f"✅ 이미지 로드됨: {image.shape}")

# ✅ HSV 변환 및 마스크
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
lower_black = np.array([0, 0, 0])
upper_black = np.array([180, 255, 50])
mask = cv2.inRange(hsv, lower_black, upper_black)
mask = cv2.medianBlur(mask, 5)
cv2.imshow("🖤 검정 마스크", mask)

# ✅ 윤곽선 찾기
contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
print(f"🔍 검출된 contour 수: {len(contours)}")

# ✅ 결과 이미지 준비
result_img = image.copy()

for i, cnt in enumerate(contours):
    area = cv2.contourArea(cnt)
    if area < 3000 or area > 7000:
        continue  # 면적 필터

    rect = cv2.minAreaRect(cnt)
    (x, y), (w, h), angle = rect

    if not (211 <= x <= 577 and 6 <= y <= 392):
        continue  # 중심 위치 필터

    if w < 55 or w > 70 or h < 55 or h > 100:
        continue  # 크기 필터

    ratio = min(w, h) / max(w, h)
    if ratio < 0.6:
        continue  # 비율 필터

    # ✅ X축 기준 회전각 보정
    box = cv2.boxPoints(rect)
    box = np.intp(box)

    # 가장 긴 변 계산
    def get_angle(p1, p2):
        dx, dy = p2[0] - p1[0], p2[1] - p1[1]
        return np.degrees(np.arctan2(dy, dx))

    d1 = np.linalg.norm(box[0] - box[1])
    d2 = np.linalg.norm(box[1] - box[2])
    angle_deg = get_angle(box[0], box[1]) if d1 >= d2 else get_angle(box[1], box[2])
    if angle_deg < 0:
        angle_deg += 360  # 항상 양수로

    # ✅ 로봇 좌표 변환
    robot_x, robot_y = pixel_to_robot_coordinates(x, y)

    # ✅ 로그 출력
    print(f"🟩 Object {i+1}: 중심=({x:.1f}, {y:.1f})px → 로봇=({robot_x:.1f}, {robot_y:.1f}), 회전각={angle_deg:.1f}°")

    # ✅ 시각화
    cv2.drawContours(result_img, [box], 0, (0, 255, 0), 2)
    cv2.circle(result_img, (int(x), int(y)), 4, (0, 0, 255), -1)

    text = f"({robot_x:.1f}, {robot_y:.1f}) / {angle_deg:.1f}°"
    text_pos = (int(x) + 10, int(y) + 10)

    # 외곽선 + 텍스트 출력
    cv2.putText(result_img, text, text_pos,
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(result_img, text, text_pos,
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)

# ✅ 결과 출력 및 저장
cv2.imshow("✅ 인식 결과", result_img)
cv2.imwrite(r"C:\Users\SANG\Desktop\image\captured_images\final_result.jpg", result_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
