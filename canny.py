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

# 1. 그레이스케일 변환
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
equalized = cv2.equalizeHist(gray)
blurred = cv2.GaussianBlur(equalized, (5, 5), 0)
edges = cv2.Canny(blurred, 50, 80)



# 3. Morphology 연산 (선택적 노이즈 제거)
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
edges = cv2.morphologyEx(edges, cv2.MORPH_OPEN, kernel)

# 4. 외곽선 찾기
contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# 5. 결과 시각화
result_img = image.copy()
detected_objects = []

print(f"🔍 총 contour 개수: {len(contours)}")

for i, cnt in enumerate(contours):
    area = cv2.contourArea(cnt)
    rect = cv2.minAreaRect(cnt)
    (x, y), (w, h), angle = rect
    center_x, center_y = int(x), int(y)
    box = cv2.boxPoints(rect)
    box = np.intp(box)

    color = (0, 0, 255)
    reason = ""

    # 조건 필터링
    if area < 800:
        reason = "면적이 너무 작음"
    elif area > 15000:
        reason = "면적이 너무 큼"
    elif not (211 <= x <= 577 and 6 <= y <= 392):
        reason = "중심 좌표 범위 벗어남"
    elif w < 20:
        reason = "w 작음"
    elif w > 120:
        reason = "w 큼"
    elif h < 30:
        reason = "h 작음"
    elif h > 130:
        reason = "h 큼"
    else:
        # 검출 성공
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
    label = f"ID:{i}"
    cv2.putText(result_img, label, (center_x - 20, center_y - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    if reason == "✔️ 검출됨":
        print(f"[✔️] ID:{i}, 중심=({center_x}, {center_y}), 회전각={angle:.1f}°, h={h:.1f}, w={w:.1f}, area={area:.1f}, robot=({robot_x:.1f}, {robot_y:.1f})")
    else:
        print(f"[❌] ID:{i}, 이유: {reason}")

# 결과 보기
cv2.imshow("🟦 엣지 마스크", edges)
cv2.imshow("✅ 결과", result_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
