import cv2
import numpy as np

def pixel_to_robot_coordinates(pixel_x, pixel_y):
    robot_x = (1 / 15) * pixel_y - (113 / 15)
    robot_y = (-19 / 286) * pixel_x + (12697 / 286)
    return robot_x, robot_y

# img_path = r"C:\Users\SANG\Desktop\image\opencv_image\4.jpg"
img_path = r"C:\Users\SANG\Desktop\image\captured_images\77.jpg"

image = cv2.imread(img_path)
if image is None:
    print("❌ 이미지 로드 실패!")
    exit()

hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# lower_black = np.array([0, 0, 0])
# upper_black = np.array([180, 200, 70])

# 노란색일때 사용
lower_black = np.array([13, 70, 90])
upper_black = np.array([30, 255, 255])


mask = cv2.inRange(hsv, lower_black, upper_black)
mask = cv2.medianBlur(mask, 5)

contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
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

    # 기본 표시 색: 빨간색 (검출 실패)
    color = (0, 0, 255)
    reason = ""

    # 조건 필터링
    if area < 1500:
        reason = "면적이 너무 작음"
    elif area > 7000:
        reason = "면적이 너무 큼"
    elif not (211 <= x <= 577 and 6 <= y <= 392):
        reason = "중심 좌표 범위 벗어남"
    elif w < 20 :
        reason = "w 작음"
    elif w > 120 :
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

        detected_object = {
            'id': i,
            'pixel_x': float(center_x),
            'pixel_y': float(center_y),
            'angle': float(angle),
            'robot_x': float(robot_x),
            'robot_y': float(robot_y)
        }
        detected_objects.append(detected_object)
        reason = "✔️ 검출됨"

    # contour 표시
    cv2.drawContours(result_img, [box], 0, color, 2)
    cv2.circle(result_img, (center_x, center_y), 4, (0, 0, 255), -1)
    label = f"ID:{i}"
    cv2.putText(result_img, label, (center_x - 20, center_y - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    # 로그 출력
    if reason == "✔️ 검출됨":
        print(f"[✔️] ID:{i}, 중심=({center_x}, {center_y}), 회전각={angle:.1f}°, h={h:.1f}, w={w:.1f}, area ={area:.1f}, robot=({robot_x:.1f}, {robot_y:.1f})")
    else:
        print(f"[❌] ID:{i}, 이유: {reason}")

# 결과 표시
cv2.imshow("🖤 마스크", mask)
cv2.imshow("✅ 결과", result_img)
cv2.imwrite(r"C:\Users\SANG\Desktop\image\captured_images\final_result.jpg", result_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
