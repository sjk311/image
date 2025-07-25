import cv2
import numpy as np
import json

def pixel_to_robot_coordinates(pixel_x, pixel_y):
    robot_x = (1 / 15) * pixel_y - (113 / 15)
    robot_y = (-19 / 286) * pixel_x + (12697 / 286)
    return robot_x, robot_y

def get_angle(p1, p2):
    dx, dy = p2[0] - p1[0], p2[1] - p1[1]
    return np.degrees(np.arctan2(dy, dx))

img_path = r"C:\Users\SANG\Desktop\image\captured_images\image_20250722_165139.jpg"
image = cv2.imread(img_path)
if image is None:
    print("❌ 이미지 로드 실패! 경로 확인 필요")
    exit()
print(f"✅ 이미지 로드됨: {image.shape}")

hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
lower_black = np.array([0, 0, 0])
upper_black = np.array([180, 255, 50])
mask = cv2.inRange(hsv, lower_black, upper_black)
mask = cv2.medianBlur(mask, 5)

contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
print(f"🔍 검출된 contour 수: {len(contours)}")

result_img = image.copy()
detected_objects = []

object_id = 0
for cnt in contours:
    area = cv2.contourArea(cnt)
    if area < 3000 or area > 7000:
        continue

    rect = cv2.minAreaRect(cnt)
    (x, y), (w, h), _ = rect

    if not (211 <= x <= 577 and 6 <= y <= 392):
        continue
    if w < 55 or w > 70 or h < 55 or h > 100:
        continue
    ratio = min(w, h) / max(w, h)
    if ratio < 0.6:
        continue

    box = cv2.boxPoints(rect)
    box = np.intp(box)
    d1 = np.linalg.norm(box[0] - box[1])
    d2 = np.linalg.norm(box[1] - box[2])
    angle = get_angle(box[0], box[1]) if d1 < d2 else get_angle(box[1], box[2])
    if angle < 0:
        angle += 360

    robot_x, robot_y = pixel_to_robot_coordinates(x, y)

    detected_object = {
        'id': object_id,
        'pixel_x': float(x),
        'pixel_y': float(y),
        'angle': float(round(angle, 2)),
        'depth_mm': None,
        'robot_x': float(round(robot_x, 2)),
        'robot_y': float(round(robot_y, 2))
    }
    detected_objects.append(detected_object)
    object_id += 1

    cv2.drawContours(result_img, [box], 0, (0, 255, 0), 2)
    cv2.circle(result_img, (int(x), int(y)), 4, (0, 0, 255), -1)
    text = f"ID {detected_object['id']}: ({detected_object['robot_x']}, {detected_object['robot_y']}) / {detected_object['angle']}°"
    cv2.putText(result_img, text, (int(x)+10, int(y)+10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(result_img, text, (int(x)+10, int(y)+10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)

# 결과 이미지 저장
cv2.imwrite("/home/pi/final_result.jpg", result_img)

# JSON 형식으로 결과 출력
print("\n📦 Detected Objects:")
print(json.dumps(detected_objects, indent=4))
