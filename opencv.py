import cv2
import numpy as np

def pixel_to_robot_coordinates(pixel_x, pixel_y):
    robot_x = (1 / 15) * pixel_y - (113 / 15)
    robot_y = (-19 / 286) * pixel_x + (12697 / 286)
    return robot_x, robot_y

# 이미지 로드
img_path = r"C:\Users\SANG\Desktop\image\opencv_image\8.jpg"
image = cv2.imread(img_path)

if image is None:
    print("❌ 이미지 로드 실패")
    exit()

print(f"✅ 이미지 로드됨: {image.shape}")

# HSV 변환 + 마스크
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
lower_black = np.array([0, 0, 0])
upper_black = np.array([180, 255, 60])  # 밝기 조절 가능
mask = cv2.inRange(hsv, lower_black, upper_black)
mask = cv2.medianBlur(mask, 5)

# 마스크 확인용
cv2.imshow("🖤 마스크 결과", mask)


# contour 검출
contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
print(f"🔍 contour 수: {len(contours)}")

result_img = image.copy()
object_id = 0

for i, cnt in enumerate(contours):
    area = cv2.contourArea(cnt)
    rect = cv2.minAreaRect(cnt)
    (x, y), (w, h), angle = rect
    box = cv2.boxPoints(rect)
    box = np.intp(box)

    reason = None
    color = (0, 255, 0)

    if not (211 <= x <= 577 and 6 <= y <= 392):
        reason = "ROI 범위 밖"
    elif area < 3000 or area > 7000:
        reason = f"면적({area:.0f}) 범위 밖"
    elif w < 55 or w > 70 or h < 55 or h > 100:
        reason = f"크기({w:.1f},{h:.1f}) 범위 밖"
    elif min(w, h) / max(w, h) < 0.6:
        reason = "비율 < 0.6"

    if reason:
        color = (0, 0, 255)
        print(f"⛔ contour {i}: {reason}")
    else:
        if w < h:
            angle += 90
            w, h = h, w

        robot_x, robot_y = pixel_to_robot_coordinates(x, y)

        detected_object = {
            'id': object_id,
            'pixel_x': float(x),
            'pixel_y': float(y),
            'angle': float(angle),
            'depth_mm': None,
            'robot_x': float(robot_x),
            'robot_y': float(robot_y)
        }

        print(f"🟩 object {object_id}: {detected_object}")
        object_id += 1

        cv2.circle(result_img, (int(x), int(y)), 4, (0, 0, 255), -1)
        text = f"{detected_object['id']} ({int(x)}, {int(y)}) / {angle:.1f}°"
        text_pos = (int(x) + 10, int(y) + 10)
        cv2.putText(result_img, text, text_pos,
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3)
        cv2.putText(result_img, text, text_pos,
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

    cv2.drawContours(result_img, [box], 0, color, 2)

# 결과 확인 및 저장
cv2.imshow("✅ 인식 결과", result_img)

cv2.waitKey(0)
cv2.destroyAllWindows()
