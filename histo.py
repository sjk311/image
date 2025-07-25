import cv2
import numpy as np

def pixel_to_robot_coordinates(pixel_x, pixel_y):
    robot_x = (1 / 15) * pixel_y - (113 / 15)
    robot_y = (-19 / 286) * pixel_x + (12697 / 286)
    return robot_x, robot_y

# 원본 이미지 불러오기
img_path = r"C:\Users\SANG\Desktop\image\new\7.jpg"
image = cv2.imread(img_path)
if image is None:
    print("❌ 이미지 로드 실패!")
    exit()

# ROI 설정
x, y, w, h = 200, 10, 350, 380  # ← 여기가 ROI 영역
roi = image[y:y+h, x:x+w]

# # HSV 변환 및 히스토그램 평활화
# hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
# h_ch, s_ch, v_ch = cv2.split(hsv)
# v_eq = cv2.equalizeHist(v_ch)
# hsv_eq = cv2.merge((h_ch, s_ch, v_eq))

# 검정색 기준
lower_black = np.array([0, 0, 0])
upper_black = np.array([180, 220, 70])

# # 마스크 생성 및 블러링
# mask = cv2.inRange(hsv_eq, lower_black, upper_black)
# mask = cv2.medianBlur(mask, 5)

# HSV 변환 및 V 채널 평활화
hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
h, s, v = cv2.split(hsv)
v_eq = cv2.equalizeHist(v)
hsv_eq = cv2.merge((h, s, v_eq))

# inRange로 검정 객체 추출
mask = cv2.inRange(hsv_eq, lower_black, upper_black)

# 침식 연산으로 연결된 객체 분리
kernel = np.ones((5, 5), np.uint8)
mask = cv2.erode(mask, kernel, iterations=1)

# 선택: medianBlur로 경계 다듬기
mask = cv2.medianBlur(mask, 5)




# 컨투어 검출
contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
result_img = image.copy()
detected_objects = []

print(f"🔍 총 contour 개수: {len(contours)}")

for i, cnt in enumerate(contours):
    area = cv2.contourArea(cnt)
    rect = cv2.minAreaRect(cnt)
    (cx, cy), (w_box, h_box), angle = rect
    center_x, center_y = int(cx), int(cy)
    box = cv2.boxPoints(rect)
    box = np.intp(box)

    # ROI → 원본 좌표 보정
    center_x_global = center_x + x
    center_y_global = center_y + y
    box_global = box + np.array([[x, y]])

    color = (0, 0, 255)
    reason = ""

    if area < 1200:
        reason = "면적이 너무 작음"
    elif area > 4000:
        reason = "면적이 너무 큼"
    elif not (211 <= center_x_global <= 577 and 6 <= center_y_global <= 392):
        reason = "중심 좌표 범위 벗어남"
    elif w_box < 20:
        reason = "w 작음"
    elif w_box > 120:
        reason = "w 큼"
    elif h_box < 30:
        reason = "h 작음"
    elif h_box > 150:
        reason = "h 큼"
    else:
        color = (0, 255, 0)
        if w_box < h_box:
            angle += 90
        robot_x, robot_y = pixel_to_robot_coordinates(center_x_global, center_y_global)
        detected_objects.append({
            'id': i,
            'pixel_x': float(center_x_global),
            'pixel_y': float(center_y_global),
            'angle': float(angle),
            'robot_x': float(robot_x),
            'robot_y': float(robot_y)
        })
        reason = "✔️ 검출됨"

    cv2.drawContours(result_img, [box_global], 0, color, 2)
    cv2.circle(result_img, (center_x_global, center_y_global), 4, (0, 0, 255), -1)
    label = f"ID:{i}"
    cv2.putText(result_img, label, (center_x_global - 20, center_y_global - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    if reason == "✔️ 검출됨":
        print(f"[✔️] ID:{i}, 중심=({center_x_global}, {center_y_global}), 회전각={angle:.1f}°, h={h_box:.1f}, w={w_box:.1f}, area={area:.1f}, robot=({robot_x:.1f}, {robot_y:.1f})")
    else:
        print(f"[❌] ID:{i}, 이유: {reason}")

# 결과 출력
cv2.imshow("🖤 ROI 마스크", mask)
cv2.imshow("✅ 결과", result_img)
cv2.imwrite(r"C:\Users\SANG\Desktop\image\captured_images\final_result.jpg", result_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
