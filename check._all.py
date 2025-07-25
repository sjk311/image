import cv2
import numpy as np
import os

def pixel_to_robot_coordinates(pixel_x, pixel_y):
    robot_x = (1 / 15) * pixel_y - (113 / 15)
    robot_y = (-19 / 286) * pixel_x + (12697 / 286)
    return robot_x, robot_y

input_folder = r"C:\Users\SANG\Desktop\image\new"
roi_x, roi_y, roi_w, roi_h = 200, 10, 350, 380  # ROI 영역

for index in range(1, 80):
    filename = f"{index}.jpg"
    img_path = os.path.join(input_folder, filename)

    image = cv2.imread(img_path)
    if image is None:
        print(f"❌ 이미지 로드 실패: {img_path}")
        continue

    print(f"\n📷 처리 중: {filename}")

    # ROI 설정
    roi = image[roi_y:roi_y+roi_h, roi_x:roi_x+roi_w]

    # HSV 변환 및 V 채널 평활화
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    v_eq = cv2.equalizeHist(v)
    hsv_eq = cv2.merge((h, s, v_eq))

    # 검정색 범위 마스크
    lower_black = np.array([0, 0, 0])
    upper_black = np.array([180, 220, 70])
    mask = cv2.inRange(hsv_eq, lower_black, upper_black)

    # 침식 → 객체 분리
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.erode(mask, kernel, iterations=1)

    # 경계 다듬기
    mask = cv2.medianBlur(mask, 5)

    # 컨투어 검출
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    result_img = image.copy()

    print(f"🔍 총 contour 개수: {len(contours)}")

    for i, cnt in enumerate(contours):
        area = cv2.contourArea(cnt)
        rect = cv2.minAreaRect(cnt)
        (cx, cy), (w_box, h_box), angle = rect
        center_x, center_y = int(cx), int(cy)
        box = cv2.boxPoints(rect)
        box = np.intp(box)

        # ROI 기준 → 원본 이미지 기준으로 좌표 보정
        center_x_global = center_x + roi_x
        center_y_global = center_y + roi_y
        box_global = box + np.array([[roi_x, roi_y]])

        color = (0, 0, 255)
        reason = ""

        # 필터 조건
        if area < 1500:
            reason = "면적이 너무 작음"
        elif area > 10000:
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
            reason = "✔️ 검출됨"
            print(f"[✔️] ID:{i}, 중심=({center_x_global}, {center_y_global}), 회전각={angle:.1f}°, h={h_box:.1f}, w={w_box:.1f}, area={area:.1f}, robot=({robot_x:.1f}, {robot_y:.1f})")

        if reason != "✔️ 검출됨":
            print(f"[❌] ID:{i}, 이유: {reason}")

        # 시각화
        cv2.drawContours(result_img, [box_global], 0, color, 2)
        cv2.circle(result_img, (center_x_global, center_y_global), 4, (0, 0, 255), -1)
        label = f"ID:{i}"
        cv2.putText(result_img, label, (center_x_global - 20, center_y_global - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    # 결과 출력
    window_name_result = f"{filename} - Result"
    cv2.imshow(window_name_result, result_img)

    key = cv2.waitKey(0)
    if key == 27:  # ESC 키
        break

cv2.destroyAllWindows()
