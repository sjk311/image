import cv2
import numpy as np

# === 배경 제거기 생성 ===
bg_subtractor = cv2.createBackgroundSubtractorMOG2(history=1, varThreshold=25, detectShadows=False)

# === 배경 학습 ===
background = cv2.imread("C:/Users/SANG/Desktop/image/captured_images/background.jpg")
bg_subtractor.apply(background)  # 배경 모델 학습

# === 검출할 이미지 ===
frame = cv2.imread("C:/Users/SANG/Desktop/image/captured_images/object.jpg")
fg_mask = bg_subtractor.apply(frame)  # 전경 추출

# === 전처리 (노이즈 제거) ===
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))
fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)
fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel)

# === 컨투어 추출 ===
contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

for i, cnt in enumerate(contours):
    area = cv2.contourArea(cnt)
    if area < 1000:
        continue  # 너무 작으면 무시

    rect = cv2.minAreaRect(cnt)
    (x, y), (w, h), angle = rect
    box = cv2.boxPoints(rect)
    box = np.intp(box)

    cv2.drawContours(frame, [box], 0, (0,255,0), 2)
    cv2.circle(frame, (int(x), int(y)), 5, (0,0,255), -1)

    print(f"ID:{i}, 중심=({int(x)}, {int(y)}), 회전각={angle:.1f}°")

cv2.imshow("Foreground Mask", fg_mask)
cv2.imshow("Result", frame)
cv2.waitKey(0)
cv2.destroyAllWindows()
