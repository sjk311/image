import cv2

# 이미지 경로
img_path = r"C:\Users\SANG\Desktop\image\new\7.jpg"
image = cv2.imread(img_path)

# 마우스 콜백 함수 정의
def show_pixel_info(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        b, g, r = image[y, x]
        print(f"📍 좌표: ({x}, {y}), 색(BGR): ({b}, {g}, {r})")

# 창 생성 및 콜백 등록
cv2.namedWindow("Click to Get Pixel")
cv2.setMouseCallback("Click to Get Pixel", show_pixel_info)

# 이미지 보기
cv2.imshow("Click to Get Pixel", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
