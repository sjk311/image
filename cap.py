import pyrealsense2 as rs
import numpy as np
import cv2
import os
import time

# 저장 폴더
save_dir = "new"
os.makedirs(save_dir, exist_ok=True)

# 📌 현재 폴더에 존재하는 숫자 파일 중 가장 큰 번호 찾기
def get_next_index(path):
    existing = [f for f in os.listdir(path) if f.endswith(".jpg")]
    indices = []
    for f in existing:
        name = os.path.splitext(f)[0]  # 확장자 제거
        if name.isdigit():
            indices.append(int(name))
    return max(indices) + 1 if indices else 1

# RealSense 파이프라인
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

pipeline.start(config)
time.sleep(2)

try:
    print("📸 프레임 대기 중...")
    frames = pipeline.wait_for_frames()
    color_frame = frames.get_color_frame()

    if not color_frame:
        raise RuntimeError("❌ 컬러 프레임을 가져올 수 없습니다.")

    color_image = np.asanyarray(color_frame.get_data())

    # 저장할 숫자 파일명 결정
    next_index = get_next_index(save_dir)
    filename = os.path.join(save_dir, f"{next_index}.jpg")

    # 저장
    cv2.imwrite(filename, color_image)
    print(f"✅ 이미지 저장 완료: {filename}")

    # 시각화
    cv2.imshow("Captured", color_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

finally:
    pipeline.stop()
