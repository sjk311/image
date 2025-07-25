import pyrealsense2 as rs

ctx = rs.context()
devices = ctx.query_devices()

if len(devices) == 0:
    print("❌ RealSense 장치를 찾을 수 없습니다.")
else:
    print(f"✅ {len(devices)}개의 RealSense 장치가 연결되어 있습니다:")
    for dev in devices:
        print(f"- 이름: {dev.get_info(rs.camera_info.name)}")
        print(f"  직렬번호: {dev.get_info(rs.camera_info.serial_number)}")
