import argparse
import cv2
from detect_vehicle import detect_vehicle_from_video  # Gọi file detect_vehicle.py
from lp_video import detect_license_plate  # Gọi file lp_video.py

def main(video_path):
    """
    Hàm chính để nhận diện phương tiện và biển số từ video.
    :param video_path: Đường dẫn tới video đầu vào
    """
    print("Đang xử lý video...")
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("Không thể mở video.")
        return

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Không thể đọc frame từ video.")
            break

        # Phát hiện phương tiện trong khung hình
        vehicles = detect_vehicle_from_video(frame)

        if vehicles:
            for vehicle_crop in vehicles:
                # Nhận diện biển số từ từng phương tiện
                lp = detect_license_plate(vehicle_crop)
                if lp != "unknown":
                    print("Biển số xe:", lp)
                    cv2.putText(frame, lp, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)

        # Hiển thị kết quả
        resized_frame = cv2.resize(frame, (800, 800))  # Điều chỉnh kích thước hiển thị
        cv2.imshow('Video', resized_frame)

        # Nhấn 'q' để thoát
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Giải phóng tài nguyên
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # Sử dụng argparse để nhận tham số từ dòng lệnh
    parser = argparse.ArgumentParser(description="Phát hiện phương tiện và biển số từ video")
    parser.add_argument('-i', '--video', required=True, help="Đường dẫn tới video đầu vào")
    args = parser.parse_args()

    # Chạy hàm chính
    main(args.video)


