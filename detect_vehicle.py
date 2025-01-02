import cv2
import torch

# Tải mô hình YOLOv5 được huấn luyện trước để nhận diện ô tô
yolo_vehicle_detect = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
yolo_vehicle_detect.conf = 0.5  # Độ tự tin tối thiểu

def detect_vehicle_from_video(video_path, output_path):
    """
    Hàm phát hiện ô tô từ video và lưu video kết quả.
    :param video_path: Đường dẫn đến video đầu vào
    :param output_path: Đường dẫn lưu video đầu ra
    """
    # Đọc video đầu vào
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Không thể mở video.")
        return
    
    # Lấy thông tin video
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    # Ghi video đầu ra
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Dự đoán đối tượng trong khung hình
        results = yolo_vehicle_detect(frame)
        for _, row in results.pandas().xyxy[0].iterrows():
            if row['name'] == 'car':  # Lớp 'car' trong tập COCO
                x_min, y_min, x_max, y_max = int(row['xmin']), int(row['ymin']), int(row['xmax']), int(row['ymax'])
                # Vẽ khung hình chữ nhật quanh ô tô
                cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
                cv2.putText(frame, 'Car', (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Ghi khung hình vào video đầu ra
        out.write(frame)
        # Hiển thị khung hình (tuỳ chọn)
        cv2.imshow('Video', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()

# Sử dụng hàm
video_path = "video_input.mp4"  # Đường dẫn video đầu vào
output_path = "video_output.mp4"  # Đường dẫn lưu video đầu ra
detect_vehicle_from_video(video_path, output_path)

