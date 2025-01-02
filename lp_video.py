from PIL import Image
import cv2
import torch
import math 
import function.utils_rotate as utils_rotate
from IPython.display import display
import os
import time
import argparse
import function.helper as helper

# Chấp nhận đường dẫn video thay vì ảnh
ap = argparse.ArgumentParser()
ap.add_argument('-i', '--video', required=True, help='path to input video')
args = ap.parse_args()

# Tải mô hình YOLOv5 cho nhận diện biển số và OCR
yolo_LP_detect = torch.hub.load('yolov5', 'custom', path='model/LP_detector.pt', force_reload=True, source='local')
yolo_license_plate = torch.hub.load('yolov5', 'custom', path='model/LP_ocr.pt', force_reload=True, source='local')
yolo_license_plate.conf = 0.60

# Mở video
cap = cv2.VideoCapture(args.video)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Không thể đọc frame từ video.")
        break

    plates = yolo_LP_detect(frame, size=640)
    list_plates = plates.pandas().xyxy[0].values.tolist()
    list_read_plates = set()

    if len(list_plates) == 0:
        lp = helper.read_plate(yolo_license_plate, frame)
        if lp != "unknown":
            cv2.putText(frame, lp, (7, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)
            list_read_plates.add(lp)
    else:
        for plate in list_plates:
            flag = 0
            x = int(plate[0]) # xmin
            y = int(plate[1]) # ymin
            w = int(plate[2] - plate[0]) # xmax - xmin
            h = int(plate[3] - plate[1]) # ymax - ymin  
            crop_img = frame[y:y+h, x:x+w]
            cv2.rectangle(frame, (int(plate[0]),int(plate[1])), (int(plate[2]),int(plate[3])), color = (0,0,225), thickness = 2)
            cv2.imwrite("crop.jpg", crop_img)
            rc_image = cv2.imread("crop.jpg")
            lp = ""
            for cc in range(0,2):
                for ct in range(0,2):
                    lp = helper.read_plate(yolo_license_plate, utils_rotate.deskew(crop_img, cc, ct))
                    if lp != "unknown":
                        list_read_plates.add(lp)
                        cv2.putText(frame, lp, (int(plate[0]), int(plate[1]-10)), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)
                        flag = 1
                        break
                if flag == 1:
                    break
    
    # In biển số ra dạng văn bản
    if list_read_plates:
        print("Biển số xe vào bãi:", list(list_read_plates))
        
    # Hiển thị kết quả
    width, height = 800, 800  # Kích thước mong muốn
    resized_frame = cv2.resize(frame, (width, height))

    cv2.imshow('frame', resized_frame)

    # Nhấn 'q' để thoát khi xem video
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Giải phóng tài nguyên khi kết thúc
cap.release()
cv2.destroyAllWindows()
