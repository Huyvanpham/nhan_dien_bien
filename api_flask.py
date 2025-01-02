from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import cv2
import torch
import qrcode
import io
import os
from detect_vehicle import detect_vehicle_from_video
from lp_video import detect_license_plate
from datetime import datetime

app = Flask(__name__)
CORS(app)  # Cho phép CORS để giao tiếp với frontend

# Cơ sở dữ liệu tạm thời lưu thông tin xe
parking_lot = {}

# API: Khi xe vào bãi
@app.route('/api/vehicle-entry', methods=['POST'])
def vehicle_entry():
    video_file = request.files.get('video')
    if not video_file:
        return jsonify({"error": "No video file provided"}), 400

    video_path = "temp_video.mp4"
    video_file.save(video_path)

    # Gọi hàm phát hiện phương tiện và biển số
    vehicle_detected = False
    cap = cv2.VideoCapture(video_path)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Phát hiện phương tiện
        vehicles = detect_vehicle_from_video(frame)
        if vehicles:
            vehicle_detected = True
            for vehicle_crop in vehicles:
                # Nhận diện biển số
                lp = detect_license_plate(vehicle_crop)
                if lp != "unknown":
                    # Lưu biển số vào cơ sở dữ liệu
                    parking_lot[lp] = {
                        "entry_time": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                        "status": "IN"
                    }
                    return jsonify({"message": "Vehicle detected", "license_plate": lp}), 200

    if not vehicle_detected:
        return jsonify({"message": "No vehicles detected"}), 400

    cap.release()
    os.remove(video_path)  # Xóa video tạm thời
    return jsonify({"message": "No license plates detected"}), 400

# API: Khi xe ra bãi
@app.route('/api/vehicle-exit', methods=['POST'])
def vehicle_exit():
    data = request.json
    license_plate = data.get("license_plate")
    if not license_plate:
        return jsonify({"error": "License plate is required"}), 400

    if license_plate not in parking_lot:
        return jsonify({"error": "Vehicle not found"}), 404

    # Cập nhật trạng thái xe
    parking_lot[license_plate]["status"] = "OUT"
    parking_lot[license_plate]["exit_time"] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    # Sinh QR code
    qr_data = {
        "license_plate": license_plate,
        "entry_time": parking_lot[license_plate]["entry_time"],
        "exit_time": parking_lot[license_plate]["exit_time"],
        "status": parking_lot[license_plate]["status"]
    }
    qr_img = qrcode.make(qr_data)
    img_io = io.BytesIO()
    qr_img.save(img_io, 'PNG')
    img_io.seek(0)

    return send_file(img_io, mimetype='image/png', as_attachment=True, download_name=f'{license_plate}_qr.png')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
