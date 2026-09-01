import cv2
import numpy as np
import time
import requests
import threading
from ultralytics import YOLO


kalibrasyon_verisi = np.load("kamera_kalibrasyon.npz")
camera_matrix = kalibrasyon_verisi["camera_matrix"]
dist_coeffs = kalibrasyon_verisi["dist_coeffs"]

yolo_model = YOLO('yolov8n.pt') 

# === 2. TELEGRAM AYARLARI ===
import os
from dotenv import load_dotenv

load_dotenv()

TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

def send_telegram_photo(image_path, message):
    """Arka planda çalışarak fotoğrafı Telegram'a gönderir."""
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendPhoto"
    try:
        with open(image_path, "rb") as image_file:
            payload = {"chat_id": TELEGRAM_CHAT_ID, "caption": message}
            files = {"photo": image_file}
            response = requests.post(url, data=payload, files=files)
            if response.status_code == 200:
                print(">>> Telegram'a başarıyla gönderildi!")
            else:
                print(">>> Telegram Hatası:", response.text)
    except Exception as e:
        print(f">>> Gönderim Başarısız: {e}")
# ============================

marker_length = 0.05 
obj_points = np.array([
    [-marker_length / 2,  marker_length / 2, 0],
    [ marker_length / 2,  marker_length / 2, 0],
    [ marker_length / 2, -marker_length / 2, 0],
    [-marker_length / 2, -marker_length / 2, 0]
], dtype=np.float32)

dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_50)
parameters = cv2.aruco.DetectorParameters()
detector = cv2.aruco.ArucoDetector(dictionary, parameters)

def point_to_segment_dist(p, a, b):
    p, a, b = np.array(p), np.array(a), np.array(b)
    l2 = np.sum((a - b)**2)
    if l2 == 0: return np.linalg.norm(p - a)
    t = max(0, min(1, np.dot(p - a, b - a) / l2))
    proj = a + t * (b - a)
    return np.linalg.norm(p - proj)

cap = cv2.VideoCapture(1)
last_photo_time = 0
COOLDOWN_SECONDS = 5 #

print("Yapay Zeka Destekli IoT Güvenlik Sistemi Başlatıldı!")

while True:
    ret, frame = cap.read()
    if not ret: break
        
    results = yolo_model(frame, classes=[0], verbose=False)
    corners, ids, rejected = detector.detectMarkers(frame)
    
    fence_active = False
    c1, c2 = None, None
    
    if ids is not None and len(ids) >= 2:
        cv2.aruco.drawDetectedMarkers(frame, corners, ids)
        flat_ids = ids.flatten()
        sorted_indices = np.argsort(flat_ids)
        
        fence_idx_1 = sorted_indices[0]
        fence_idx_2 = sorted_indices[1]
        
        c1 = (int(np.mean(corners[fence_idx_1][0][:, 0])), int(np.mean(corners[fence_idx_1][0][:, 1])))
        c2 = (int(np.mean(corners[fence_idx_2][0][:, 0])), int(np.mean(corners[fence_idx_2][0][:, 1])))
        
        cv2.line(frame, c1, c2, (255, 0, 0), 4)
        fence_active = True

    violation = False

   
    for r in results:
        boxes = r.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            
            
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            
            if fence_active:
               
                rect = (x1, y1, x2 - x1, y2 - y1)
                
                
                intersects, pt1, pt2 = cv2.clipLine(rect, c1, c2)
                
                if intersects:
                    violation = True
                    
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 4)
                    cv2.putText(frame, "IHLAL YAPAN!", (x1, y1 - 10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)

    if violation and fence_active:
        cv2.line(frame, c1, c2, (0, 0, 255), 8) 
        cv2.putText(frame, "!!! IHLAL TESPIT EDILDI !!!", (50, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 4, cv2.LINE_AA)
        
        current_time = time.time()
        if current_time - last_photo_time > COOLDOWN_SECONDS:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename = f"ihlal_{timestamp}.jpg"
            cv2.imwrite(filename, frame)
            print(f"[ALARM] İhlal tespit edildi! Fotoğraf kaydedildi: {filename}")
            
            
            mesaj = f"🚨 DİKKAT! Güvenlik İhlali Tespit Edildi! 🚨\nSaat: {time.strftime('%H:%M:%S')}"
            
            
            t = threading.Thread(target=send_telegram_photo, args=(filename, mesaj))
            t.start()
            
            last_photo_time = current_time

    cv2.imshow("Yapay Zeka Guvenlik Sistemi", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()