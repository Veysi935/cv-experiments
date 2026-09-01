import cv2
import numpy as np
from ultralytics import YOLO

# YOLO Modelini Yükle
print("YOLOv8 Yükleniyor...")
yolo_model = YOLO('yolov8n.pt')

# ArUco Ayarları
dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_50)
parameters = cv2.aruco.DetectorParameters()
detector = cv2.aruco.ArucoDetector(dictionary, parameters)

# --- RADAR AYARLARI ---
RADAR_W, RADAR_H = 600, 600

dst_pts = np.array([
    [250, 400],
    [350, 400],
    [350, 500],
    [250, 500]
], dtype=np.float32)
# ----------------------

def point_to_segment_dist(p, a, b):
    p, a, b = np.array(p), np.array(a), np.array(b)
    l2 = np.sum((a - b)**2)
    if l2 == 0: return np.linalg.norm(p - a)
    t = max(0, min(1, np.dot(p - a, b - a) / l2))
    proj = a + t * (b - a)
    return np.linalg.norm(p - proj)

cap = cv2.VideoCapture(1) # Iriun kameran
print("Kuşbakışı Radar Sistemi Başlatıldı!")

while True:
    ret, frame = cap.read()
    if not ret: break
    
    
    radar_bg = np.zeros((RADAR_H, RADAR_W, 3), dtype=np.uint8)
    
    results = yolo_model(frame, classes=[0], verbose=False)
    corners, ids, rejected = detector.detectMarkers(frame)
    
    fence_active = False
    H_matrix = None
    radar_c1, radar_c2 = None, None
    
    if ids is not None and len(ids) >= 2:
        cv2.aruco.drawDetectedMarkers(frame, corners, ids)
        flat_ids = ids.flatten()
        sorted_indices = np.argsort(flat_ids)
        
        fence_idx_1 = sorted_indices[0]
        fence_idx_2 = sorted_indices[1]
        
        
        src_pts = corners[fence_idx_1][0].astype(np.float32)
        
        H_matrix = cv2.getPerspectiveTransform(src_pts, dst_pts)
        
        # Orijinal ekrandaki 2B merkez noktaları
        c1 = (int(np.mean(corners[fence_idx_1][0][:, 0])), int(np.mean(corners[fence_idx_1][0][:, 1])))
        c2 = (int(np.mean(corners[fence_idx_2][0][:, 0])), int(np.mean(corners[fence_idx_2][0][:, 1])))
        
        cv2.line(frame, c1, c2, (255, 0, 0), 2)
        
        
        pts_to_transform = np.array([[[c1[0], c1[1]], [c2[0], c2[1]]]], dtype=np.float32)
        try:
            transformed_pts = cv2.perspectiveTransform(pts_to_transform, H_matrix)
            radar_c1 = (int(transformed_pts[0][0][0]), int(transformed_pts[0][0][1]))
            radar_c2 = (int(transformed_pts[0][1][0]), int(transformed_pts[0][1][1]))
            
           
            cv2.line(radar_bg, radar_c1, radar_c2, (255, 0, 0), 4)
            
            cv2.circle(radar_bg, radar_c1, 15, (0, 255, 255), -1) 
            fence_active = True
        except:
            pass 
            
    violation = False

    for r in results:
        boxes = r.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            
          
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            
            px, py = int((x1 + x2) / 2), y2
            cv2.circle(frame, (px, py), 5, (0, 255, 0), -1)

           
            if fence_active and H_matrix is not None:
                feet_pts = np.array([[[px, py]]], dtype=np.float32)
                try:
                    feet_radar = cv2.perspectiveTransform(feet_pts, H_matrix)
                    rx, ry = int(feet_radar[0][0][0]), int(feet_radar[0][0][1])
                    
                   
                    cv2.circle(radar_bg, (rx, ry), 10, (0, 255, 0), -1)
                    
                   
                    dist_on_radar = point_to_segment_dist((rx, ry), radar_c1, radar_c2)
                    
                    
                    if dist_on_radar < 80:
                        violation = True
                        # İhlal varsa radar noktasını ve ana ekran kutusunu kırmızı yap
                        cv2.circle(radar_bg, (rx, ry), 10, (0, 0, 255), -1)
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 4)
                        cv2.putText(frame, "IHLAL!", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                except:
                    pass

    if violation and fence_active:
        cv2.line(frame, c1, c2, (0, 0, 255), 8) 
        cv2.line(radar_bg, radar_c1, radar_c2, (0, 0, 255), 8) # Radardaki çit de kırmızı olsun
        cv2.putText(frame, "!!! IHLAL !!!", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 4)

    # İki pencereyi yan yana izle
    cv2.imshow("Ana Kamera", frame)
    cv2.imshow("Kusbakisi Radar", radar_bg)

    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()