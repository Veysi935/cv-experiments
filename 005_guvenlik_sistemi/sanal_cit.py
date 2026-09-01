import cv2
import numpy as np
import math


kalibrasyon_verisi = np.load("kamera_kalibrasyon.npz")
camera_matrix = kalibrasyon_verisi["camera_matrix"]
dist_coeffs = kalibrasyon_verisi["dist_coeffs"]


marker_length = 0.05 

# === AYARLAR ===

FENCE_POST_IDS = {1, 2}

VIOLATION_THRESHOLD = marker_length 
# ==============

obj_points = np.array([
    [-marker_length / 2,  marker_length / 2, 0],
    [ marker_length / 2,  marker_length / 2, 0],
    [ marker_length / 2, -marker_length / 2, 0],
    [-marker_length / 2, -marker_length / 2, 0]
], dtype=np.float32)

dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_50)
parameters = cv2.aruco.DetectorParameters()
detector = cv2.aruco.ArucoDetector(dictionary, parameters)

cap = cv2.VideoCapture(0)
print("İhlal Algılamalı Sanal Çit Sistemi Başlatıldı.")
print(f"Bariyer Markerları: {FENCE_POST_IDS}")

while True:
    ret, frame = cap.read()
    if not ret: break
        
    corners, ids, rejected = detector.detectMarkers(frame)
    
    violation_detected = False
    
    if ids is not None:
        cv2.aruco.drawDetectedMarkers(frame, corners, ids)
        
        flat_ids = ids.flatten()
        
        
        fence_tvecs = {}
        fence_centers_2d = {}
        
        for i, m_id in enumerate(flat_ids):
            success, rvec, tvec = cv2.solvePnP(obj_points, corners[i][0], camera_matrix, dist_coeffs)
            if success and m_id in FENCE_POST_IDS:
                fence_tvecs[m_id] = tvec.flatten()
                # 2B Merkez noktası
                fence_centers_2d[m_id] = (int(np.mean(corners[i][0][:, 0])), 
                                         int(np.mean(corners[i][0][:, 1])))
        
      
        if len(fence_tvecs) == len(FENCE_POST_IDS):
            t1 = fence_tvecs[list(FENCE_POST_IDS)[0]]
            t2 = fence_tvecs[list(FENCE_POST_IDS)[1]]
            c1 = fence_centers_2d[list(FENCE_POST_IDS)[0]]
            c2 = fence_centers_2d[list(FENCE_POST_IDS)[1]]
            
           
            fence_vector = t2 - t1
            fence_length_m = np.linalg.norm(fence_vector)
            fence_unit_vector = fence_vector / fence_length_m
            
            
            cv2.line(frame, c1, c2, (255, 0, 0), 4) # Ana çizgi mavi

            
            for i, m_id in enumerate(flat_ids):
                if m_id not in FENCE_POST_IDS:
                    success, rvec, intruder_tvec = cv2.solvePnP(obj_points, corners[i][0], camera_matrix, dist_coeffs)
                    if success:
                        intruder_pos = intruder_tvec.flatten()
                        v13 = intruder_pos - t1 
                        
                        
                        proj13 = np.dot(v13, fence_unit_vector)
                        
                       
                        if 0 <= proj13 <= fence_length_m:
                            # 3. En kısa dik mesafeyi hesapla (dist = sqrt(|v13|^2 - proj13^2))
                            perp_dist = np.sqrt(max(0, np.linalg.norm(v13)**2 - proj13**2))
                            
                            
                            if perp_dist < VIOLATION_THRESHOLD:
                                violation_detected = True
                                break 
            
            
            mid_point = ((c1[0] + c2[0]) // 2, (c1[1] + c2[1]) // 2)
            cv2.putText(frame, f"{fence_length_m*100:.1f} cm", (mid_point[0] - 40, mid_point[1] - 15), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)

            if violation_detected:
               
                cv2.line(frame, c1, c2, (0, 0, 255), 6) 
                cv2.putText(frame, "!!! IHLAL !!!", (frame.shape[1] // 2 - 120, 60), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 6, cv2.LINE_AA)

    cv2.imshow("Ihlal Algilama Sistemi", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()