import cv2
import numpy as np
import math

# 1. KULLANDIĞIN MARKER BOYUTUNU BURAYA GİR (Metre cinsinden)
# Örneğin marker 5 cm ise 0.05 yazmalısın.
marker_length = 0.1 

# Marker'ın 3B uzaydaki köşe koordinatları (Merkezi 0,0,0 kabul ediyoruz)
obj_points = np.array([
    [-marker_length / 2,  marker_length / 2, 0],
    [ marker_length / 2,  marker_length / 2, 0],
    [ marker_length / 2, -marker_length / 2, 0],
    [-marker_length / 2, -marker_length / 2, 0]
], dtype=np.float32)

# 2. TAHMİNİ KAMERA MATRİSİ (Standart 640x480 Webcam için)
focal_length = 600.0
center = (320, 240)
camera_matrix = np.array([
    [focal_length, 0, center[0]],
    [0, focal_length, center[1]],
    [0, 0, 1]
], dtype="double")
dist_coeffs = np.zeros((4, 1)) # Distorsiyonu (bükülmeyi) şimdilik sıfır varsayıyoruz

# ArUco Sözlüğünü ve Dedektörünü Ayarla (DICT_4X4_50 kullandığını varsayıyorum, sendekine göre değiştir)
dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_50)
parameters = cv2.aruco.DetectorParameters()
detector = cv2.aruco.ArucoDetector(dictionary, parameters)

cap = cv2.VideoCapture(1)

print("Kamera açıldı. Çıkmak için 'q' tuşuna basın.")

while True:
    ret, frame = cap.read()
    if not ret:
        break
        
    # Markerları tespit et
    corners, ids, rejected = detector.detectMarkers(frame)
    
    if ids is not None:
        cv2.aruco.drawDetectedMarkers(frame, corners, ids)
        
        for i in range(len(ids)):
            # solvePnP ile kameranın marker'a göre konumunu (tvec) ve dönüşünü (rvec) bul
            success, rvec, tvec = cv2.solvePnP(obj_points, corners[i][0], camera_matrix, dist_coeffs)
            
            if success:
                # tvec (Translation Vector) bize kameranın X, Y ve Z eksenlerindeki uzaklığını verir
                # Gerçek mesafeyi bulmak için 3 boyutlu öklid uzaklığını hesaplıyoruz
                distance = math.sqrt(tvec[0]**2 + tvec[1]**2 + tvec[2]**2)
                
                # Ekrana eksenleri çiz
                cv2.drawFrameAxes(frame, camera_matrix, dist_coeffs, rvec, tvec, marker_length)
                
                # Mesafeyi ekrana yazdır (Metreyi santimetreye çevirerek)
                text = f"Mesafe: {distance * 100:.1f} cm"
                cv2.putText(frame, text, (10, 50 + (i * 40)), cv2.FONT_HERSHEY_SIMPLEX, 
                            1, (0, 255, 0), 2, cv2.LINE_AA)

    cv2.imshow("ArUco Tracker", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()