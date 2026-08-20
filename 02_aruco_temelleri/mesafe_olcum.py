import cv2
import cv2.aruco as aruco
import numpy as np
import math

cap = cv2.VideoCapture(0)
ret, frame = cap.read()
h, w = frame.shape[:2]

# Tahmini kamera matrisi
focal_length = w
camera_matrix = np.array([
    [focal_length, 0, w / 2],
    [0, focal_length, h / 2],
    [0, 0, 1]
], dtype=np.float32)
dist_coeffs = np.zeros((4, 1))

# Marker boyutu (metre cinsinden 0.05 = 5 cm)
marker_length = 0.05 

aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_5X5_250)
parameters = aruco.DetectorParameters()
detector = aruco.ArucoDetector(aruco_dict, parameters)

# Markerın ideal 3D noktaları
obj_points = np.array([
    [-marker_length/2,  marker_length/2, 0],
    [ marker_length/2,  marker_length/2, 0],
    [ marker_length/2, -marker_length/2, 0],
    [-marker_length/2, -marker_length/2, 0]
], dtype=np.float32)

while True:
    ret, frame = cap.read()
    if not ret: break
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    corners, ids, rejected = detector.detectMarkers(gray)

    # Ekranda en az 2 tane marker varsa işleme başla
    if ids is not None and len(ids) >= 2:
        cv2.aruco.drawDetectedMarkers(frame, corners, ids)
        
        tvecs = []   # Z eksenini de içeren 3D konum vektörleri
        centers = [] # Ekrana çizgi çekmek için 2D merkez pikselleri
        
        for i in range(len(ids)):
            success, rvec, tvec = cv2.solvePnP(obj_points, corners[i], camera_matrix, dist_coeffs)
            if success:
                tvecs.append(tvec)
                
                # 2D Piksel merkezini hesapla (köşelerin ortalaması)
                c = corners[i][0]
                center_x = int(np.mean(c[:, 0]))
                center_y = int(np.mean(c[:, 1]))
                centers.append((center_x, center_y))
                
        # Başarıyla hesaplanmış en az 2 marker konumu varsa
        if len(tvecs) >= 2:
            t1 = tvecs[0]
            t2 = tvecs[1]
            
            # 3 Boyutlu Öklid Uzaklığı Formülü
            distance_m = math.sqrt((t1[0]-t2[0])**2 + (t1[1]-t2[1])**2 + (t1[2]-t2[2])**2)
            distance_cm = distance_m * 100
            
            # İki marker'ın merkezi arasına sarı bir çizgi çek
            cv2.line(frame, centers[0], centers[1], (0, 255, 255), 3)
            
            # Mesafeyi çizginin tam ortasına yazdır
            mid_x = int((centers[0][0] + centers[1][0]) / 2)
            mid_y = int((centers[0][1] + centers[1][1]) / 2)
            cv2.putText(frame, f"{distance_cm:.1f} cm", (mid_x, mid_y - 15), 
                        cv2.FONT_HERSHEY_DUPLEX, 0.8, (0, 255, 0), 2)

    cv2.imshow('ArUco 3D Mesafe Olcumu', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'): 
        break

cap.release()
cv2.destroyAllWindows()