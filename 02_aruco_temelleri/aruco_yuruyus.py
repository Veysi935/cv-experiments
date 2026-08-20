import cv2
import numpy as np
import math

# 1. KALİBRASYON VERİLERİNİ YÜKLE
print("Kalibrasyon verileri yükleniyor...")
kalibrasyon_verisi = np.load("kamera_kalibrasyon.npz")
camera_matrix = kalibrasyon_verisi["camera_matrix"]
dist_coeffs = kalibrasyon_verisi["dist_coeffs"]
print("Veriler başarıyla yüklendi!\n")

# 2. MARKER BOYUTUNU GİR (Metre cinsinden)
# ÖRNEK: Marker 5 cm ise 0.05, 10 cm ise 0.10 yaz.
marker_length = 0.1

# Marker'ın 3B uzaydaki köşe koordinatları (Merkezi 0,0,0)
obj_points = np.array([
    [-marker_length / 2,  marker_length / 2, 0],
    [ marker_length / 2,  marker_length / 2, 0],
    [ marker_length / 2, -marker_length / 2, 0],
    [-marker_length / 2, -marker_length / 2, 0]
], dtype=np.float32)

# ArUco Dedektörünü Ayarla
dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_50)
parameters = cv2.aruco.DetectorParameters()
detector = cv2.aruco.ArucoDetector(dictionary, parameters)

cap = cv2.VideoCapture(1)
print("Kamera açıldı. Marker'ı yere yapıştırıp bilgisayarla yürümeye başla.")
print("Çıkmak için 'q' tuşuna bas.")

while True:
    ret, frame = cap.read()
    if not ret:
        break
        
    # Markerları tespit et
    corners, ids, rejected = detector.detectMarkers(frame)
    
    if ids is not None:
        cv2.aruco.drawDetectedMarkers(frame, corners, ids)
        
        for i in range(len(ids)):
            # Kalibre edilmiş matrislerimizle kameranın marker'a göre konumunu bul
            success, rvec, tvec = cv2.solvePnP(obj_points, corners[i][0], camera_matrix, dist_coeffs)
            
            if success:
                # tvec [X, Y, Z] eksenlerindeki uzaklıkları metre cinsinden verir.
                # Pisagor (Öklid) teoremiyle gerçek uzaklığı (hipotenüs) hesaplıyoruz
                distance_m = math.sqrt(tvec[0].item()**2 + tvec[1].item()**2 + tvec[2].item()**2)
                distance_cm = distance_m * 100
                
                # Eksenleri çiz (Kalibre matrislerimizi kullanarak)
                cv2.drawFrameAxes(frame, camera_matrix, dist_coeffs, rvec, tvec, marker_length / 2)
                
                # Bilgileri ekrana yazdır
                marker_id = ids.flatten()[i]
                text = f"ID: {marker_id} | Mesafe: {distance_cm:.1f} cm"
                
                # Yere doğru yürürken yazının üst üste binmemesi için dinamik konum
                cv2.putText(frame, text, (10, 30 + (i * 40)), cv2.FONT_HERSHEY_SIMPLEX, 
                            0.7, (0, 255, 0), 2, cv2.LINE_AA)

    cv2.imshow("ArUco Mesafe Olcumu", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()