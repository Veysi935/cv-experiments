import cv2
import cv2.aruco as aruco
import numpy as np

# Küpü ekrana çizdirecek fonksiyonumuz
def draw_cube(img, imgpts):
    # Küpün noktalarını tamsayı piksel koordinatlarına çeviriyoruz
    imgpts = np.int32(imgpts).reshape(-1, 2)
    
    # Zemin yüzeyi (Marker'ın tam üzeri) - Mavi çizgi
    img = cv2.drawContours(img, [imgpts[:4]], -1, (255, 0, 0), 3)
    
    # Sütunlar (Zeminden havaya kalkan direkler) - Yeşil çizgi
    for i, j in zip(range(4), range(4, 8)):
        img = cv2.line(img, tuple(imgpts[i]), tuple(imgpts[j]), (0, 255, 0), 3)
        
    # Tavan yüzeyi (Küpün üstü) - Kırmızı çizgi
    img = cv2.drawContours(img, [imgpts[4:8]], -1, (0, 0, 255), 3)
    return img

cap = cv2.VideoCapture(1)
ret, frame = cap.read()
h, w = frame.shape[:2]

focal_length = w
camera_matrix = np.array([
    [focal_length, 0, w / 2],
    [0, focal_length, h / 2],
    [0, 0, 1]
], dtype=np.float32)

dist_coeffs = np.zeros((4, 1))
marker_length = 0.05 
l = marker_length

# Küpün 8 Köşesinin 3 Boyutlu Uzaydaki Koordinatları
# Z=0 markerin zemini. Z=-l ise kameraya (bize) doğru yükselen kısmıdır
cube_points = np.float32([
    [-l/2,  l/2, 0], [ l/2,  l/2, 0], [ l/2, -l/2, 0], [-l/2, -l/2, 0], # Zemin (İlk 4 nokta)
    [-l/2,  l/2, -l],[ l/2,  l/2, -l],[ l/2, -l/2, -l],[-l/2, -l/2, -l] # Tavan (Son 4 nokta)
])

aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_6X6_250)
parameters = aruco.DetectorParameters()
detector = aruco.ArucoDetector(aruco_dict, parameters)

while True:
    ret, frame = cap.read()
    if not ret: break
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    corners, ids, rejected = detector.detectMarkers(gray)

    if ids is not None:
        for i in range(len(ids)):
            # Çerçeveyi değil, sadece marker'ın konumunu bulmak için obj_points
            obj_points = np.array([
                [-l/2,  l/2, 0],
                [ l/2,  l/2, 0],
                [ l/2, -l/2, 0],
                [-l/2, -l/2, 0]
            ], dtype=np.float32)
            
            # Markerın kamera olan pozisyonunu hesapla
            success, rvec, tvec = cv2.solvePnP(obj_points, corners[i], camera_matrix, dist_coeffs)
            
            if success:
                # SİHİR BURADA OLUYOR!
                # 3D küp noktalarını, o anki kamera açısına göre 2D piksele dönüştür
                imgpts, jac = cv2.projectPoints(cube_points, rvec, tvec, camera_matrix, dist_coeffs)
                
                # Fonksiyonu çağırıp küpü çizdir
                frame = draw_cube(frame, imgpts)

    cv2.imshow('ArUco 3D Kup (AR)', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'): 
        break

cap.release()
cv2.destroyAllWindows()