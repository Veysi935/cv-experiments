import cv2
import cv2.aruco as aruco
import numpy as np

# 1. Kamera ve ArUco ayarları
cap = cv2.VideoCapture(1)
aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_5X5_250)
parameters = aruco.DetectorParameters()
detector = aruco.ArucoDetector(aruco_dict, parameters)

# 2. Oynatılacak videoyu yükle
video_path = 'test_video.mp4'
video_cap = cv2.VideoCapture(video_path)

# Videonun başarıyla açılıp açılmadığını kontrol et
if not video_cap.isOpened():
    print(f"HATA: '{video_path}' dosyası bulunamadı! Lütfen klasöre bir video ekle.")
    exit()

# 3. Ana Döngü
while True:
    ret, frame = cap.read()
    if not ret: break
    
    # Videodan bir kare (frame) oku
    ret_vid, video_frame = video_cap.read()
    
    # Eğer video bittiyse başa sar (sürekli döngü)
    if not ret_vid:
        video_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        ret_vid, video_frame = video_cap.read()
        
    # Video karesinin boyutlarını al
    h_vid, w_vid, _ = video_frame.shape
    video_pts = np.float32([[0,0], [w_vid,0], [w_vid,h_vid], [0,h_vid]])

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    corners, ids, rejected = detector.detectMarkers(gray)

    if ids is not None:
        for i in range(len(ids)):
            marker_corners = corners[i].reshape(4, 2)
            
            # Matrisi hesapla
            H = cv2.getPerspectiveTransform(video_pts, np.float32(marker_corners))
            
            # Video karesini kameradaki markerın açısına göre yamult
            warped_video = cv2.warpPerspective(video_frame, H, (frame.shape[1], frame.shape[0]))
            
            # Maskeleme ve Birleştirme
            mask = np.zeros(frame.shape, dtype=np.uint8)
            cv2.fillConvexPoly(mask, np.int32(marker_corners), (255, 255, 255))
            
            mask_inv = cv2.bitwise_not(mask)
            frame_bg = cv2.bitwise_and(frame, mask_inv)
            frame_fg = cv2.bitwise_and(warped_video, mask)
            
            frame = cv2.add(frame_bg, frame_fg)

    cv2.imshow('ArUco Sinema Salonu', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'): 
        break

# İşlem bitince her şeyi kapat
cap.release()
video_cap.release()
cv2.destroyAllWindows()