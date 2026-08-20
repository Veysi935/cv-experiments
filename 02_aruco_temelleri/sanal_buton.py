import cv2
import cv2.aruco as aruco
import numpy as np

cap = cv2.VideoCapture(0)
aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_6X6_250)
parameters = aruco.DetectorParameters()
detector = aruco.ArucoDetector(aruco_dict, parameters)

# BUTONUN KOORDİNATLARI
# Sol üst köşe (x1, y1) ve Sağ alt köşe (x2, y2)
btn_x1, btn_y1 = 40, 40
btn_x2, btn_y2 = 200, 120

while True:
    ret, frame = cap.read()
    if not ret: break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    corners, ids, rejected = detector.detectMarkers(gray)

    # Varsayılan buton durumu (Tıklanmadı -> Kırmızı)
    btn_color = (0, 0, 255) 
    btn_text = "TETIKLE"

    if ids is not None:
        # Marker'ın merkez noktasını hesapla
        c = corners[0][0]
        center_x = int(np.mean(c[:, 0]))
        center_y = int(np.mean(c[:, 1]))
        
        # Marker'ın merkezini ekranda mavi bir nokta ile göster
        cv2.circle(frame, (center_x, center_y), 10, (255, 0, 0), -1)
        
        # ÇARPIŞMA KONTROLÜ (Nokta kutunun içinde mi?)
        if (btn_x1 < center_x < btn_x2) and (btn_y1 < center_y < btn_y2):
            # Eğer içindeyse butonu "Tıklandı" durumuna geçir (Yeşil)
            btn_color = (0, 255, 0)
            btn_text = "TIKLANDI!"
            
            # ---- SİHİR BURADA BAŞLAR ----
            # Buraya butona basıldığında olmasını istediğin kodu yazabilirsin!
            # Örnek: Kameradan fotoğraf çekip kaydetmek, bilgisayarda bir ses çalmak, 
            # bir motoru döndürmek veya başka bir Python fonksiyonunu tetiklemek...
            # -----------------------------

    # Butonu ekrana çizdir
    cv2.rectangle(frame, (btn_x1, btn_y1), (btn_x2, btn_y2), btn_color, -1)
    
    # Butonun üzerine yazısını ekle
    cv2.putText(frame, btn_text, (btn_x1 + 15, btn_y1 + 50), 
                cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 255, 255), 2)

    cv2.imshow('ArUco Sanal Buton', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'): 
        break

cap.release()
cv2.destroyAllWindows()