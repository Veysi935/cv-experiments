import cv2
import cv2.aruco as aruco
import numpy as np
import math
import keyboard

cap = cv2.VideoCapture(0)
aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_5X5_250)
parameters = aruco.DetectorParameters()
detector = aruco.ArucoDetector(aruco_dict, parameters)

while True:
    ret, frame = cap.read()
    if not ret: break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    corners, ids, rejected = detector.detectMarkers(gray)

    if ids is not None:
        c = corners[0][0]
        sol_ust = c[0]
        sag_ust = c[1]
        
        # 1. DİREKSİYON MANTIĞI (Açı)
        dx = sag_ust[0] - sol_ust[0]
        dy = sag_ust[1] - sol_ust[1]
        aci = math.degrees(math.atan2(dy, dx))
        
        # 2. GAZ/FREN MANTIĞI (Genişlik)
        # İki köşe arasındaki piksel mesafesini (genişliği) Pisagor ile buluyoruz
        marker_genislik = math.sqrt(dx**2 + dy**2)
        
        # -- KLAVYE KONTROLLERİ --
        
        # SAĞ/SOL (Direksiyon)
        if aci < -15:
            yon = "SOL (A)"
            keyboard.press('a')
            keyboard.release('d')
        elif aci > 15:
            yon = "SAG (D)"
            keyboard.press('d')
            keyboard.release('a')
        else:
            yon = "DUZ"
            keyboard.release('a')
            keyboard.release('d')
            
        # İLERİ/GERİ (Gaz/Fren)
        # DİKKAT: Bu 150 ve 90 değerleri kamerana göre değişebilir, gerekirse değiştir!
        if marker_genislik > 150: # Marker büyükse (Yakınsa)
            hiz = "GAZ (W)"
            hiz_renk = (0, 255, 0)
            keyboard.press('w')
            keyboard.release('s')
        elif marker_genislik < 90: # Marker küçükse (Uzaksa)
            hiz = "FREN/GERI (S)"
            hiz_renk = (0, 0, 255)
            keyboard.press('s')
            keyboard.release('w')
        else:
            hiz = "BOS (Sabit)"
            hiz_renk = (255, 255, 0)
            keyboard.release('w')
            keyboard.release('s')

        # -- GÖRSELLEŞTİRME (Dashboard) --
        # Marker'ın etrafını çiz
        cv2.aruco.drawDetectedMarkers(frame, corners, ids)
        
        # Ekrana bir araç paneli çizdiriyoruz
        cv2.putText(frame, f"Yon: {yon} ({int(aci)} deg)", (20, 50), cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(frame, f"Motor: {hiz}", (20, 90), cv2.FONT_HERSHEY_DUPLEX, 0.8, hiz_renk, 2)
        cv2.putText(frame, f"Marker Boyutu: {int(marker_genislik)} px", (20, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

    else:
        # Marker ekranda yoksa tüm tuşları serbest bırak (Kaza yapmayalım!)
        keyboard.release('w')
        keyboard.release('a')
        keyboard.release('s')
        keyboard.release('d')
        cv2.putText(frame, "MARKER BEKLENIYOR...", (20, 50), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 255), 2)

    cv2.imshow('ArUco Yaris Simulatoru', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'): 
        break

# Çıkarken ortalığı temizle
keyboard.release('w'); keyboard.release('a'); keyboard.release('s'); keyboard.release('d')
cap.release()
cv2.destroyAllWindows()