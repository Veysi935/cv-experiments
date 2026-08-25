import cv2
import cv2.aruco as aruco
import numpy as np
import math

cap = cv2.VideoCapture(0)
aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_6X6_250)
parameters = aruco.DetectorParameters()
detector = aruco.ArucoDetector(aruco_dict, parameters)

while True:
    ret, frame = cap.read()
    if not ret: break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    corners, ids, rejected = detector.detectMarkers(gray)

    if ids is not None:
        # Marker'ın 4 köşesini alıyoruz
        # [0]=Sol-Üst, [1]=Sağ-Üst, [2]=Sağ-Alt, [3]=Sol-Alt
        c = corners[0][0]
        sol_ust = c[0]
        sag_ust = c[1]
        
        # İki nokta arasındaki X ve Y farklarını bul
        dx = sag_ust[0] - sol_ust[0]
        dy = sag_ust[1] - sol_ust[1]
        
        # Trigonometri (math.atan2) ile açıyı hesapla ve dereceye çevir
        aci = math.degrees(math.atan2(dy, dx))
        
        # Açı 0'a yakınsa düz, eksi değerler SOLA, artı değerler SAĞA dönüşü gösterir
        # Ufak titremeleri önlemek için -10 ile 10 derece arasını "Düz" kabul ediyoruz (Deadzone)
        if aci < -10:
            durum = "SOLA DONUYOR"
            renk = (0, 0, 255) # Kırmızı
        elif aci > 10:
            durum = "SAGA DONUYOR"
            renk = (0, 255, 0) # Yeşil
        else:
            durum = "DUZ GIDIYOR"
            renk = (255, 255, 0) # Turkuaz
            
        # Görselleştirme: Marker'ın üst kenarına kalın bir referans çizgisi çek
        pt1 = (int(sol_ust[0]), int(sol_ust[1]))
        pt2 = (int(sag_ust[0]), int(sag_ust[1]))
        cv2.line(frame, pt1, pt2, renk, 6)
        
        # Bilgileri ekrana yaz
        cv2.putText(frame, f"Aci: {int(aci)} derece", (40, 50), cv2.FONT_HERSHEY_DUPLEX, 1, (255, 255, 255), 2)
        cv2.putText(frame, durum, (40, 100), cv2.FONT_HERSHEY_DUPLEX, 1.5, renk, 3)

    cv2.imshow('ArUco Sanal Direksiyon', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'): 
        break

cap.release()
cv2.destroyAllWindows()