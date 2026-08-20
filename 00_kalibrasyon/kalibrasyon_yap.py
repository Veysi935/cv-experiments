import cv2
import numpy as np
import os

# --- AYARLAR ---
# Kağıttaki İÇ köşe sayısı (Kare sayısının 1 eksiğidir)
# Eğer kamera algılamazsa burayı (6, 8) olarak değiştirin.
CHESSBOARD_SIZE = (7, 9) 

# Kare boyutu (20mm = 0.02 metre)
SQUARE_SIZE = 0.02 

# Kalibrasyon verisinin kaydedileceği dosya adı
SAVE_FILE = "kamera_kalibrasyon.npz"
# ---------------

# 3B uzaydaki referans noktalarını hazırla
objp = np.zeros((CHESSBOARD_SIZE[0] * CHESSBOARD_SIZE[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:CHESSBOARD_SIZE[0], 0:CHESSBOARD_SIZE[1]].T.reshape(-1, 2)
objp *= SQUARE_SIZE

objpoints = [] # Gerçek dünyadaki 3B noktalar
imgpoints = [] # Resim düzlemindeki 2B noktalar

cap = cv2.VideoCapture(0)
print("\n--- KAMERA KALİBRASYON ARACI ---")
print("1. Kameraya satranç tahtasını farklı açılardan gösterin.")
print("2. Desen algılandığında (renkli çizgiler çıktığında) 'c' tuşuna basarak fotoğraf çekin.")
print("3. En az 15-20 başarılı fotoğraf çektikten sonra 'q' tuşuna basarak kalibrasyonu tamamlayın.\n")

while True:
    ret, frame = cap.read()
    if not ret:
        break
        
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    display_frame = frame.copy()
    
    # Satranç tahtası köşelerini bul
    ret_corners, corners = cv2.findChessboardCorners(gray, CHESSBOARD_SIZE, None)
    
    if ret_corners:
        # Algılanan köşeleri ekrana çiz
        cv2.drawChessboardCorners(display_frame, CHESSBOARD_SIZE, corners, ret_corners)
        cv2.putText(display_frame, "Desen Algilandi! Cekmek icin 'c'ye bas.", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    else:
        cv2.putText(display_frame, "Desen araniyor...", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
    cv2.putText(display_frame, f"Cekilen Fotograf: {len(objpoints)}", (10, 70), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

    cv2.imshow('Kamera Kalibrasyonu', display_frame)
    key = cv2.waitKey(1) & 0xFF
    
    # 'c' tuşuna basıldığında ve desen ekrandayken kaydet
    if key == ord('c') and ret_corners:
        # Köşe konumlarını daha hassas (sub-pixel) hale getir
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        corners_refined = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        
        objpoints.append(objp)
        imgpoints.append(corners_refined)
        print(f"Fotoğraf {len(objpoints)} kaydedildi!")
        
    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# Yeterli fotoğraf varsa hesapla ve kaydet
if len(objpoints) >= 10:
    print("\nKalibrasyon hesaplanıyor, lütfen bekleyin...")
    ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
        objpoints, imgpoints, gray.shape[::-1], None, None)
    
    # Matrisleri .npz dosyasına kaydet
    np.savez(SAVE_FILE, camera_matrix=camera_matrix, dist_coeffs=dist_coeffs)
    
    print("\n" + "="*50)
    print("KALİBRASYON BAŞARILI VE KAYDEDİLDİ!")
    print(f"Dosya: {os.path.abspath(SAVE_FILE)}")
    print("="*50)
    print("\nHata Payı (RMS):", ret) # 1.0'ın altındaysa kalibrasyon çok iyidir.
else:
    print(f"\nİşlem iptal edildi. Yeterli fotoğraf çekilmedi (Sadece {len(objpoints)} tane).")