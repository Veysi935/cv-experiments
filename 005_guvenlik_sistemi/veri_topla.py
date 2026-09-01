import cv2
import os
import numpy as np  # İŞTE EKSİK OLAN KAHRAMAN BURADA

# Verilerimizin birikeceği ana klasör
klasor = "veri_seti"
if not os.path.exists(klasor):
    os.makedirs(klasor)

cap = cv2.VideoCapture(0)
sayac = 0

# Neyi tanımak istiyorsan adını buraya yaz (Boşluk kullanma)
nesne_adi = "veysi_yuz" 

while True:
    ret, frame = cap.read()
    if not ret: break

    # Ekrana yönlendirme metinleri yazdır
    cv2.putText(frame, "Cekmek icin 's' - Cikmak icin 'q'", (20, 40), 
                cv2.FONT_HERSHEY_DUPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(frame, f"Toplanan Veri: {sayac}", (20, 80), 
                cv2.FONT_HERSHEY_DUPLEX, 0.7, (0, 255, 255), 2)

    # Görseli göster
    cv2.imshow('Veri Toplama Araci', frame)

    key = cv2.waitKey(1) & 0xFF
    
    # 's' tuşuna basıldığında fotoğrafı kaydet
    if key == ord('s'):
        dosya_yolu = os.path.join(klasor, f"{nesne_adi}_{sayac}.jpg")
        
        cv2.imwrite(dosya_yolu, frame)
        print(f"Basariyla kaydedildi: {dosya_yolu}")
        
        sayac += 1
        
        # Çekildiğini belli etmek için ekranı anlık beyazlat (Flaş efekti)
        flas = 255 * np.ones(frame.shape, dtype=np.uint8)
        cv2.imshow('Veri Toplama Araci', flas)
        cv2.waitKey(50) # 50 milisaniye bekle
        
    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()