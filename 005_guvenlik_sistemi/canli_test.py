import cv2
from ultralytics import YOLO
import os
import time
from datetime import datetime
import requests  # İnternete bağlanmamızı sağlayan kütüphane

# --- TELEGRAM AYARLARI ---
import os
from dotenv import load_dotenv

load_dotenv()

BOT_TOKEN = os.getenv("TELEGRAM_TOKEN")
CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

def telegram_foto_gonder(foto_yolu, mesaj):
    """Telegram'a fotoğraf gönderen özel fonksiyonumuz"""
    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendPhoto"
    try:
        with open(foto_yolu, "rb") as foto:
            payload = {"chat_id": CHAT_ID, "caption": mesaj}
            files = {"photo": foto}
            # Telegram sunucusuna postala
            requests.post(url, data=payload, files=files) 
            print("TELEGRAM'A GONDERILDI -> Telefonunu kontrol et!")
    except Exception as e:
        print("Telegram gonderim hatasi:", e)
# -------------------------

model_path = 'runs/detect/train/weights/best.pt'
print("Yapay zeka yukleniyor...")
model = YOLO(model_path)

cap = cv2.VideoCapture(0)

kayit_klasoru = "guvenlik_kayitlari"
if not os.path.exists(kayit_klasoru):
    os.makedirs(kayit_klasoru)

son_kayit_zamani = 0
bekleme_suresi = 10  

while True:
    ret, frame = cap.read()
    if not ret: break

    results = model(frame, conf=0.7)
    annotated_frame = results[0].plot()

    bulunan_id_listesi = results[0].boxes.cls.tolist()
    bulunan_isimler = [model.names[int(idx)] for idx in bulunan_id_listesi]

    if 'veysi_yuz' in bulunan_isimler:
        su_an = time.time()
        if (su_an - son_kayit_zamani) > bekleme_suresi:
            
            # 1. Fotoğrafı kaydet
            zaman_etiketi = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            dosya_adi = f"tespit_{zaman_etiketi}.jpg"
            dosya_yolu = os.path.join(kayit_klasoru, dosya_adi)
            cv2.imwrite(dosya_yolu, annotated_frame)
            print(f"LOG KAYDEDILDI: {dosya_adi}")
            
            
            uyari_mesaji = f"DIKKAT! Kamerada tespit yapildi.\nTarih: {zaman_etiketi}"
            telegram_foto_gonder(dosya_yolu, uyari_mesaji)
            
            son_kayit_zamani = su_an

    cv2.imshow('Akilli Guvenlik Kamerasi', annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()