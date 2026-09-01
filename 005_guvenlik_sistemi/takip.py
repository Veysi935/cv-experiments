from collections import defaultdict
import cv2
import numpy as np
from ultralytics import YOLO
import time 

model = YOLO('runs/detect/train/weights/best.pt')
track_history = defaultdict(lambda: [])

giris_sayisi = 0
cikis_sayisi = 0
id_sayildi_mi = defaultdict(lambda: False)

cap = cv2.VideoCapture(0)
cizgi_y = 300 
onceki_zaman = 0 

def ciz_hud_kosesi(kare, pt1, pt2, renk, kalinlik, uzunluk=30):
    x, y = pt1
    x2, y2 = pt2
    cv2.line(kare, (x, y), (x + uzunluk, y), renk, kalinlik)
    cv2.line(kare, (x, y), (x, y + uzunluk), renk, kalinlik)
    cv2.line(kare, (x2, y2), (x2 - uzunluk, y2), renk, kalinlik)
    cv2.line(kare, (x2, y2), (x2, y2 - uzunluk), renk, kalinlik)

while cap.isOpened():
    success, frame = cap.read()
    if not success: break

    # --- FPS HESAPLAMA ---
    su_anki_zaman = time.time()
    fps = 1 / (su_anki_zaman - onceki_zaman)
    onceki_zaman = su_anki_zaman
    # ---------------------

    overlay = frame.copy()
    h, w = frame.shape[:2]
    cv2.rectangle(overlay, (20, 20), (350, 250), (0, 0, 0), -1) # Paneli biraz uzattık
    cv2.addWeighted(overlay, 0.4, frame, 0.6, 0, frame)

    results = model.track(frame, persist=True, tracker="bytetrack.yaml", conf=0.6, verbose=False)
    annotated_frame = frame 

    neon_mavi = (255, 200, 0)
    neon_yesil = (0, 255, 100)
    neon_kirmizi = (0, 50, 255)

    if results[0].boxes.id is not None:
        boxes = results[0].boxes.xywh.cpu() 
        track_ids = results[0].boxes.id.int().cpu().tolist()

        for box, track_id in zip(boxes, track_ids):
            x, y, bw, bh = box
            merkez_noktasi = (float(x), float(y))
            
            sol_ust = (int(x - bw/2), int(y - bh/2))
            sag_alt = (int(x + bw/2), int(y + bh/2))
            
            ciz_hud_kosesi(annotated_frame, sol_ust, sag_alt, neon_mavi, 3, uzunluk=25)
            cv2.circle(annotated_frame, (int(x), int(y)), 4, neon_kirmizi, -1)
            cv2.putText(annotated_frame, f"TRK_ID: {track_id}", (sol_ust[0], sol_ust[1] - 10), 
                        cv2.FONT_HERSHEY_PLAIN, 1.2, neon_mavi, 2)
            
            track = track_history[track_id]
            track.append(merkez_noktasi)  
            if len(track) > 30: track.pop(0)

            points = np.hstack(track).astype(np.int32).reshape((-1, 1, 2))
            cv2.polylines(annotated_frame, [points], isClosed=False, color=neon_yesil, thickness=2)

            if len(track) >= 2:
                onceki_y = track[-2][1]
                su_anki_y = merkez_noktasi[1]

                if not id_sayildi_mi[track_id]:
                    if onceki_y < cizgi_y and su_anki_y >= cizgi_y:
                        giris_sayisi += 1
                        id_sayildi_mi[track_id] = True
                    elif onceki_y > cizgi_y and su_anki_y <= cizgi_y:
                        cikis_sayisi += 1
                        id_sayildi_mi[track_id] = True

    cv2.line(annotated_frame, (0, cizgi_y), (w, cizgi_y), neon_kirmizi, 2)
    cv2.putText(annotated_frame, "RESTRICTED ZONE BORDER", (w - 300, cizgi_y - 10), cv2.FONT_HERSHEY_PLAIN, 1, neon_kirmizi, 1)

    cv2.putText(annotated_frame, "SYSTEM METRICS", (35, 50), cv2.FONT_HERSHEY_COMPLEX, 0.7, (255, 255, 255), 2)
    cv2.line(annotated_frame, (35, 60), (320, 60), neon_mavi, 1) 
    
    cv2.putText(annotated_frame, f"> INBOUND  : {giris_sayisi}", (35, 100), cv2.FONT_HERSHEY_DUPLEX, 0.7, neon_yesil, 2)
    cv2.putText(annotated_frame, f"> OUTBOUND : {cikis_sayisi}", (35, 140), cv2.FONT_HERSHEY_DUPLEX, 0.7, neon_mavi, 2)
    
    # --- TELEMETRİ VERİLERİNİ EKRANA YAZDIRMA ---
    cv2.line(annotated_frame, (35, 165), (320, 165), neon_mavi, 1) 
    cv2.putText(annotated_frame, f"FPS        : {int(fps)}", (35, 195), cv2.FONT_HERSHEY_PLAIN, 1.5, (255, 255, 255), 2)
    # Modelin bir kareyi işleme süresi (milisaniye cinsinden)
    islem_suresi = (1/fps) * 1000 if fps > 0 else 0
    cv2.putText(annotated_frame, f"LATENCY    : {int(islem_suresi)} ms", (35, 230), cv2.FONT_HERSHEY_PLAIN, 1.5, (255, 255, 255), 2)
    # --------------------------------------------

    cv2.imshow("CYBER-SEC DASHBOARD V1.0", annotated_frame)
    if cv2.waitKey(1) & 0xFF == ord("q"): break

cap.release()
cv2.destroyAllWindows()