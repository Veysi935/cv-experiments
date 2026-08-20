# cv-experiments

ArUco marker tabanlı poz tahmini (pose estimation), MediaPipe ile el/yüz takibi ve YOLO tabanlı nesne tespiti üzerine yaptığım küçük-orta ölçekli deneyleri ve bir uçtan-uca güvenlik sistemini içeren bir koleksiyon.

Her klasör, aynı temel tekniğin (kamera kalibrasyonu, solvePnP, homografi, segmentasyon) farklı bir problemde nasıl kullanılabileceğini gösteriyor.

---

## 📁 Repo Yapısı

| Klasör | İçerik | Ana Teknikler |
|---|---|---|
| `00_kalibrasyon` | Kamera kalibrasyonu (satranç tahtası ile) | `cv2.calibrateCamera`, chessboard corner detection |
| `01_klasik_cv` | Görünmezlik pelerini, arka plan bulanıklaştırma, air canvas, monoküler mesafe tahmini | HSV renk maskeleme, YOLOv8-seg, MediaPipe Hands/Face |
| `02_aruco_temelleri` | Marker arası mesafe ölçümü, marker üzerine AR video/3B küp projeksiyonu, sanal buton, iz bırakma | `solvePnP`, homografi, `projectPoints` |
| `03_aruco_fusion` | ArUco ile direksiyon kontrolü, el ile tetiklenen AR nesneleri, pinch-zoom etkileşimi | ArUco + MediaPipe Hands, klavye otomasyonu |
| Marker'ları referans alarak odayı 3B haritalama (basit SLAM) | Zincirleme koordinat dönüşümü, Rodrigues, Matplotlib 3D |
| `04_guvenlik_sistemi` | Veri toplama → YOLO eğitimi → sanal çit → kuşbakışı radar → gerçek zamanlı ihlal tespiti + Telegram alarmı → giriş/çıkış sayacı | YOLOv8, ByteTrack, Telegram Bot API, point-to-segment distance |

---

## 🛠️ Genel Kullanılan Teknolojiler

- **Dil:** Python 3.x
- **Görüntü İşleme:** OpenCV (`cv2`, `cv2.aruco`)
- **Nesne Tespiti / Takip:** YOLOv8 (Ultralytics), ByteTrack
- **El / Yüz Takibi:** MediaPipe (Hands, Face Detection)
- **Matematik:** NumPy, `math`
- **Görselleştirme:** Matplotlib (3B harita)
- **Bildirim / Otomasyon:** Telegram Bot API, `requests`, `threading`, `keyboard`

---

## 📂 Klasör Detayları

### 00_kalibrasyon
- `kalibrasyon_yap.py` — Satranç tahtası deseniyle kamera matrisini (`camera_matrix`) ve distorsiyon katsayılarını (`dist_coeffs`) hesaplayıp `.npz` dosyasına kaydeder. Diğer tüm ArUco projelerinin kalibre veri kaynağıdır.

### 01_klasik_cv
- `pelerin.py` — HSV renk uzayında maskeleme ile "görünmezlik pelerini" efekti.
- `odak_modu.py` — YOLOv8-seg ile insan segmentasyonu, arka planı bulanıklaştırıp öznenin net kalmasını sağlar.
- `sanal_kalem.py` — MediaPipe ile işaret parmağı ucunu takip ederek havada çizim (air canvas).
- `mesafe_radari.py` — Tek kameradan (monoküler), yüz genişliği üzerinden benzer üçgenler formülüyle mesafe tahmini.

### 02_aruco_temelleri
- `mesafe_olcum.py` — İki ArUco marker arasındaki 3B öklid mesafesini `solvePnP` ile ölçer.
- `image_overlay.py` — Homografi ile marker üzerine gerçek zamanlı video projeksiyonu.
- `main.py` — `projectPoints` ile marker üzerinde 3B küp çizen AR demosu.
- `sanal_buton.py` — Marker'ın ekran konumuna göre tetiklenen sanal buton sistemi.
- `yuruyus.py` / `aruco_yuruyus.py` — Marker'a olan mesafeyi gerçek zamanlı ölçüp ekrana yazdırma.

### 03_aruco_fusion
- `oyun_kontrol.py` — Marker'ın açısı ve boyutuna göre klavye tuşlarını tetikleyip bir yarış oyununu kontrol eder.
- *(AR motor / pinch-zoom drone projeleri bulunduğunda buraya eklenecek.)*

- `3ucboyutlutarama.py` — İlk görülen marker'ı orijin kabul edip, zincirleme koordinat dönüşümüyle odadaki tüm marker'ları ve kameranın konumunu 3B olarak haritalayan basitleştirilmiş bir SLAM sistemi.

### 04_guvenlik_sistemi
- `veri_topla.py` — Kameradan eğitim verisi (fotoğraf) toplama aracı.
- `egitim.py` — Toplanan veriyle YOLOv8 modelini eğitir.
- `sanal_cit.py` — İki marker arasına matematiksel bir çit çizer, point-to-segment distance ile ihlal tespiti yapar.
- `sanal_cit_radar.py` — Sanal çit sistemini homografi ile kuşbakışı radar görünümüne taşır.
- `sanal_cit_ihlal.py` — YOLO ile gerçek zamanlı insan tespiti + çit ihlali anında Telegram'a otomatik fotoğraf/bildirim gönderimi.
- `takip.py` — ByteTrack ile giriş/çıkış sayan, FPS/latency gösteren HUD panelli takip sistemi.
- `canli_test.py` — Eğitilen özel modelle gerçek zamanlı tespit + Telegram bildirimi.

---
