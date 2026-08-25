import cv2
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 1. Kalibrasyon Yükle
kalibrasyon_verisi = np.load("kamera_kalibrasyon.npz")
camera_matrix = kalibrasyon_verisi["camera_matrix"]
dist_coeffs = kalibrasyon_verisi["dist_coeffs"]

marker_length = 0.05 
obj_points = np.array([
    [-marker_length / 2,  marker_length / 2, 0],
    [ marker_length / 2,  marker_length / 2, 0],
    [ marker_length / 2, -marker_length / 2, 0],
    [-marker_length / 2, -marker_length / 2, 0]
], dtype=np.float32)

dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_50)
parameters = cv2.aruco.DetectorParameters()
detector = cv2.aruco.ArucoDetector(dictionary, parameters)

room_map = {} 
origin_id = None 

plt.ion()
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')

cap = cv2.VideoCapture(1)
print("Otonom SLAM Başladı. Kameranızı markerlara doğru tutun.")

while True:
    ret, frame = cap.read()
    if not ret: break
        
    corners, ids, rejected = detector.detectMarkers(frame)
    current_camera_pos = None # Her karede kameranın konumunu sıfırla
    active_ref_id = None      # Kameranın o an baktığı referans marker
    
    if ids is not None:
        cv2.aruco.drawDetectedMarkers(frame, corners, ids)
        flat_ids = ids.flatten()
        
        if origin_id is None:
            origin_id = flat_ids[0]
            room_map[origin_id] = np.array([0.0, 0.0, 0.0])
        
        known_markers = [m_id for m_id in flat_ids if m_id in room_map]
        unknown_markers = [m_id for m_id in flat_ids if m_id not in room_map]
        
        if len(known_markers) > 0:
            cv2.putText(frame, "TAKIP: AKTIF", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            ref_id = known_markers[0]
            ref_idx = np.where(flat_ids == ref_id)[0][0]
            active_ref_id = ref_id # Çizgi çekmek için kaydediyoruz
            
            succ_ref, rvec_ref, tvec_ref = cv2.solvePnP(obj_points, corners[ref_idx][0], camera_matrix, dist_coeffs)
            
            if succ_ref:
                # KAMERANIN (SENİN) KONUMUNU HESAPLIYORUZ
                R_ref, _ = cv2.Rodrigues(rvec_ref) 
                R_cam_to_world = R_ref.T
                t_cam_to_ref = -np.dot(R_ref.T, tvec_ref.flatten()) 
                
                # İşte senin uzaydaki X, Y, Z konumun!
                current_camera_pos = room_map[ref_id] + t_cam_to_ref
                
                # Bilinmeyen marker varsa haritaya ekle
                if len(unknown_markers) > 0:
                    for unk_id in unknown_markers:
                        unk_idx = np.where(flat_ids == unk_id)[0][0]
                        succ_unk, rvec_unk, tvec_unk = cv2.solvePnP(obj_points, corners[unk_idx][0], camera_matrix, dist_coeffs)
                        
                        if succ_unk:
                            t_unk_world = np.dot(R_cam_to_world, tvec_unk.flatten())
                            final_world_pos = current_camera_pos + t_unk_world
                            room_map[unk_id] = final_world_pos

        elif len(known_markers) == 0 and origin_id is not None:
            cv2.putText(frame, "ZINCIR KOPTU! Bilinen bir marker'a donun...", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    # --- 3B CANLI ÇİZİM ---
    if origin_id is not None:
        ax.cla() 
        ax.set_title("Oda 3D Haritası ve Senin Konumun")
        ax.set_xlabel("X (Metre)")
        ax.set_ylabel("Y (Metre)")
        ax.set_zlabel("Z (Metre)")
        
        # 1. Duvarlardaki Markerları Çiz (Harita)
        for m_id, pos in room_map.items():
            if m_id == origin_id:
                ax.scatter(pos[0], pos[1], pos[2], c='red', s=100, label=f'Orijin (ID:{m_id})')
            else:
                ax.scatter(pos[0], pos[1], pos[2], c='blue', s=50)
            ax.text(pos[0], pos[1], pos[2], f" ID:{m_id}", color='black')
            
        # 2. SENİN KONUMUNU ÇİZ (Yeşil Üçgen)
        if current_camera_pos is not None:
            # Kamerayı büyük yeşil bir üçgen olarak çiziyoruz
            ax.scatter(current_camera_pos[0], current_camera_pos[1], current_camera_pos[2], 
                       c='green', marker='^', s=250, label='SEN (Kamera)')
            
            # Kameranın o an hangi marker'a baktığını gösteren kesik yeşil bir görüş çizgisi (Line of Sight)
            if active_ref_id is not None:
                ref_pos = room_map[active_ref_id]
                ax.plot([current_camera_pos[0], ref_pos[0]], 
                        [current_camera_pos[1], ref_pos[1]], 
                        [current_camera_pos[2], ref_pos[2]], 'g--')
            
        ax.legend(loc='upper left')
        plt.pause(0.001)

    cv2.imshow("Kamera Gorus Acisi", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()
plt.ioff()
plt.show()