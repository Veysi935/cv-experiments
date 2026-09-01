from ultralytics import YOLO


model = YOLO('yolov8n.pt') 

print("Egitim basliyor! Arkana yaslan...")


results = model.train(data='data.yaml', epochs=50, imgsz=640)