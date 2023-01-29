from ultralytics import YOLO

model = YOLO('yolov8n.pt')

#Train YOLO V8
model.train(data="custom.yaml", epochs=5)