from ultralytics import YOLO
import torch

model = YOLO('yolov8n.pt')

#Train YOLO V8
model.train(data="custom.yaml", epochs=20,device=[0,1],batch=40)
torch.cuda.empty_cache()
model.val()