from ultralytics import YOLO
import torch
import cv2
import numpy as np

# model = YOLO('yolov8n.pt')

#Train YOLO V8
def train():
    model.train(data="custom.yaml", epochs=20,device=[0,1],batch=40)
    torch.cuda.empty_cache()
    model.val()

def pred():
    model = YOLO("yolov8n.pt")  # load an official model
    model = YOLO("/home/pahuja/Projects/Apple tracking/deep_sort/yolo_v8/runs/detect/train2/weights/best.pt")  # load a custom model
    # Predict with the model
    image=cv2.imread("/home/pahuja/Projects/Apple tracking/deep_sort/yolo_v8/datasets/train/images/left0002.jpg")
    results = model("/home/pahuja/Projects/Apple tracking/deep_sort/yolo_v8/datasets/train/images/left0002.jpg",conf=0.15,iou=0.3)  # predict on an image
    # labels, cord_thres = results.xyxyn[0][:, -1].numpy(), results.xyxyn[0][:, :-1].numpy()
    boxes = np.asarray(results[0].cpu().numpy().boxes.xyxy,np.uint16)

    for box in boxes:
        cv2.rectangle(image,(box[0],box[1]),(box[2],box[3]),(255,0,0),2)
    
    cv2.namedWindow("Predicted Image",cv2.WINDOW_NORMAL)
    cv2.imshow("Predicted Image",image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cv2.imwrite('yolo_v8.png',image)
pred()