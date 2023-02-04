from ultralytics import YOLO
import torch
import cv2
import numpy as np

class V8:

    def __init__(self,model_type=None,model_path=None):

        if model_path is None:
            self.model = YOLO("./yolo_v8/yolov8n.pt")  # load an official model
            self.model = YOLO("./yolo_v8/runs/detect/train2/weights/best.pt")  # load a custom model
        
        else:
            self.model=YOLO(model_type)
            self.model=YOLO(model_path)

    def pred(self, image, conf=0.15, iou=0.3, debug=False):

        # Predict with the model
        results = self.model(image,conf=conf,iou=iou)  # predict on an image
        boxes = np.asarray(results[0].cpu().numpy().boxes.xyxy,np.uint16)
        scores=results[0].boxes.conf.cpu().numpy()
        # import ipdb; ipdb.set_trace()
        if debug:
            #Displaying all the boxes and scores on the image
            for box,score in zip(boxes,results[0].boxes.conf.numpy()):
                cv2.rectangle(image,(box[0],box[1]),(box[2],box[3]),(255,0,0),2)
                cv2.putText(image,str(round(score,3)),(int(box[0]), int(box[1]-11)),0, 0.8, (255,255,255),2, lineType=cv2.LINE_AA)

            cv2.namedWindow("Predicted Image",cv2.WINDOW_NORMAL)
            cv2.imshow("Predicted Image",image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            cv2.imwrite('yolo_v8_score.png',image)
        
        return boxes,scores
