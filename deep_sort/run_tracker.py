from detection_helpers import *
from tracking_helpers import *
from  bridge_wrapper import *
import cv2

left_video_path='../apples_left.mp4'
right_video_path='../apples_right.mp4'
image=cv2.imread("/home/pahuja/Projects/Apple tracking/deep_sort/yolo_v8/datasets/train/images/left0002.jpg")
net=cv2.dnn.readNetFromDarknet("./backup_V3/yolov3.cfg","./backup_V3/yolov3_last.weights")
# yolo_dets, scores, class_ID=YOLOV3(net,image)

# for row in yolo_dets:
#         # image,point_mask=find_contour_center(mask[int(row[1]):int(row[1]+row[3]),int(row[0]):int(row[0]+row[2])],image,row,point_mask,display=False)
#     cv2.rectangle(image,(int(row[0]),int(row[1])),(int(row[0]+row[2]),int(row[1]+row[3])),(255,255,0),4)

# cv2.namedWindow('image',cv2.WINDOW_NORMAL)
# cv2.imshow('image',image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
# cv2.imwrite('yolo_v3.jpg',image)

tracker=YOLOv7_DeepSORT(reID_model_path="./deep_sort/model_weights/mars-small128.pb", detector=net)

tracker.track_video(left_video_path,right_video_path ,output="./IO_data/output/street_conf_0.3.mp4",show_live =True, skip_frames = 0, count_objects = True, verbose=1,dir_path='./Tests/3D_cost')

