from detection_helpers import *
from tracking_helpers import *
from  bridge_wrapper import *
from yolo_v8.pred import V8
from predict import Segmentation
from demo import Disparity

import cv2


left_video_path='../apples_left.mp4'
right_video_path='../apples_right.mp4'

#Declare detector
detector= V8()

#Declare segmentation object
segmentation= Segmentation()

#Declare disparity object
disparity=Disparity()

# tracker=YOLOv7_DeepSORT(reID_model_path="./deep_sort/model_weights/mars-small128.pb", detector=net)

# tracker.track_video(left_video_path,right_video_path ,output="./IO_data/output/street_conf_0.3.mp4",show_live =False, skip_frames = 0, count_objects = True, verbose=1,dir_path='./Tests/3D_cost')

