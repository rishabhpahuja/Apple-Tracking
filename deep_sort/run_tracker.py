from detection_helpers import *
from tracking_helpers import *
from  bridge_wrapper import *
import cv2

video_path='../../apples.mp4'

net=cv2.dnn.readNetFromDarknet("./backup_V3/yolov3.cfg","./backup_V3/yolov3_last.weights")
tracker=YOLOv7_DeepSORT(reID_model_path="./deep_sort/model_weights/mars-small128.pb", detector=net)

tracker.track_video(video_path, output="./IO_data/output/street.avi",show_live =False, skip_frames = 0, count_objects = True, verbose=1,dir_path='./Tests/')