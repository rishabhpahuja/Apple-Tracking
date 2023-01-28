from detection_helpers import *
from tracking_helpers import *
from  bridge_wrapper import *
import cv2

left_video_path='../apples_left.mp4'
right_video_path='../apples_right.mp4'

net=cv2.dnn.readNetFromDarknet("./backup_V3/yolov3.cfg","./backup_V3/yolov3_last.weights")
tracker=YOLOv7_DeepSORT(reID_model_path="./deep_sort/model_weights/mars-small128.pb", detector=net)

tracker.track_video(left_video_path,right_video_path ,output="./IO_data/output/street.avi",show_live =True, skip_frames = 0, count_objects = True, verbose=1,dir_path='./Tests/3D_cost')

cv2.namedWindow('Image',cv2.WINDOW_NORMAL)
image=cv2.imshow('Image',cv2.imread('./Tests/002.png'))
cv2.waitKey(0)
cv2.destroyAllWindows()

# q=([[          1,           0,           0,     -987.14],
#        [          0,           1,           0,     -782.62],
#        [          0,           0,           0,      1142.3],
#        [          0,           0,   0.0073371,          -0]])
# b=np.array([[1177],[1109],[70],[1]])
# a=q@b
# a=a[:-1]/a[3]
# print(a)