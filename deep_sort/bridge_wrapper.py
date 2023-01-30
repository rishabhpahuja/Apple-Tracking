'''
A Moduele which binds Yolov7 repo with Deepsort with modifications
'''
import sys
import os
sys.path.append('/home/pahuja/Projects/Apple tracking')
sys.path.append('/home/pahuja/Projects/Apple tracking/Disparity')
sys.path.append('/home/pahuja/Projects/Apple tracking/Segmentation')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # comment out below line to enable tensorflow logging outputs
import time
import tensorflow as tf

physical_devices = tf.config.experimental.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
    tf.config.experimental.set_memory_growth(physical_devices[1], True)
    
import cv2
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.compat.v1 import ConfigProto # DeepSORT official implementation uses tf1.x so we have to do some modifications to avoid errors

# deep sort imports
from deep_sort import preprocessing, nn_matching
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker

# import from helpers
from tracking_helpers import read_class_names, create_box_encoder
from detection_helpers import *
import Utils as ut


 # load configuration for object detector
config = ConfigProto()
config.gpu_options.allow_growth = True



class YOLOv7_DeepSORT:
    '''
    Class to Wrap ANY detector  of YOLO type with DeepSORT
    '''
    def __init__(self, reID_model_path:str, detector, max_cosine_distance:float=0.4, nn_budget:float=None, nms_max_overlap:float=1.0,
    coco_names_path:str ="./io_data/input/classes/coco.names",  ):
        '''
        args: 
            reID_model_path: Path of the model which uses generates the embeddings for the cropped area for Re identification
            detector: object of YOLO models or any model which gives you detections as [x1,y1,x2,y2,scores, class]
            max_cosine_distance: Cosine Distance threshold for "SAME" person matching
            nn_budget:  If not None, fix samples per class to at most this number. Removes the oldest samples when the budget is reached.
            nms_max_overlap: Maximum NMs allowed for the tracker
            coco_file_path: File wich contains the path to coco naames
        '''
        self.detector = detector
        self.coco_names_path = ['apples']
        self.nms_max_overlap = nms_max_overlap
        # self.class_names = read_class_names()

        # initialize deep sort
        self.encoder = create_box_encoder(reID_model_path, batch_size=1)
        metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget) # calculate cosine distance metric
        self.tracker = Tracker(metric) # initialize tracker


    def track_video(self,left_video:str, right_video:str,output:str, skip_frames:int=0, show_live:bool=False, count_objects:bool=False, verbose:int = 0, YOLOVER='V3',dir_path='./Tests/3D_cost'):
        '''
        Track any given webcam or video
        args: 
            video: path to input video or set to 0 for webcam
            output: path to output video
            skip_frames: Skip every nth frame. After saving the video, it'll have very visuals experience due to skipped frames
            show_live: Whether to show live video tracking. Press the key 'q' to quit
            count_objects: count objects being tracked on screen
            verbose: print details on the screen allowed values 0,1,2
        '''
        try: # begin video capture
            vid_left = cv2.VideoCapture(int(left_video))
            vid_right = cv2.VideoCapture(int(right_video))
        except:
            vid_left = cv2.VideoCapture(left_video)
            vid_right = cv2.VideoCapture(right_video)

        out = None
        if output: # get video ready to save locally if flag is set
            width = int(vid_left.get(cv2.CAP_PROP_FRAME_WIDTH))  # by default VideoCapture returns float instead of int
            height = int(vid_left.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = int(vid_left.get(cv2.CAP_PROP_FPS))
            codec = cv2.VideoWriter_fourcc(*"XVID")
            out = cv2.VideoWriter(output, codec, fps, (width, height))

        frame_num = 0
        while True: # while video is running
            return_value_left, frame_left = vid_left.read()
            return_value_right, frame_right = vid_right.read()
            if not return_value_left or not return_value_right:
                print('Video has ended or failed!')
                break
            frame_num +=1

            if skip_frames and not frame_num % skip_frames: continue # skip every nth frame. When every frame is not important, you can use this to fasten the process
            if verbose >= 1:start_time = time.time()

            # -----------------------------------------PUT ANY DETECTION MODEL HERE -----------------------------------------------------------------
            if YOLOVER=='V3':
                    # import ipdb; ipdb.set_trace()
                    yolo_dets, scores, class_ID=YOLOV3(self.detector,frame_left)
                    # import ipdb; ipdb.set_trace()
            
            else:
                yolo_dets = self.detector.detect(frame_left.copy(), plot_bb = False)  # Get the detections
                frame_left = cv2.cvtColor(frame_left, cv2.COLOR_BGR2RGB)
                frame_right = cv2.cvtColor(frame_right, cv2.COLOR_BGR2RGB)

            # if yolo_dets is None:
            #     bboxes = []
            #     scores = []
            #     classes = []
            #     num_objects = 0
            
            # ---------------------------------------- DETECTION PART COMPLETED ---------------------------------------------------------------------
            
            # names = np.array(class_ID)
            # import ipdb; ipdb.set_trace()
            count = len(class_ID) #Class number. It will be zero for our case since there is only one class
            names=np.array(["Apple"]*count)

            
            # ---------------------------------------- Find 3D information -------------------------------------------------------------------------- 
            
            # cv2.namedWindow('Right',cv2.WINDOW_NORMAL)
            # cv2.imshow('Right',frame_right)
            disparity=ut.find_disparity_RAFT(frame_left, frame_right,model_path=None,resize=(1024,1366),display=False)
            mask=ut.segmentation(frame_left,scale=0.4,display=False)
            image, point_mask=ut.find_center(yolo_dets, frame_left, mask)            
            point_disparity=cv2.bitwise_and(disparity,point_mask)
            points_3d=ut.obtain_3d_volume(point_disparity,frame_left,save_file=True, frame_num=frame_num)

            if len(points_3d)!=len(yolo_dets):
                print("*"*20,"STOPPING CODE","*"*20)
                print("All boxes do not have a center point")
                break
                    
            if count_objects:
                cv2.putText(frame_left, "Objects being tracked: {}".format(count), (5, 35), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1.5, (255,255,255), 2)

            # ---------------------------------- DeepSORT tacker work starts here ------------------------------------------------------------
            
            features = self.encoder(frame_left, yolo_dets) # encode detections and feed to tracker. [No of BB / detections per frame, embed_size]
            detections = [Detection(bbox, score, class_name, feature, point_3D) for bbox, score, class_name, feature, point_3D \
                                    in zip(yolo_dets, scores, names, features,points_3d)] # [No of BB per frame] deep_sort.detection.Detection object
            cmap = plt.get_cmap('tab20b') #initialize color map
            colors = [cmap(i)[:3] for i in np.linspace(0, 1, 20)]   

            self.tracker.predict()  # Call the tracker
            unmatched_tracks,unmatched_detections=self.tracker.update(detections) #  update using Kalman Gain

            # import ipdb;ipdb.set_trace()
            for track in self.tracker.tracks:  # update new findings AKA tracks                
                
                if not track.is_confirmed() or track.time_since_update > 1:
                    continue 
                bbox = track.to_tlbr()
                class_name = track.get_class()

                color = colors[int(track.track_id) % len(colors)]  # draw bbox on screen
                color = [i * 255 for i in color]
                text_color=(255,255,255)
                if track.time_since_update<2:
                    pass
                else:
                    color=(255,255,255)
                    text_color=(0,0,0)
                # import ipdb; ipdb.set_trace()
                # print(bbox)
                cv2.rectangle(frame_left, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), color, 2)
                cv2.rectangle(frame_left, (int(bbox[0]), int(bbox[1]-30)), (int(bbox[0])+(len(class_name)+len(str(track.track_id)))*30, int(bbox[1])), color, -1) #To make a solid rectangle box to write text on
                # cv2.putText(frame_left, class_name + ":" + str(track.track_id)+'-'+str(round(track.confidence,2)),(int(bbox[0]), int(bbox[1]-11)),0, 0.8, (text_color),2, lineType=cv2.LINE_AA)
                cv2.putText(frame_left, class_name + " " + str(track.track_id)+':'+str(round(ut.occlusion_score(bbox,mask),3)),(int(bbox[0]), int(bbox[1]-11)),0, 0.8, (text_color),2, lineType=cv2.LINE_AA)  
                cv2.putText(frame_left, "Frame_num:"+str(frame_num),(len(frame_left[0])-300,len(frame_left)-100),0, 1.2, (255,255,255),2, lineType=cv2.LINE_AA)  
                if verbose == 2:
                    print("Tracker ID: {}, Class: {},  BBox Coords (xmin, ymin, xmax, ymax): {}".format(str(track.track_id), class_name, (int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3]))))
                    
            # -------------------------------- Tracker work ENDS here -----------------------------------------------------------------------

            if verbose >= 1:
                fps = 1.0 / (time.time() - start_time) # calculate frames per second of running detections
                if not count_objects: print(f"Processed frame no: {frame_num} || Current FPS: {round(fps,2)}")
                else: print(f"Processed frame no: {frame_num} || Current FPS: {round(fps,2)} || Objects tracked: {count}")
            
            result = np.asarray(frame_left)
            result = cv2.cvtColor(frame_left, cv2.COLOR_RGB2BGR)
            
            if output: out.write(result) # save output video

            if show_live:
                cv2.namedWindow("Output Video",cv2.WINDOW_NORMAL)
                cv2.imshow("Output Video", result)
                if cv2.waitKey(1) & 0xFF == ord('q'): break

            if frame_num>9 and frame_num<100:
                name=dir_path+'/00'+str(frame_num)+'_.png'
            elif frame_num>100:
                name=dir_path+'/0'+str(frame_num)+'_.png'
            elif frame_num<10:
                name=dir_path+'/000'+str(frame_num)+'_.png'
            cv2.imwrite(name,frame_left)
        cv2.destroyAllWindows()


def YOLOV3(model, frame):

    blob = cv2.dnn.blobFromImage(frame, 1.0/255,(416,416),(0,0,0),swapRB = True,crop= False)
    model.setInput(blob)

    hight,width,_=frame.shape

    model.enableWinograd(False)
    output_layers_name = model.getUnconnectedOutLayersNames()

    layerOutputs = model.forward(output_layers_name)

    boxes= []
    confidences= []
    class_ids= []

    for output in layerOutputs:
        for detection in output:
            score= detection[5:]
            class_id= np.argmax(score)
            confidence= score[class_id]

            if confidence>0.7:
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * hight)
                # ipdb.set_trace()
                w = int(detection[2] * width)
                h = int(detection[3]* hight)
                x = int(center_x - w/2)
                y = int(center_y - h/2)
                boxes.append([x,y,w,h])
                confidences.append((float(confidence)))
                class_ids.append(class_id)

    indexes = cv2.dnn.NMSBoxes(boxes,confidences,.3,.4)
    # indexes = cv2.dnn.NMSBoxes(boxes,confidences,.15,.3)
    
    return [boxes[i]for i in indexes],[confidences[i]for i in indexes], [class_ids[i] for i in indexes]
