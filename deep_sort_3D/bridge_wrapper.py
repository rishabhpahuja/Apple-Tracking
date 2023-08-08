import sys
import os
sys.path.append('../')
sys.path.append('../Disparity')
sys.path.append('../Segmentation')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # comment out below line to enable tensorflow logging outputs
import time
# import tensorflow as tf

# physical_devices = tf.config.experimental.list_physical_devices('GPU')
# if len(physical_devices) > 0:
#     tf.config.experimental.set_memory_growth(physical_devices[0], True)
#     tf.config.experimental.set_memory_growth(physical_devices[1], True)
    
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
# from tensorflow.compat.v1 import ConfigProto # DeepSORT official implementation uses tf1.x so we have to do some modifications to avoid errors

# deep sort imports
from deep_sort import preprocessing, nn_matching
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker

# import from helpers
# from tracking_helpers import read_class_names, create_box_encoder
# from detection_helpers import *
# import ipdb; ipdb.set_trace()
import Utils as ut
import csv


 # load configuration for object detector
# config = ConfigProto()
# config.gpu_options.allow_growth = True

chi2inv95 = {
    1: 3.8415,
    2: 5.9915,
    3: 7.8147,
    4: 9.4877,
    5: 11.070,
    6: 12.592,
    7: 14.067,
    8: 15.507,
    9: 16.919}


class YOLOv8_SORT_3D:
    '''
    Class to Wrap ANY detector  of YOLO type with DeepSORT
    '''
    def __init__(self, detector, segment,disparity,rover_coor_path,nn_budget:float=None, nms_max_overlap:float=1.0, base_coordinate=np.array([0,0,0]),vel=np.array([12,0,0]),
                    _st_weight_position=300 ):
        '''
        args: 
            detector: object of YOLO models or any model which gives you detections as [x1,y1,x2,y2,scores, class]
            segment: object of UNet segmentation model which gives out the segmentation mask
            disparity: object of RAFT stereo model to generate disparity map of the stereo pair
            max_cosine_distance: Cosine Distance threshold for "SAME" person matching
            nn_budget:  If not None, fix samples per class to at most this number. Removes the oldest samples when the budget is reached.
            nms_max_overlap: Maximum NMs allowed for the tracker
            base_cord: initial coordinates of the rover (world coordinate origin)
            v: velocity vector of the rover
        '''
        self.detector = detector
        self.segment=segment
        self.disparity=disparity
        self.coco_names_path = ['apples']
        self.nms_max_overlap = nms_max_overlap


        # initialize sort3D
        metric=nn_matching.NearestNeighborDistanceMetric("mahalanobis",matching_threshold=chi2inv95[3],budget=nn_budget)
        self.tracker = Tracker(metric=metric,v=vel,mean=base_coordinate,_st_weight_position=_st_weight_position) # initialize tracker
        #Initialize rover
        self.base_cord=base_coordinate
        self.v=vel

    def track_video(self,left_video:str, right_video:str,output:str, skip_frames:int=0, show_live:bool=True,
                     count_objects:bool=False, verbose:int = 0, debug=False, save_frames=True, frame_save_dir_path='./Tests/3D_cost'):
        '''
        Track any given webcam or video
        args: 
            video: path to input video or set to 0 for webcam
            output: path to output video
            skip_frames: Skip every nth frame. After saving the video, it'll have very visuals experience due to skipped frames
            show_live: Whether to show live video tracking. Press the key 'q' to quit
            count_objects: count objects being tracked on screen
            verbose: print details on the screen allowed values 0,1,2
            frame_save_dir_path: Path to save the frames of the inputted video wrt left camera
            debug: To display the calculated apple centers, segmented apples and bounding boxes
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

            #Downsample image
            # frame_left=cv2.resize(frame_left,dsize=(800,640),interpolation=cv2.INTER_LANCZOS4)
            # frame_right=cv2.resize(frame_right,dsize=(800,640),interpolation=cv2.INTER_LANCZOS4)

            if skip_frames and not frame_num % skip_frames: continue # skip every nth frame. When every frame is not important, you can use this to fasten the process
            if verbose >= 1:start_time = time.time()

            ############################  Detection Model #############################################
            yolo_dets,scores,_=self.detector.pred(frame_left.copy(),debug=False)       
                        
            ############################ Find 3D information ###########################################
            disparity=self.disparity.find_disparity(frame_left, frame_right)           
            seg_mask=self.segment.predict_img(cv2.cvtColor(frame_left.copy(),cv2.COLOR_BGR2RGB))

            image, point_mask,yolo_dets,points_2D=ut.find_center(yolo_dets, frame_left.copy(), seg_mask, debug=debug)
            '''
            Image: Left frame showing fruit center and bounding boxes if debug=True        
            '''
            if debug:
                cv2.namedWindow("Fruit center & Bounding Box",cv2.WINDOW_NORMAL)
                cv2.namedWindow("Only Fruit",cv2.WINDOW_NORMAL)

                image_added=cv2.addWeighted(frame_left,1.0,cv2.cvtColor(seg_mask,cv2.COLOR_BGR2RGB),0.5,0.0)

                cv2.imshow("Only Fruit",image_added)
                cv2.imshow("Fruit center & Bounding Box",image)

                cv2.waitKey(0)
                cv2.destroyAllWindows()
                
            disparity=cv2.resize(disparity,dsize=(point_mask.shape[1],point_mask.shape[0]),interpolation=cv2.INTER_LANCZOS4)

            points_3d=ut.obtain_3d_volume(disparity,frame_left.copy(),point_mask=point_mask,fruit_mask=seg_mask,points_2D=points_2D,\
                                          save_file=True, frame_num=frame_num)

            if frame_num!=1: #Update rover coordinates only from 2nd frame
                points_3d=self.rover_detec(self.base_cord)+points_3d #Shifting 3D cordinates from rover to world origin

            else: 
                points_3d=self.base_cord+points_3d
 
            names=np.array(["Apple"]*len(yolo_dets))
        
            # import ipdb; ipdb.set_trace()
            if len(points_3d)!=len(yolo_dets): #Condition to check if all the bounding boxes have a fruit
                print("*"*20,"STOPPING CODE","*"*20)
                print("All boxes do not have a center point")
                break
                    
            if count_objects:
                cv2.putText(frame_left, "Objects being tracked: {}".format(len(names)), (5, 35), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1.5, (255,255,255), 2)

            ######################################### SORT_3D ##############################################

            position_3D=np.vstack((self.base_cord,points_3d))  #Saving measured objects: rover and apples in one data structure
            yolo_dets=np.vstack((np.array([0,0,0,0]),yolo_dets)) # Making 2d location of the apples of the similar structure, i.e. apple location start from pos=1
            scores=np.hstack((np.array([0]),scores))            #Detection scores; Concatenting 0 at pos=0 so have similar structure
            detections = Detection(yolo_dets, scores, names,  position_3D) # detection object for rover and all the apples

            if frame_num!=1:
                self.tracker.predict()  # Call the tracker except for frame 1 since tracker object of detections is made at the end of frame 1

            matches=self.tracker.update(detections) #  update using Kalman Gain

            # Visualizing           
            cmap = plt.get_cmap('tab20b') #initialize color map
            colors = [cmap(i)[:3] for i in np.linspace(0, 1, 20)]   
            
            if len(matches)!=0:
                matches=np.vstack((matches)) #Add matched tracks to the datastructure

            for match in matches:  # update new findings AKA tracks                
                if self.tracker.tracks.is_confirmed([match[0]])==False:
                    # import ipdb; ipdb.set_trace()
                    continue 
                bbox = detections.points_2D[match[1]]
                class_name = 'Apple'

                color = colors[int(self.tracker.tracks.track_id[match[0]]) % len(colors)]  # draw bbox on screen
                color = [i * 255 for i in color]
                text_color=(255,255,255)

                cv2.rectangle(frame_left, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), color, 2)
                cv2.rectangle(frame_left, (int(bbox[0]), int(bbox[1]-30)), (int(bbox[0])+(len(class_name)+len(str(self.tracker.tracks.track_id[match[0]])))*10, int(bbox[1])), color, -1) #To make a solid rectangle box to write text on
                cv2.putText(frame_left, class_name + ":" + str(self.tracker.tracks.track_id[match[0]]),(int(bbox[0]), int(bbox[1]-11)),0, 0.5, (text_color),2, lineType=cv2.LINE_AA)
                # cv2.putText(frame_left, class_name + " " + str(track.track_id)+':'+str(round(ut.occlusion_score(bbox,mask),3)),(int(bbox[0]), int(bbox[1]-11)),0, 0.8, (text_color),2, lineType=cv2.LINE_AA)  
                cv2.putText(frame_left, "Frame_num:"+str(frame_num),(len(frame_left[0])-100,len(frame_left)-50),0, 0.5, (255,255,255),2, lineType=cv2.LINE_AA)
            
                if verbose == 2:
                    print("Tracker ID: {}, Class: {},  BBox Coords (xmin, ymin, xmax, ymax): {}".format(str(self.tracker.tracks.track_id[match[0]]), class_name, (int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3]))))
                    
            # -------------------------------- Tracker work ENDS here -----------------------------------------------------------------------

            if verbose >= 1:
                fps = 1.0 / (time.time() - start_time) # calculate frames per second of running detections
                if not count_objects: print(f"Processed frame no: {frame_num} || Current FPS: {round(fps,2)}")
                else: print(f"Processed frame no: {frame_num} || Current FPS: {round(fps,2)} || Objects tracked: {len(self.tracker.tracks.mean_2D)}")
            
            result = np.asarray(frame_left)
            result = cv2.cvtColor(frame_left, cv2.COLOR_RGB2BGR)
            
            if output: out.write(result) # save output video

            if show_live:
                cv2.namedWindow("Output Video",cv2.WINDOW_NORMAL)
                cv2.imshow("Output Video", result)
                if cv2.waitKey(1) & 0xFF == ord('q'): break
            
            if save_frames:
                if frame_num>9 and frame_num<100:
                    name=frame_save_dir_path+'/00'+str(frame_num)+'_.png'
                    # print(name)
                elif frame_num>100:
                    name=frame_save_dir_path+'/0'+str(frame_num)+'_.png'
                    # print(name)
                elif frame_num<10:
                    name=frame_save_dir_path+'/000'+str(frame_num)+'_.png'
                print(name)
                cv2.imwrite(name,frame_left)
        cv2.destroyAllWindows()


    def rover_detec(self,pos):

        T=1
        M=np.array([[T,0,0],[0,T,0],[0,0,T]])
        noise= np.random.normal(0,1,3)
        self.base_cord=pos.T + M@self.v + noise
        return self.base_cord

    # def rover_detec(self,frame):

    #     return self.rover_coor[frame]
