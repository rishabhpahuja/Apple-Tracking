import sys
import os
sys.path.append(os.getcwd()+'/Disparity')
sys.path.append(os.getcwd()+'/Segmentation')
import cv2
import numpy as np
import Utils as ut
import argparse
# import Segmentation as sg
# import ipdb;ipdb.set_Trace()
#Defining parameters
def get_args():
    parser=argparse.ArgumentParser(description='Run DeepSORT in 3D')
    
    parser.add_argument('--debug',default=True,type=bool,help='Bool value whether to see the output of each step')

    #Disparity model parameters
    parser.add_argument('--disparity_model',default=None,type=str,help='Path for disparity model')
    parser.add_argument('--left',default='./L0050.jpeg',type=str,help='Left Stereo Image')
    parser.add_argument('--right',default='./Dataset/image_rect_right/R0050.jpeg',type=str,help='Right Stereo Image')
    parser.add_argument('--size',default=(1024,1366),type=tuple,help='Size for downsizing image')
    # parser.add_argument('--size',default=None,type=tuple,help='Size for downsizing image')
    
    #Segmentation model parameters
    parser.add_argument('--seg_model',default=None,type=str,help='path for segmentation model')
    parser.add_argument('--scale',default=0.4,type=int,help='Downscaling factor for image inputed')
    parser.add_argument('--match',default=False,type=bool,help='Bool value whether to perform histogram equilizzation or not')

    #Find fruit center bounding box
    parser.add_argument('--csv',default='./data_10_Jan_latest/Detections left/L0050.csv',type=str,help='Path to ground truth of bounding boxes')
    
    return parser.parse_args()


if __name__=='__main__':

    args=get_args()
    # import ipdb;ipdb.set_trace()
    
    #Find diparity map
    disparity=ut.find_disparity_RAFT(args.left,args.right,model_path=args.disparity_model,resize=args.size)
    
    #Find segmentation mask for fruit
    mask=ut.segmentation(args.left,args.scale,match=args.match)
    
    #Find the center of each fruit. Image show the center marked on each fruit while point_mask is a mask to show fruit center
    image, point_mask=ut.find_center(args.csv, cv2.imread(args.left), mask)
    added=cv2.addWeighted(image,1.0,cv2.cvtColor(mask,cv2.COLOR_GRAY2BGR),0.5,0)

    point_disparity=cv2.bitwise_and(disparity,point_mask)
    ut.obtain_3d_volume(point_disparity,args.left)
    
    if args.debug:
        cv2.namedWindow("Image",cv2.WINDOW_NORMAL)
        cv2.namedWindow("Added",cv2.WINDOW_NORMAL)
        cv2.namedWindow("Point Mask",cv2.WINDOW_NORMAL)
        cv2.namedWindow("Point Disparity Map",cv2.WINDOW_NORMAL)
        cv2.namedWindow("Disparity", cv2.WINDOW_NORMAL)
        cv2.imshow("Image",image)
        cv2.imshow("Added",added)
        cv2.imshow("Point Mask",point_mask)
        cv2.imshow("Point Disparity Map",point_disparity)
        cv2.imshow("Disparity",disparity)

        cv2.waitKey(0)
        cv2.destroyAllWindows()

        cv2.imwrite('image_box_center.jpg',image)
        cv2.imwrite('image_box_segment_center.jpg',added)
        cv2.imwrite('point_mask.jpg',point_mask)
        cv2.imwrite('Disparity.jpg',disparity)


