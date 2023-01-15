import cv2
import numpy as np
import utils as ut
import argparse

#Defining parameters
def get_args():
    parser=argparse.ArgumentParser(description='Run DeepSORT in 3D')
    parser.add_argument('--left',default='./Dataset/image_rect_left/L0050.jpeg',type=str,help='Left Stereo Image')
    parser.add_argument('--right',default='./Dataset/image_rect_right/R0050.jpeg',type=str,help='Right Stereo Image')
    parser.add_argument('--size',default=(512,683),type=tuple,help='Right Stereo Image')

    return parser.parse_args()


if __name__=='__main__':

    args=get_args()
    # import ipdb;ipdb.set_trace()
    
    disparity=ut.find_disparity_RAFT(args.left,args.right,resize=args.size)
    


