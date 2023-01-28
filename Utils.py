import glob
import os
import sys
sys.path.append(os.getcwd()+'/Disparity')
sys.path.append(os.getcwd()+'/Segmentation')
import cv2
from skimage import exposure
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
# import open3d as od
import csv
import open3d as o3d

from Disparity import demo
# import ipdb;ipdb.set_trace()
from Segmentation import predict as Unet

def make_video_from_frames(path, Iframe=0, Fframe=100,REF=False):

    img_array=[]

    files=sorted(glob.glob(path))[Iframe:Fframe]
    #files=sorted(glob.glob(path))
    # import ipdb;ipdb.set_trace()
    h,w,c=cv2.imread(files[0]).shape
    size=(w,h)

    for i,file in enumerate(files):
        img=cv2.imread(file)
        img_array.append(img)
    
    video=cv2.VideoWriter('Baseline.mp4',cv2.VideoWriter_fourcc(*'mp4v'),1,size, isColor=True)

    if REF:
        ref=cv2.imread('AAA_4420.png')

    for image in img_array:
        
        if REF:
            image=exposure.match_histograms(image, ref, channel_axis==2)
        
        video.write(image)
    
    video.release()

def clahe_(img_path, display=True, save=True): #Histogram equalization to distribute pixel values uniformely

    img=cv2.imread(img_path,0)
    clahe=cv2.createCLAHE(clipLimit=2.0,tileGridSize=(8,8))
    equalized=clahe.apply(img)
    if display:
        cv2.namedWindow("Output",cv2.WINDOW_NORMAL)
        cv2.imshow('Output',equalized)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    if save:
        cv2.imwrite(img_path.split('/'),equalized)

def match_hist(ref_img_path, img_path, display=False, save=True):

    ref_img= cv2.imread(ref_img_path,1)
    img=cv2.imread(img_path,1)
    # import ipdb;ipdb.set_trace()
    multi= True if ref_img.shape[-1]>1 else False
    matched = exposure.match_histograms(img, ref_img, multichannel=multi)

    if display:
        cv2.namedWindow("Output",cv2.WINDOW_NORMAL)
        cv2.imshow('Output',matched)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    if save:
        cv2.imwrite('./Dataset/image_rect_right/'+img_path.split('/')[-1],matched)
    
    return matched

'''

The function (find_disparity) is taken from official tutorial of openCV
https://opencv24-python-tutorials.readthedocs.io/en/latest/py_tutorials/py_calib3d/py_depthmap/py_depthmap.html

'''
def find_disparity(image_left, image_right, display= True, save=True):
    
    
    imgL=cv2.imread(image_left,1)
    imgR=cv2.imread(image_right,1)
    # import ipdb; ipdb.set_trace()
    # imgR=cv2.resize(imgR,(int(1536/6),int(2048/6)))
    # imgL=cv2.resize(imgL,(int(1536/6),int(2048/6)))
    stereo = cv2.StereoSGBM_create(
                                    # minDisparity=0, # 0
                                   numDisparities=200, # 320
                                   blockSize=5, # 3, 1
                                #    disp12MaxDiff=0, # 0
                                #    uniquenessRatio=15, # 15
                                #    speckleWindowSize=100, # 175
                                #    speckleRange=2, # 20
                                #    P1=100,
                                #    P2=200,
                                #    preFilterCap=31
                                   )
    disparity=stereo.compute(imgL, imgR)
    # import ipdb; ipdb.set_trace()

    if display:
        cv2.namedWindow("Left Image",cv2.WINDOW_NORMAL)
        cv2.namedWindow("Right Image",cv2.WINDOW_NORMAL)
        # cv2.namedWindow("Disparity Image",cv2.WINDOW_NORMAL)
        cv2.imshow("Left Image",imgL)
        cv2.imshow("Right Image",imgR)
        # cv2.imshow("Disparity Image",disparity)
        plt.imshow(disparity)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        plt.show()
    
    if save:
        plt.imsave("./DisparityImage_plt.png",disparity)
        cv2.imwrite("./DisparityImage.png",disparity)

def segmentation(image,scale,model_path=None,display=True,match=False):

    '''
    image: numoy, image
    '''
    # image=cv2.imread(image)
    # plt.imshow(image)
    # plt.show()
    if match:  
        # import ipdb;ipdb.set_trace()      
        image=match_hist('cam1_image003893.png',image)
    seg_ob=Unet.Segmentation(model_path)
    mask=np.asarray(seg_ob.predict_img(cv2.cvtColor(image,cv2.COLOR_BGR2RGB),scale_factor=scale),np.uint8)
    # mask=cv2.cvtColor(np.asarray(mask,np.uint8),cv2.COLOR_GRAY2BGR)
    # import ipdb; ipdb.set_trace()
    # mask.save('1.png')

    if display:
        
        added=cv2.addWeighted(image,1.2,cv2.cvtColor(np.asarray(mask,np.uint8),cv2.COLOR_GRAY2BGR),0.8,0)
        cv2.namedWindow('Mask',cv2.WINDOW_NORMAL)
        cv2.namedWindow('Added',cv2.WINDOW_NORMAL)
        cv2.imshow('Mask',mask)
        cv2.imshow('Added',added)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return mask      
    

def find_disparity_RAFT(imgl,imgr,display=False,save=False,resize=None,model_path=None):

    '''
    imgl: Left stereo image path (numpy)
    imgr: Right stereo image path (numpy)
    display: bool, if calculated disparity has to be viewed 
    '''
    
    disparity_ob=demo.Disparity(resize,model_path)    
    disparity=disparity_ob.find_disparity(imgl,imgr,save).astype(np.uint8)

    if display:
        cv2.namedWindow('Disparity Image', cv2.WINDOW_NORMAL)
        cv2.imshow('Disparity Image', disparity)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return disparity
    

def crop_disparity_map(disparity_map, locations, map_type=0,display=True,save=False):

    '''
    map_type=0 means point cloud of only fruit will be generated
    map_type=1 means fruit shall be represented by a single point
    '''
    with open(locations,'r') as file:
        csvreader=csv.reader(file)
        map=np.zeros_like(disparity_map)
        # import ipdb; ipdb.set_trace()
        for row in csvreader:
            if map_type==1:
                map[int((int(row[1])+int(row[3]))/2),int((int(row[0])+int(row[2]))/2)]=255
            
            elif map_type==0:
                map[10+int(row[1]):10+int(row[3]),int(row[0]):int(row[2])]=255
    
    cropped_disparity_map=cv2.bitwise_and(disparity_map, map)
    if display:
        cv2.namedWindow('Cropped_disparity',cv2.WINDOW_NORMAL)
        cv2.namedWindow('Orignal_disparity',cv2.WINDOW_NORMAL)
        cv2.imshow('Cropped_disparity',cropped_disparity_map)
        cv2.imshow('Orignal_disparity',disparity_map)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    if save:
        # import ipdb; ipdb.set_trace()
        cv2.imwrite('Cropped_disparity_map.png',cropped_disparity_map)
    return cropped_disparity_map

def find_contour_center(mask_cropped,image_,row,point_mask ,display=False):

    '''
    mask_cropped: The mask cropped along bounding box
    image_: Left stereo image
    row: The coordinates of one bounding box
    point_mask: zeros mask saving location of each fruit
    '''

    contours, _ = cv2.findContours(mask_cropped, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if display:
        cv2.imshow('Cropped Image',mask_cropped)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    if len(contours)==0:
        print('*'*50)
    
    elif len(contours)==1:
        M=cv2.moments(contours[0])        
        # try:
        cx=int(M['m10']/M['m00'])+int(row[0])
        cy=int(M['m01']/M['m00'])+int(row[1])
        cv2.circle(image_, (cx, cy), 10, (0, 0, 255), -1)
        point_mask[cy,cx]=255
            # cv2.circle(point_mask, (cx, cy), 10, (255, 255,255), -1) 
        # except:
        #     cv2.namedWindow('cropped',cv2.WINDOW_NORMAL)
        #     mask_cropped=cv2.cvtColor(mask_cropped,cv2.COLOR_GRAY2BGR)
        #     cv2.drawContours(mask_cropped,contours,-1,(255,0,0),10)
        #     cv2.imshow('cropped', mask_cropped)
        #     cv2.waitKey(0)
        #     cv2.destroyAllWindows()
            # import ipdb; ipdb.set_trace()
        return image_,point_mask 

    elif len(contours)>1:
        max=0
        for contour in contours:
            k=cv2.contourArea(contour) 
            if max<k:
                max=cv2.contourArea(contour)
                c=contour
        M=cv2.moments(c)
        cx=int(M['m10']/M['m00'])+int(row[0])
        cy=int(M['m01']/M['m00'])+int(row[1])
        cv2.circle(image_, (cx, cy), 10, (0, 0, 255), -1) 
        point_mask[cy,cx]=255                   
        # cv2.circle(point_mask, (cx, cy), 10, (255, 255,255), -1)
    return image_,point_mask

#Find bounding boxes
def find_center(detections,image,mask):

    '''
    csv_file: file consisting of bounding box location of left stereo image
    image: numpy, left stereo image
    mask: segmentation mask of left stereo image
    '''
    point_mask=np.zeros((image.shape[0],image.shape[1]),np.uint8)    
    # import ipdb; ipdb.set_trace()      
    for i,row in enumerate(detections):
        image,point_mask=find_contour_center(mask[int(row[1]):int(row[1]+row[3]),int(row[0]):int(row[0]+row[2])],image,row,point_mask,display=False)
        # cv2.rectangle(image,(int(row[0]),int(row[1])),(int(row[0]+row[2]),int(row[1]+row[3])),(255,255,0),4)
    
    return image, point_mask

def obtain_3d_volume(disparity_map,left_image , fruit_location=None,only_fruit=None,\
                    single_point=False,save_file=False,frame_num=None):

    # disparity_map=cv2.imread(disparity_map,0)

    '''
    map_type=0 means point cloud of only fruit will be generated
    map_type=1 means fruit shall be represented by a single point
    
    K: Intrinsic camera matrix for raw distorted images. Projects 3D points in the camera coordinate frame to 2D pixel coordinates
    using focal lengths and principal point
    K=[[fx 0 cs]
        [0 fy cy]
        [0  0  1]]

    R: Rectification matrix (stereo cameras only). A rotation matrix aligning camera coordinate system to the ideal stereo image 
    plane so that epipolar lines in both stereo images are parallel.

    P: By convention, this matrix specifies the intrinsic (camera) matrix of the processed (rectified) image. That is, the left 3x3 portion
    is the normal camera intrinsic matrix for the rectified image. It projects 3D points in the camera coordinate frame to 2D pixel
    coordinates using the focal lengths (fx', fy') and principal point (cx', cy') - these may differ from the values in K.
        [fx'  0  cx' Tx]
    P = [ 0  fy' cy' Ty]
        [ 0   0   1   0]
    the fourth column [Tx Ty 0]' is related to the position of the optical center of the second camera in the first camera's frame
    Tx = -fx' * B, where B is the baseline between the cameras
    '''

        
    Kr=np.array([[1052.350202570253, 0.0, 1031.808590719438],
                            [0.0, 1051.888280928595, 771.0661229952285],
                            [0.0, 0.0, 1.0]]) #Intrinsic parameter to convert camera frame to image frame)
    
    Rr= np.array([[0.9996990506423458, 0.002517310775589418, 0.02440228045187234],\
                    [-0.002557834410859948, 0.9999954009677927, 0.001629578592843086],\
                    [-0.02439806606924718, -0.001691505164855578, 0.9997008918583387]]) 
    
    Dr= np.array([-0.03098864107712216, 0.04051128735759788, -0.001361885214239114, -0.0008816601680637922, 0.0]) #Distortion matrix
    Pr=np.array([1142.280571388607, 0.0, 988.5422821044922, 0.0, 0.0, 1142.280571388607, 782.6398086547852, 0.0, 0.0, 0.0, 1.0, 0.0]).reshape((3,4))
    
    Kl= np.array([1052.382387969279, 0.0, 1058.58421357867, 
                0.0, 1052.123571352367, 800.4517901498787, 
                0.0, 0.0, 1.0]).reshape((3,3))

    Rl=np.array([0.9996673945529461, 0.001496854974222017, 0.02574606169709769, 
                -0.001454093723616705, 0.9999975323978124, -0.001679526638459951,
                -0.02574851217386264, 0.001641530832029927, 0.9996671043389194]).reshape(3,3)

    Dl=np.array([-0.02203560874783666, 0.02887488453134448, 0.000593652652803611, -0.00298638958591028, 0.0])
    Pl=np.array([1142.280571388607, 0.0, 988.5422821044922, -136.2942110803947, 0.0, 1142.280571388607, 782.6398086547852, 0.0, 0.0, 0.0, 1.0, 0.0]).reshape(3,4)
    
    rev_proj_matrix = np.zeros((4,4))
    
    cv2.stereoRectify(cameraMatrix1=Pr[0:3,0:3], distCoeffs1=0, cameraMatrix2=Pl[0:3,0:3], distCoeffs2=0, R=np.linalg.inv(Rr)@Rl, 
                            T=np.array([[-136.2942110803947],[0],[0]]),imageSize=(2048,1536),Q = rev_proj_matrix)

    # Q = np.float32([[1, 0, 0, 0],
	# 			[0, -1, 0, 0],
	# 			[0, 0, 1142, 0],
	# 			[0, 0, 0, 1]])
    '''
    R matrix is used to convert one camera frame to a central frame.
    
    REF: http://docs.ros.org/en/melodic/api/sensor_msgs/html/msg/CameraInfo.html
        https://wiki.ros.org/image_pipeline/CameraInfo
    '''
    points_3D = cv2.reprojectImageTo3D(disparity_map, rev_proj_matrix) #finding co-ordinates of point cloud from disparity image
    # b=np.array([[59],[990],[disparity_map[59,990]],[1]]).astype(np.float32)
    # import ipdb;ipdb.set_trace()
    # a=rev_proj_matrix@b
    # d=a[:3]/a[3,0]
    # imgL=cv2.imread(left_image,1)

    # import ipdb; ipdb.set_trace()
    mask_map = disparity_map > disparity_map.min()   #condition to remove pixel with zero values
    output_points = points_3D[mask_map]

    print(output_points.shape) #[0] of output gives number of vertices

    if save_file:
        colors = cv2.cvtColor(left_image, cv2.COLOR_BGR2RGB)
        output_colors = colors[mask_map].astype(int)     #saving colour of each point
        file_name='default_'+str(frame_num)+'.ply'
        if only_fruit:
            file_name='only_fruit_85.ply'
            disparity_map=crop_disparity_map(disparity_map,locations=fruit_location,map_type=0 ,display=True)
        
        elif single_point:
            file_name='single_point.ply'
            disparity_map=crop_disparity_map(disparity_map,locations=fruit_location,map_type=1 ,display=False)
    # import ipdb; ipdb.set_trace()
        header = '''ply
        format ascii 1.0
        comment - Made for Apple Tracking
        comment - This file represents a cube's corner vertices
        element vertex ''' + str(output_points.shape[0])+\
        '''
        property float32 x
        property float32 y
        property float32 z
        property uint8 red
        property uint8 green
        property uint8 blue
        end_header
        '''
        #writing ply file
        # f.write(a)
        # import ipdb; ipdb.set_trace()

        verts=np.hstack([output_points,output_colors])
        # for i in tqdm(range(len(output_points))):
        #     a = np.array2string(output_points[i,0]) + " " + \
        #         np.array2string(output_points[i,1])+ " " + \
        #         np.array2string(output_points[i,2])+ "  "+ \
        #         np.array2string(output_colors[i,0]) + " " +\
        #         np.array2string(output_colors[i,1])+ " " + \
        #         np.array2string(output_colors[i,2]) + "\n"

        with open(file_name, 'wb') as f:     #opening file to make a ply file

            f.write((header % dict(vert_num=len(verts))).encode('utf-8'))
            np.savetxt(f, verts, fmt='%f %f %f %d %d %d ')
                
        f.close()
        # pcd=o3d.io.read_point_cloud('./test.ply')
    return output_points

def display_inlier_outlier(cloud, ind):
    inlier_cloud = cloud.select_by_index(ind)
    outlier_cloud = cloud.select_by_index(ind, invert=True)

    print("Showing outliers (red) and inliers (gray): ")
    outlier_cloud.paint_uniform_color([1, 0, 0])
    inlier_cloud.paint_uniform_color([0.8, 0.8, 0.8])
    o3d.visualization.draw_geometries([inlier_cloud, outlier_cloud],
                                      zoom=0.3412,
                                      front=[0.4257, -0.2125, -0.8795],
                                      lookat=[2.6172, 2.0475, 1.532],
                                      up=[-0.0694, -0.9768, 0.2024])

def clean_point_cloud_points(file_path, csv_file,display=False,save=False):

    # import ipdb; ipdb.set_trace()
    point_cloud=o3d.io.read_point_cloud(file_path)
    cl,ind=point_cloud.remove_radius_outlier(nb_points=50,radius=10)
    inlier_cloud = point_cloud.select_by_index(ind)
    numpy_inliner_cloud=np.asarray(inlier_cloud.points)

    fruit_loc=list()

    f = open('test_remove_green.ply', 'w')

    a = '''ply
    format ascii 1.0
    comment - Made for Apple Tracking
    comment - This file represents a cube's corner vertices
    element vertex ''' + str(len())+\
    '''
    property float32 x
    property float32 y
    property float32 z
    property uint8 red
    property uint8 green
    property uint8 blue
    end_header
    '''
    #writing ply file
    f.write(a)

    with open(csv_file) as csvfile:
        spamreader=csv.reader(csvfile)
        for row in spamreader:
            # import ipdb;ipdb.set_trace()
            x_filter=np.bitwise_and(np.asarray(numpy_inliner_cloud[:,0],int)>=int(row[0]), np.asarray(numpy_inliner_cloud[:,0],int)<=int(row[2]))
            rem_pc=numpy_inliner_cloud[x_filter]
            # import ipdb; ipdb.set_trace()
            y_filter=np.bitwise_and(rem_pc[:,1]>=int(row[1]), rem_pc[:,1]<=int(row[3]))
            rem_pc=rem_pc[y_filter].mean(axis=0)
            
        
    if display:
        display_inlier_outlier(point_cloud, ind)
    
    if save:
        # o3d.io.write_point_cloud('cleaned_test.ply',inlier_cloud)
        o3d.io.write_point_cloud('single_point.ply',inlier_cloud)


def image_shift(imgr_path,imgl_path, display=True, save= True):

    imgr=cv2.imread(imgr_path)
    imgl=cv2.imread(imgl_path)
    M=np.array([[1,0,136.2942110803947],[0,1,0]])

    shifted=cv2.warpAffine(imgr,M,(imgr.shape[1],imgr.shape[0]))

    if display:
        cv2.namedWindow('Translated Image',cv2.WINDOW_NORMAL)
        cv2.namedWindow('Left Image',cv2.WINDOW_NORMAL)
        cv2.namedWindow('Right Image',cv2.WINDOW_NORMAL)
        # cv2.namedWindow('Undistorted_Image',cv2.WINDOW_NORMAL)
        cv2.imshow('Translated Image',shifted)
        cv2.imshow('Left Image',imgl)
        cv2.imshow('Right Image',imgr)
        # cv2.imshow("Undistorted_Image",undistort_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows() 

    if save:
        cv2.imwritimage,scalee('Shifted.jpeg',shifted)     

def draw_boxes(img, csv_file, display=False,save=True):

    # import ipdb;ipdb.set_trace()
    img=cv2.imread(img,1)
    with open(csv_file) as csvfile:
        spamreader=csv.reader(csvfile)
        for row in spamreader:
            cv2.rectangle(img,(int(row[0]),int(row[1])),(int(row[2]),int(row[3])),(255,255,0),4)
    
    if save:
        cv2.imwrite("Image_with_boxes.png",img)
    
    if display:
        cv2.namedWindow("Image",cv2.WINDOW_NORMAL)
        cv2.imshow("Image",img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def image_undistort(imgr_path, K,R,D,C, display=True, save= True):

    imgr=cv2.imread(imgr_path)

    w,h=len(imgr[0]),len(imgr)
   
    newcameramtx, roi=cv2.getOptimalNewCameraMatrix(K,D,(w,h),1,(w,h))
    undistort_image=cv2.undistort(imgr,K,D,None,newcameramtx)
    x,y,w_,h_=roi
    undistort_image=undistort_image[y:y+h_, x:x+w_]

    if display:
        cv2.namedWindow('Right Image',cv2.WINDOW_NORMAL)
        cv2.namedWindow('Undistorted_Image',cv2.WINDOW_NORMAL)

        cv2.imshow('Right Image',imgr)
        cv2.imshow("Undistorted_Image",undistort_image)

        cv2.waitKey(0)
        cv2.destroyAllWindows()
    if save:
        # import ipdb; ipdb.set_trace()
        cv2.imwrite('.'+imgr_path.split('.')[1]+'_undistorted.jpg',undistort_image)

# def find_disparity(imgl_path,imgr_path):


if __name__=='__main__':

    # images=os.listdir('./Dataset/image_rect_right')
    # for image in tqdm(images):
    #     match_hist('./cam1_image003893.png','./Dataset/image_rect_right/'+image,save=True)
    # find_disparity('./data_10_Jan/Images/Left Images/L0050.jpeg','./data_10_Jan/Images/Right Images/R0050.jpeg')
    # obtain_3d_volume('./Disparity/Output.png','./Disparity/L0058.jpeg' ,'./Disparity/L0058.csv')
    # disparity_image=crop_disparity_map('./Disparity/Output.png','./Disparity/L0058.csv')
    # clahe_('./L0058.jpeg')
    # make_video_from_frames('./deep_sort/Implementation 2/Tests/*.png')
    # draw_boxes('./data_10_Jan_latest/Images/Left Images/L0050.jpeg','./data_10_Jan_latest/Detections left/L0050.csv')
    # cropped_disparity_image=crop_disparity_map(cv2.imread('./data_temp/Disparity Maps/50.png',1),'./data_temp/Detections left/L0050.csv', map_type=0)
    # obtain_3d_volume('./DisparityImage_plt.png','./data_10_Jan/Images/Right Images/R0050.jpeg',fruit_location='./data_10_Jan/Detections left/L0050.csv' ,only_fruit=True)
    # file_path, csv_file,display=False,save=False
    # clean_point_cloud_points('./only_fruit_85.ply','./data_10_Jan/Disparity Maps/85.png')
    # find_disparity_RAFT(cv2.imread('./Disparity/L0058.jpeg',1),cv2.imread('./Disparity/R0058.jpeg',1))
    # image=cv2.imread('cam1_image003893.png',1)
    # image=cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
    # segmentation(image,scale=0.4,display=True)
    make_video_from_frames(path='./deep_sort/Tests/Baseline/*.png', Iframe=0, Fframe=36)
    Kr=np.array([[1052.350202570253, 0.0, 1031.808590719438],
                            [0.0, 1051.888280928595, 771.0661229952285],
                            [0.0, 0.0, 1.0]]) #Intrinsic parameter to convert camera frame to image frame)
    
    Rr= np.array([[0.9996990506423458, 0.002517310775589418, 0.02440228045187234],\
                    [-0.002557834410859948, 0.9999954009677927, 0.001629578592843086],\
                    [-0.02439806606924718, -0.001691505164855578, 0.9997008918583387]])
    
    Dr= np.array([-0.03098864107712216, 0.04051128735759788, -0.001361885214239114, -0.0008816601680637922, 0.0]) #Distortion matrix
    
    Cr=np.array([[0],[0],[0]])

    Pr= np.array([[1142.280571388607, 0.0, 988.5422821044922, -136.2942110803947], 
                [0.0, 1142.280571388607, 782.6398086547852, 0.0],
                [ 0.0, 0.0, 1.0, 0.0]])
    
    Kl= np.array([1052.382387969279, 0.0, 1058.58421357867, 
                0.0, 1052.123571352367, 800.4517901498787, 
                0.0, 0.0, 1.0]).reshape((3,3))

    Rl=np.array([0.9996673945529461, 0.001496854974222017, 0.02574606169709769, 
                -0.001454093723616705, 0.9999975323978124, -0.001679526638459951,
                -0.02574851217386264, 0.001641530832029927, 0.9996671043389194]).reshape(3,3)
    
    Pl=[1142.280571388607, 0.0, 988.5422821044922, 0.0, 
        0.0, 1142.280571388607, 782.6398086547852, 0.0, 
        0.0, 0.0, 1.0, 0.0]

    Dl=np.array([-0.02203560874783666, 0.02887488453134448, 0.000593652652803611, -0.00298638958591028, 0.0])
    # image_undistort('./Disparity/R0058.jpeg',K=Kr,R=Rr,D=Dr,C=Cr)
    # image_undistort('./Disparity/L0058.jpeg',K=Kl,R=Rl,D=Dl,C=Cr)

    # image_shift('./Disparity/R0058_undistorted.jpg','./Disparity/L0058_undistorted.jpg')