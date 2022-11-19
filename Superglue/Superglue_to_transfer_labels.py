import argparse
import numpy as np
import numpy.random as random
import cv2
from super_matching import SuperMatching
import matplotlib.pyplot as plt
import imutils
import pandas as pd
import seaborn as sns

# from IPython import embed

def transf_matrix(theta=0, translation=[0,0]):
    assert len(translation) == 2
    tx, ty  = translation

    # First two columns correspond to the rotation b/t images
    M = np.zeros((2,3))
    M[:,0:2] = np.array([[np.cos(theta), np.sin(theta)],\
                         [ -np.sin(theta), np.cos(theta)]])

    # Last column corresponds to the translation b/t images
    M[0,2] = tx
    M[1,2] = ty
    
    return M

""" Convert the 2x3 rot/trans matrices to a 3x3 matrix """
def transf_mat_3x3(M):
    M_out = np.eye(3)
    M_out[0:2,0:3] = M
    return M_out

def transf_pntcld(M_est, pt_cld):
    '''
    M_est 2x3
    pt_cld nx2
    '''
    R = M_est[:,0:2]
    t = M_est[:,-1].reshape(2,-1)
    pt_cld_transf = (R@pt_cld.T + t).T 

    return pt_cld_transf


def setup_sg_class(superglue_weights_path):
    sg_matching = SuperMatching()
    sg_matching.weights = 'custom'
    sg_matching.weights_path = superglue_weights_path
    sg_matching.set_weights()
    
    return sg_matching

def find_conf_values(rect,matches,conf_score,pos,conf):

    k=0
    x0,y0,x1,y1=rect.loc[0],rect.loc[1],rect.loc[2],rect.loc[3]
    for i in range(len(matches)):
        x,y=matches[i]

        if(x>=x0 and x<=x1):
            if(y>y0 and y<=y1):
                k=k+1
                conf_score[pos]+=conf[i]
    if k!=0:
        conf_score[pos]=round(conf_score[pos]/k,3)
    return conf_score


def SuperGlueDetection(img1, img2, sg_matching,rect1=None ,rect2=None,debug=False):
    
    img1_gray=cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY)
    img2_gray=cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)

    image_mask1=np.zeros(img1_gray.shape,np.uint8)
    for i in range(len(rect1)):
        image_mask1[rect1.at[i,1]:rect1.at[i,3],rect1.at[i,0]:rect1.at[i,2]]=255
    img1_gray_masked=cv2.bitwise_and(img1_gray,image_mask1)

    image_mask2=np.zeros(img1_gray.shape,np.uint8)
    for i in range(len(rect2)):
        image_mask2[rect2.at[i,1]:rect2.at[i,3],rect2.at[i,0]:rect2.at[i,2]]=255
    img2_gray_masked=cv2.bitwise_and(img2_gray,image_mask1)

    mconf, kp1, kp2, matches1, matches2 = sg_matching.detectAndMatch(img1_gray_masked, img2_gray_masked,img1_gray,img2_gray)
    
    #! Show matched keypoints
    for x,y in kp1.astype(np.int64):
        cv2.circle(img1, (x,y), 2, (255,0,0), -1)
        cv2.circle(img2, (x,y), 2, (0,0,255), -1)
    # for x,y in kp2.astype(np.int64):

    
    # Show matches
    colours=dict()
    conf_score=[0 for i in range(len(matches1))]
    if debug:
        colour=np.array((sns.color_palette(None,len(rect1))))
        colour=list(np.asarray(colour*255,'uint8'))


        for i in range(len(rect1)):

            conf_score=find_conf_values(rect1.iloc[i], matches1,conf_score,i,mconf)
        
        for j,i in enumerate(range(len(rect1))):

            cv2.rectangle(img1,(rect1.loc[i,0],rect1.loc[i,1]),(rect1.loc[i,2],rect1.loc[i,3]),(int(colour[j][0]),int(colour[j][1]),int(colour[j][2])),3)        
            cv2.putText(img1,str(conf_score[j]),org=(rect1.loc[i,0],rect1.loc[i,1]-2),fontFace=cv2.FONT_HERSHEY_SIMPLEX,fontScale=0.5,color=(int(colour[j][0]),int(colour[j][1]),int(colour[j][2])),thickness=2)
            colours[j]=((rect1.loc[i,0],rect1.loc[i,1],rect1.loc[i,2],rect1.loc[i,3]),list(colour[j]))

        sg_matching.plot_matches(img1, img2, kp1, kp2, matches1, matches2,rect1, colours,mconf)
    
    return kp1, kp2, matches1, matches2


"""
Overlay the transformed noisy image with the original and estimate the affine
transformation between the two
# """
# def generateComposite(ref_keypoints, align_keypoints, ref_cloud, align_cloud,
#                       matches, rows, cols):
def get_warp_results(ref,align,M_est, debug=False):
    # Converts vector of keypoints to vector of points or the reverse, where each keypoint is assigned the same size and the same orientation.
    
    rows, cols = ref.shape

    M_est_inv = np.linalg.inv(transf_mat_3x3(M_est))[0:2,:]
    # from IPython import embed; embed()
    align_warped = cv2.warpAffine(align, M_est_inv, (cols, rows))

    alpha_img = np.copy(ref)
    alpha = 0.5
    composed_img = cv2.addWeighted(alpha_img, alpha, align_warped, 1-alpha, 0.0)
    if debug:
        displayImages(composed_img, 'Composite Image')
    return ref, align_warped, composed_img 

"""
Compute the translation/rotation pixel error between the estimated RANSAC
transformation and the true transformation done on the image.
"""
def computeError(M, M_est, M_est_inv):
    print('\nEstimated M\n', M_est)
    print('\nTrue M\n', M)

    # Add error
    error = M @ transf_mat_3x3(M_est_inv)
    R_del = error[0:2,0:2]
    t_del = error[0:2,2]

    print('\nTranslation Pixel Error: ', np.linalg.norm(t_del))
    print('Rotation Pixel Error: ', np.linalg.norm(R_del))
    
"""
Display a single image or display two images conatenated together for comparison
Specifying a path will save whichever image is displayed (the single or the
composite).
"""
def displayImages(img1, name1='Image 1', img2=None, name2='Image2', path=None):
    if img2 is None:
        # ASSERT: Display only 1 image
        output = img1
        cv2.namedWindow(name1, cv2.WINDOW_NORMAL)
        cv2.imshow(name1, img1)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        # Display both images concatenated
        output = np.concatenate((img1, img2), axis=1)
        cv2.namedWindow(name1 + ' and ' + name2, cv2.WINDOW_NORMAL)
        cv2.imshow(name1 + ' and ' + name2, output)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    if path is None:
        # Save the image at the current path
        print("")
    else:
        cv2.imwrite(path, output)

"""
Test feature detection, feature matching, and pose estimation on an image.
"""
def cv_kp_to_np(cv_keypoints):
    list_kp_np = []
    for idx in range(0, len(cv_keypoints)):
        list_kp_np.append(cv_keypoints[idx].pt)
    
    return np.array(list_kp_np).astype(np.int64)        
    # ref_cloud = np.float([cv_keypoints[idx].pt for idx in range(0, len(cv_keypoints))]).reshape(-1, 1, 2)


def find_transformation_SuperGlue(ref, align, sg_matching, debug=False):

    ref_keypoints, align_keypoints, matches1, matches2 = SuperGlueDetection(ref, align, sg_matching, debug)
    
    
    try :
        M_est = cv2.estimateAffinePartial2D(matches1, matches2)[0]
        
  
    except:
        print("could not find matches")
        M_est = np.array([[1,0,0],
                          [0,1,0]])

    return M_est, ref_keypoints, align_keypoints, matches1, matches2
 

def put_at_center(fg_img, bg_img):
    h, w = fg_img.shape
    hh, ww = bg_img.shape

    yoff = round((hh-h)/2)
    xoff = round((ww-w)/2)

    result = bg_img.copy()
    result[yoff:yoff+h, xoff:xoff+w] = fg_img

    return result


def resize_images(ref,align):
    h1,w1 = ref.shape[:2]
    h2,w2 = align.shape[:2]

    h = max([h1,w1,h2,w2])

    bg_img = np.zeros((h,h), dtype=ref.dtype)
    ref_padded = put_at_center(ref, bg_img)
    align_padded = put_at_center(align, bg_img)

    return ref_padded, align_padded


def adjust_contrast(img):


# converting to LAB color space
    lab= cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l_channel, a, b = cv2.split(lab)

    # Applying CLAHE to L-channel
    # feel free to try different values for the limit and grid size:
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    cl = clahe.apply(l_channel)

    # merge the CLAHE enhanced L-channel with the a and b channel
    limg = cv2.merge((cl,a,b))

    # Converting image from LAB Color model to BGR color spcae
    enhanced_img = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)

    return enhanced_img

def test_two(args):
    # from IPython import embed; embed()
    # Load images
    ref_path = args.reference_path
    align_path = args.align_path

    ref = cv2.imread(ref_path, 1) 
    # ref=adjust_contrast(ref) 
    align = cv2.imread(align_path, 1)
    # align=adjust_contrast(align)
    rect1=pd.read_csv('../detections2/L0085.csv', header=None)
    rect2=pd.read_csv('../detections2/L0086.csv', header=None)

    sg_matching = setup_sg_class(args.superglue_weights_path)
    SuperGlueDetection(ref, align, sg_matching,rect1,rect2,debug=True) #rect1 and rect2 include rectagles of detected fruit



# Main Code
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Image pair pose estimation.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

        

    parser.add_argument('-ref', '--reference_path',
                        type=str, default='./L0085.jpeg',
                        help='Reference Image')
    parser.add_argument('-align', '--align_path',
                        type=str, default='./L0086.jpeg',
                        help='Image to align')
    
    parser.add_argument('-weights', '--superglue_weights_path', default='./models/weights/global_registration_sg.pth',
                        help='SuperGlue weights path')

    
    args = parser.parse_args()
    test_two(args)

    
    