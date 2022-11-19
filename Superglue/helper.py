import cv2
import SimpleITK as sitk
import numpy as np



def display_with_cv2(simg):
    img = sitk.GetArrayFromImage(simg)
    cv2.imshow('img',img);cv2.waitKey(0)
    cv2.destroyAllWindows()

   
def get_img_diff(img1,img2):
    img_diff_float = abs(img1.astype(np.float32) - img2.astype(np.float32))
    return img_diff_float.astype(np.uint8)
