import cv2
import numpy as np

# import ipdb;ipdb.set_trace()
img=cv2.imread('./AAA_4420.png',1)
mask=cv2.imread('./mask.png',1)
added=cv2.addWeighted(img,1,mask,0.5,0)
cv2.namedWindow('Image',cv2.WINDOW_NORMAL)
cv2.imshow('Image',added)
cv2.waitKey(0)
cv2.destroyAllWindows()