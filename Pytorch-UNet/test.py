import cv2
import numpy as np

img=cv2.imread('./data/Masks/AAA_4390.png',1)
cv2.namedWindow('Image',cv2.WINDOW_NORMAL)
cv2.imshow('Image',img)
cv2.waitKey(0)
cv2.destroyAllWindows()