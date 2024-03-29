import cv2
import csv
import numpy as np

def find_contour_center(mask_cropped,image_,row, display=True):

    '''
    mask_cropped: The mask cropped along bounding box
    image_: The captire image
    row: The coordinate bounding box
    '''

    contours, _ = cv2.findContours(mask_cropped, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if display:
        cv2.imshow('Cropped Image',mask_cropped)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    if len(contours)==1:
        M=cv2.moments(contours[0])        
        try:
            cx=int(M['m10']/M['m00'])+int(row[0])
            cy=int(M['m01']/M['m00'])+int(row[1])
            cv2.circle(image_, (cx, cy), 10, (0, 0, 255), -1)
        except:
            cv2.namedWindow('cropped',cv2.WINDOW_NORMAL)
            cv2.drawContours(cv2.cvtColor(mask_cropped,cv2.COLOR_GRAY2BGR),contours,-1,(255,255,0),10)
            cv2.imshow('cropped', mask_cropped)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        return image_ 

    if len(contours)>1:
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
    return image_

#Find bounding boxes
def find_center(csv_file,image,mask):
    with open(csv_file) as csvfile:
        spamreader=csv.reader(csvfile)
        for i,row in enumerate(spamreader):
            image=find_contour_center(mask[int(row[1]):int(row[3]),int(row[0]):int(row[2])],image,row,display=False)
            cv2.rectangle(image,(int(row[0]),int(row[1])),(int(row[2]),int(row[3])),(255,255,0),4)



mask=cv2.imread('../data_10_Jan_latest/Segmentation masks/L0050.jpeg',0)
image=cv2.imread('../data_10_Jan_latest/Images/Left Images/L0050.jpeg')
csv_file='../data_10_Jan_latest/Detections left/L0050.csv'
mask=np.where(mask>200,255,0).astype(np.uint8)

image=find_center(csv_file,image,mask)

added=cv2.addWeighted(cv2.cvtColor(mask,cv2.COLOR_GRAY2BGR),0.8,image,1,0)

#Displaying windows
cv2.namedWindow('Mask',cv2.WINDOW_NORMAL)
cv2.namedWindow('Added',cv2.WINDOW_NORMAL)
cv2.namedWindow('Image',cv2.WINDOW_NORMAL)

cv2.imshow('Added',added)
cv2.imshow('Image',image)
cv2.imshow('Mask',mask)
cv2.waitKey(0)
cv2.destroyAllWindows()

#saving images
cv2.imwrite('mask+box+center.jpeg',added)
cv2.imwrite('center.jpeg',image)
# import ipdb; ipdb.set_trace()




