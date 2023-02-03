import cv2
import numpy as np
import csv

mask=cv2.imread('L0050.jpeg',0)
image=cv2.imread('L0050_.jpeg')
def polsby_popper():
    
    with open('L0050.csv','r') as file:
        bboxs= csv.reader(file)
        for bbox in bboxs:
            x1,y1,x2,y2=(int(bbox[0]),int(bbox[1]),int(bbox[2]),int(bbox[3]))
            fruit=mask[y1:y2,x1:x2]
            cont,_=cv2.findContours(fruit, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if len(cont)>1:
                max=0
                for contour in cont:
                    k=cv2.contourArea(contour) 
                    if max<k:
                        max=cv2.contourArea(contour)
                        cont=contour
            try:
                score=(4*cv2.contourArea(cont))/(cv2.arcLength(cont,False))**2
            except:
                cv2.drawContours(image, cont,-1,(255,0,0),2)
                cv2.namedWindow('image',cv2.WINDOW_NORMAL)
                cv2.namedWindow('fruit',cv2.WINDOW_NORMAL)
                cv2.imshow("image",image)
                cv2.imshow("fruit",fruit)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
            cv2.putText(image,'Score:'+str(round(score,3)),color=(255,255,255),org=(x1,y1),fontScale=0.9,fontFace=cv2.FONT_HERSHEY_DUPLEX)
            cv2.rectangle(image, (x1,y1), (x2,y2), (255,0,0), 2)
            # print(score)

    cv2.imwrite('Polsby_Popper_test.jpg',image)
    cv2.namedWindow('image',cv2.WINDOW_NORMAL)
    # https://en.wikipedia.org/wiki/Polsby%E2%80%93Popper_test
    cv2.imshow("image",image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def pixel_score():

    with open('L0050.csv','r') as file:
        bboxs= csv.reader(file)
        for bbox in bboxs:
            x1,y1,x2,y2=(int(bbox[0]),int(bbox[1]),int(bbox[2]),int(bbox[3]))
            fruit=mask[y1:y2,x1:x2]
            cont,_=cv2.findContours(fruit, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            # import ipdb;ipdb.set_trace()
            # import ipdb; ipdb.set_trace()
            if len(cont)>1:
                max=0
                for contour in cont:
                    k=cv2.contourArea(contour) 
                    if max<k:
                        max=cv2.contourArea(contour)
                        cont=contour
            else:
                cont=cont[0]
            # import ipdb; ipdb.set_trace()
            score=cv2.contourArea(cont)/(fruit.shape[0]*fruit.shape[1])
            cv2.putText(image,'Score:'+str(round(score,3)),color=(255,255,255),org=(x1,y1),fontScale=0.9,fontFace=cv2.FONT_HERSHEY_DUPLEX)
            cv2.rectangle(image, (x1,y1), (x2,y2), (255,0,0), 2)
    # cv2.namedWindow('image',cv2.WINDOW_NORMAL)
    # # https://en.wikipedia.org/wiki/Polsby%E2%80%93Popper_test
    # cv2.imshow("image",image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    cv2.imwrite('Area_wise_score.png',image)
pixel_score()