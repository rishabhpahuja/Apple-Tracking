import glob
import shutil
import os
import  jpype     
import  asposecells     
jpype.startJVM() 
from asposecells.api import Workbook
from tqdm import tqdm
import csv

def move_images():
    # images=glob.glob('./dataset/test/*.jpg')
    labels=glob.glob('/home/pahuja/Projects/Apple tracking/20_fps_moving/processed/*.csv')

    for label in labels:
        # shutil.move(image,'./dataset/test/images')
        shutil.copy(label,'/home/pahuja/Projects/Apple tracking/deep_sort/yolo_v8/datasets/train/labels')

def csv2txt():
    csv_files=glob.glob('./datasets/train/labels/*.csv')

    for file in tqdm(csv_files):
        with open(file,'r') as f:
            reader= csv.reader(f)
            file_txt = open('.'+file.split('.')[-2]+'.txt', "w")
            for row in reader:
                file_txt.write(' '.join(row))
                file_txt.write('\n')
                # import ipdb; ipdb.set_trace()
                # output=(file.split('.')[-2]+'txt','r')
            os.remove(file)
            file_txt.close()

def delete_file():
    csv_files=glob.glob('./datasets/test/labels/*.csv')

    for file in csv_files:
        os.remove(file)
    
def yolov3_yolov8_dataset():

    h,w=(1536,2048)
    # file='left0001.txt'
    txt_files=glob.glob('./datasets/train/labels/*.txt')

    for file in tqdm(txt_files):
    
        with open(file,'r') as f:
            lines=f.readlines()
            os.remove(file)
            file_new=open(file,'w')
            
            for line in lines:

                row=line.split(' ')
                new_row=[(float(row[0])+float(row[2]))/(2*w),(float(row[1])+float(row[3].split('\n')[0]))/(2*h), \
                abs(float(row[2]))/w,abs(float(row[3].split('\n')[0]))/h]
                
                # import ipdb;ipdb.set_trace()
                file_new.write(' '.join(map(str,new_row)))
                file_new.write('\n')
                # print(' '.join(map(str,new_row)))

def add_class():

    labels=glob.glob('./datasets/train/labels/*.txt')

    for label in labels:

        file=open(label,'r')
        lines=file.readlines()
        os.remove(label)
        file=open(label,'w')

        for line in lines:
            file.write('0 '+line)
            # file.write('\n')
        
        file.close()
        

# move_images()
csv2txt()
yolov3_yolov8_dataset()
# add_class()