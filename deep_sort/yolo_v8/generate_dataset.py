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
    images=glob.glob('./dataset/train/*.jpg')
    labels=glob.glob('./dataset/train/*.csv')

    for image,label in zip(images,labels):
        shutil.move(image,'./dataset/train/images')
        shutil.move(label,'./dataset/train/labels')

def csv2txt():
    csv_files=glob.glob('./datasets/test/labels/*.csv')

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
    csv_files=glob.glob('./datasets/train/labels/*.csv')

    for file in csv_files:
        os.remove(file)
    
csv2txt()
