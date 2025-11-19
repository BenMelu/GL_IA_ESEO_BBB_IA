import os
import shutil as sh

PATH=os.path.dirname(os.path.realpath(__file__))
txt_path='data\\train.txt'
img_path='data\\images\\train\\'
img_dir='\\images'
with open(PATH+'\\'+txt_path,'r') as file:
    lines= file.readlines()
    lines=[line.rstrip('\n') for line in lines]

img_names=os.scandir(PATH+img_dir)
for img_name in img_names:
    for line in lines:
        temp=line.split("/")[3]
        if temp==img_name.name:
            print(temp + " : ok")
            sh.copyfile(img_name.path,PATH+"\\"+img_path+temp)

