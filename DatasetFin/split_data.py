import random as rd
from os import path

PATH=path.dirname(path.realpath(__file__))+"\\data\\"
file_path='old_train.txt'
with open(PATH+file_path,'r') as file:
    lines= file.readlines()

    for line in lines:
        line=str('/'.join(line.split("/")[3:]))
        y=rd.randint(1,100)
        if y>79:
            with open(PATH+'test.txt','a') as test:
                test.write(line)
        elif y>58:
            with open(PATH+'val.txt','a') as val:
                val.write(line)
        else:
            with open(PATH+'train.txt','a') as train:
                train.write(line)