# author:Lenovo
# datetime:2023/6/29 19:52
# software: PyCharm
# project:pytorch项目


import re
import numpy as np
import pandas as pd



train_filename="../data/training_data.txt"
indepentdent_test_data="../data/independent_test_data.txt"



def read_file(filename):

    with open(filename,mode='r',encoding='utf-8') as f:
        i=1
        label_list=[]
        seq_list=[]
        for line in f.readlines():
            if i % 2 !=0:
                label=line.strip().split('|')[0][-1]
                label_list.append(label)
                i+=1
            else:
                sequence=line.strip()
                i+=1
                seq_list.append(sequence)

        return seq_list,label_list
        print("处理数据完毕！")


def get_data(seq_list,lable_list,filename):

    with open(filename,mode='w+',encoding='utf-8') as f:

        for sequence,label in zip(seq_list,lable_list):
            print(sequence,label)
            f.write(sequence + ',' +label+"\n")

        print("转成csv文件完毕！")

if __name__ == '__main__':

    seq_train_list,label_train_list=read_file(train_filename)
    get_data(seq_train_list,label_train_list,'../data/train_data.csv')

    seq_test_list, label_test_list = read_file(indepentdent_test_data)
    get_data(seq_test_list, label_test_list,'../data/independent_test_data.csv')