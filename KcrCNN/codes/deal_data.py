
"""
处理.fa数据：
"""

import re
import numpy as np
import pandas as pd



train_filename="../data/Kcr_CV.fa"
indepentdent_test_data="../data/Kcr_IND.fa"



def read_file(filename):

    with open(filename,mode='r',encoding='utf-8') as f:
        i=1
        label_list=[]
        seq_list=[]
        for line in f.readlines():
            if i % 2 !=0:
                label=line.strip().split('|')[1]
                # print(label)
                label_list.append(label)
                i+=1
            else:
                sequence=line.strip()
                i+=1
                seq_list.append(sequence)
        f.close()
        return seq_list,label_list
        print("处理数据完毕！")


def get_data(seq_list,lable_list,filename):

    with open(filename,mode='w+',encoding='utf-8') as f:

        for sequence,label in zip(seq_list,lable_list):
            print(sequence,label)
            #写入文件
            f.write(sequence + ',' +label+"\n")
        f.close()
        print("转成csv文件完毕！")

if __name__ == '__main__':

    seq_train_list,label_train_list=read_file(train_filename)
    # print(seq_train_list)
    # print(label_train_list)
    get_data(seq_train_list,label_train_list,'../data/Kcr_train.csv')

    seq_test_list, label_test_list = read_file(indepentdent_test_data)
    get_data(seq_test_list, label_test_list,'../data/Kcr_Ind_test.csv')