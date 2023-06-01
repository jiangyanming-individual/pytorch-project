# author:Lenovo
# datetime:2023/3/2 21:44
# software: PyCharm
# project:pytorch项目

import numpy as np
import pandas as pd



#tartgetNumber是需要处理多少条数据
def openFile(filepath,targetNumber):

    with open(filepath, encoding='utf-8', mode='r') as f:
        # 计数器
        i = 0
        count_1 = 0
        count_0 = 0
        label = []
        sequence = []
        for line in f.readlines():
            i += 1
            # print(line.strip('\n'))

            if i % 2 != 0:
                str = line.strip('\n').split('|')
                # print("str:", str)
                if int(str[1]) == 1:
                    count_1 += 1
                else:
                    count_0 += 1
                label.append((str[1]))
                # print("label:",label)
            else:
                sequence.append(line.strip('\n'))
                # print("sequence：",sequence)
            if i == targetNumber:  # 正反样本各取一半：
                break
        print("i:", i)

        print("count_1:", count_1)
        print("count_0:", count_0)
        # print("total_label_list:"+"\t"+"length:",label,len(label))
        # print("total_sequence_list:,length:",sequence,len(sequence))
    #返回序列和label的列表；
    return sequence,label


#处理训练集
# train_sequence,train_label=openFile('./Kcr_CV.fa',49048)

#处理测试集
test_sequence,test_label=openFile('./Kcr_IND.fa',36706)

#对上面处理的序列进行保存成csv的文件：
def saveFile(filepath,sequence,label):
    with open(filepath, mode='r+', encoding='utf-8') as f:
        # 先读后写：
        for i in range(len(label)):
            # 先读取
            f.read()
            # 再写入到末尾；
            contend = sequence[i] + ',' + label[i]
            # 写入内容：
            f.write(contend)
            f.write('\n')
            f.flush()
            # f.write(contend2)
        # 关闭文件标签：
        f.close()
        print("write over!")
# saveFile('data/Kcr_cv.csv', train_sequence, train_label)
saveFile('data/Kcr_ind_2.csv', test_sequence, test_label)