# author:Lenovo
# datetime:2023/3/6 20:55
# software: PyCharm
# project:pytorch项目

import numpy as np
"""
进行二进制编码：
"""
Amino_acid_sequence='ACDEFGHIKLMNPQRSTVWY'

def genereate_seq(filename):
    total_seq=[]
    with open(file=filename, mode='r', encoding='utf-8') as f:
        res_sequence = []
        # 二进制编码：
        i=0
        for line in f.readlines():
            # 字符串转为list
            # i+=1
            # print(i)
            line = list(line.strip('\n').split(','))
            sequence, label = line[0], line[-1]
            print(sequence, label)
            # 对一行序列进行编码：
            code = []
            # 对一个氨基酸进行编码：
            one_code = []
            #用于计数index后9位的氨基酸序列编码为[20 * 0+9位0.05]
            index=1
            for seq in sequence:
                """控制后9位的氨基酸编码"""
                if index >20:
                    zero_list = [0 for i in range(20)]
                    # print(zero_list)
                    # 前20位全部位0
                    one_code.extend(zero_list)
                    # 填充0.05
                    index+=1
                    one_code.extend([0.05] * 9)
                    continue # 直接跳出本次的循环
                """用于前20位的氨基酸编码"""

                for amino_acid_index in Amino_acid_sequence:
                    # 如果当前的氨基酸与遍历20种的氨基酸序对应上就进行赋值1的操作；反之赋值0
                    if amino_acid_index == seq:
                        flag = 1
                    else:
                        flag = 0
                    one_code.append(flag)
                one_code.extend([0.05] * 9)
                # print(one_code)
                # 一个序列的编码：29 * 29
                index+=1
            code.extend(one_code)
            res = np.array(code).reshape((1,29, 29))
            print(res.shape)

            res_sequence.append(res)
            res_sequence.append(np.array(int(label)))
            total_seq.append(res_sequence)
            #生成一个item的数据集
            yield total_seq
# res=genereate_seq('./data/Kcr_cv.csv')
res=genereate_seq('./data/Kcr_ind.csv')
# print(res)


for item in res:
    print(item)
    break
