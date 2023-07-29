# author:Lenovo
# datetime:2023/7/10 10:05
# software: PyCharm
# project:pytorch项目


import os, re, sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

train_filepath = "./Datas/Kcr1_cv.csv"

with open(file=train_filepath, mode='r', encoding='utf-8') as f:
    all_sample = []
    for line in f.readlines():
        # print(line.strip().split(','))
        sequence, label = line.strip().split(',')
        sequence = re.sub(r'[^ACDEFGHIKLMNPQRSTVWY]', 'X', sequence)  # replace unknow
        #
        # print(sequence)
        # print(label)
        all_sample.append((sequence, label))
    f.close()

    # print(all_sample)

samples = all_sample



with open('./AAindex/AAindex_normalized.txt') as f:
    records = f.readlines()[1:]

AAindex = []
AAindexName = []
for i in records:
    AAindex.append(i.rstrip().split()[1:] if i.rstrip() != '' else None)
    AAindexName.append(i.rstrip().split()[0] if i.rstrip() != '' else None)

# 前29种物理化学性质：
props = 'FINA910104:LEVM760101:JACR890101:ZIMJ680104:RADA880108:JANJ780101:CHOC760102:NADH010102:KYTJ820101:NAKH900110:GUYH850101:EISD860102:HUTJ700103:OLSK800101:JURD980101:FAUJ830101:OOBM770101:GARJ730101:ROSM880102:RICJ880113:KIDA850101:KLEP840101:FASG760103:WILM950103:WOLS870103:COWR900101:KRIW790101:AURR980116:NAKH920108'.split(
    ':')
if props:
    tmpIndexNames = []
    tmpIndex = []
    for p in props:
        if AAindexName.index(p) != -1:
            tmpIndexNames.append(p)
            tmpIndex.append(AAindex[AAindexName.index(p)])
    if len(tmpIndexNames) != 0:
        AAindexName = tmpIndexNames
        AAindex = tmpIndex

# print(AAindex)
# print(len(AAindex))
# print(AAindexName)
# print(len(AAindexName))

AA_aaindex = 'ARNDCQEGHILKMFPSTWYV'
index = {}
for i in range(len(AA_aaindex)):
    index[AA_aaindex[i]] = i



blosum62 = {
            'A': [4, -1, -2, -2, 0, -1, -1, 0, -2, -1, -1, -1, -1, -2, -1, 1, 0, -3, -2, 0],  # A
            'R': [-1, 5, 0, -2, -3, 1, 0, -2, 0, -3, -2, 2, -1, -3, -2, -1, -1, -3, -2, -3],  # R
            'N': [-2, 0, 6, 1, -3, 0, 0, 0, 1, -3, -3, 0, -2, -3, -2, 1, 0, -4, -2, -3],  # N
            'D': [-2, -2, 1, 6, -3, 0, 2, -1, -1, -3, -4, -1, -3, -3, -1, 0, -1, -4, -3, -3],  # D
            'C': [0, -3, -3, -3, 9, -3, -4, -3, -3, -1, -1, -3, -1, -2, -3, -1, -1, -2, -2, -1],  # C
            'Q': [-1, 1, 0, 0, -3, 5, 2, -2, 0, -3, -2, 1, 0, -3, -1, 0, -1, -2, -1, -2],  # Q
            'E': [-1, 0, 0, 2, -4, 2, 5, -2, 0, -3, -3, 1, -2, -3, -1, 0, -1, -3, -2, -2],  # E
            'G': [0, -2, 0, -1, -3, -2, -2, 6, -2, -4, -4, -2, -3, -3, -2, 0, -2, -2, -3, -3],  # G
            'H': [-2, 0, 1, -1, -3, 0, 0, -2, 8, -3, -3, -1, -2, -1, -2, -1, -2, -2, 2, -3],  # H
            'I': [-1, -3, -3, -3, -1, -3, -3, -4, -3, 4, 2, -3, 1, 0, -3, -2, -1, -3, -1, 3],  # I
            'L': [-1, -2, -3, -4, -1, -2, -3, -4, -3, 2, 4, -2, 2, 0, -3, -2, -1, -2, -1, 1],  # L
            'K': [-1, 2, 0, -1, -3, 1, 1, -2, -1, -3, -2, 5, -1, -3, -1, 0, -1, -3, -2, -2],  # K
            'M': [-1, -1, -2, -3, -1, 0, -2, -3, -2, 1, 2, -1, 5, 0, -2, -1, -1, -1, -1, 1],  # M
            'F': [-2, -3, -3, -3, -2, -3, -3, -3, -1, 0, 0, -3, 0, 6, -4, -2, -2, 1, 3, -1],  # F
            'P': [-1, -2, -2, -1, -3, -1, -1, -2, -2, -3, -3, -1, -2, -4, 7, -1, -1, -4, -3, -2],  # P
            'S': [1, -1, 1, 0, -1, 0, 0, 0, -1, -2, -2, 0, -1, -2, -1, 4, 1, -3, -2, -2],  # S
            'T': [0, -1, 0, -1, -1, -1, -1, -2, -2, -1, -1, -1, -1, -2, -1, 1, 5, -2, -2, 0],  # T
            'W': [-3, -3, -4, -4, -2, -2, -3, -2, -2, -3, -2, -3, -1, 1, -4, -3, -2, 11, 2, -3],  # W
            'Y': [-2, -2, -2, -3, -2, -1, -2, -3, 2, -1, -1, -2, -1, 3, -3, -2, -2, 2, 7, -1],  # Y
            'V': [0, -3, -3, -3, -1, -2, -2, -3, -3, 3, 1, -2, 1, -1, -2, -2, 0, -3, -1, 4],  # V
            'X': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # -
        }

for key in blosum62:
    for j, value in enumerate(blosum62[key]):
        blosum62[key][j] = round((value + 4) / 15, 3)  # 四舍五入保留三位小数；



encoding_aaindex = []
for i in samples:
    sequence, label = i[0], i[1]
    code = [label]
    # print(code)
    for aa in sequence:
        if aa == 'X':
            for j in AAindex:
                code.append(0)
            continue
        for j in AAindex:
            # print("j:",j)
            # print(j[index[aa]])
            code.append(j[index[aa]])
    # print(len(code))
    encoding_aaindex.append(code)

    # print(encoding_aaindex)

AA = 'ARNDCQEGHILKMFPSTWYV'
encoding_binary = []
for i in samples:
    sequence, label = i[0], i[1]
    code = [label]
    for aa in sequence:
        if aa == 'X':
            code = code + [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
            code += [0.05] * 9
            continue
        for aa1 in AA:
            tag = 1 if aa == aa1 else 0
            code.append(tag)
        code += [0.05] * 9
    encoding_binary.append(code)


encoding_blosum = []
for i in samples:
    sequence, label = i[0], i[1]
    code = [label]
    for aa in sequence:
        code = code + blosum62[aa]
        code += [0.267] * 9  # 0.267是它的平均值；

    encoding_blosum.append(code)

encoding_aaindex = np.array(encoding_aaindex)
encoding_binary = np.array(encoding_binary)
encoding_blosum = np.array(encoding_blosum)

# print(encoding_aaindex[:, 1:])
# print(encoding_binary[:, 2:])

encodings = np.hstack((encoding_aaindex[:, 0:], encoding_binary[:, 1:], encoding_blosum[:, 1:]))
encodings.astype(np.float32)

# print(encodings)
# print(encodings.shape) #(batch_size, 2524)第一维度是label



class DealDataset(Dataset):
    def __init__(self, np_data):
        self.__np_data = np_data
        self.X = torch.from_numpy(self.__np_data[:, 1:])
        print(self.X.shape)
        self.y = torch.from_numpy(self.__np_data[:, 0]).view(-1, 1)
        print(self.y.shape)
        self.len = self.__np_data.shape[0]

    def __getitem__(self, index):
        return self.X[index], self.y[index]

    def __len__(self):
        return self.len


ind_set = DealDataset(encodings.astype(np.float32))
print("ind_set:",ind_set)

ind = np.zeros((len(ind_set), 2))
ind[:, 0] = encodings[:, 0].astype(np.float32)

ind_loader = DataLoader(ind_set, batch_size=64, shuffle=False)
print("ind_loader:",ind_loader)
