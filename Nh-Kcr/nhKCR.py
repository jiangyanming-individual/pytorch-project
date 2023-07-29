#!/usr/bin/env python
# _*_ coding: utf-8 _*_

import os, re, sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

class DealDataset(Dataset):
    def __init__(self, np_data):
        self.__np_data = np_data
        self.X = torch.from_numpy(self.__np_data[:, 1:])
        self.y = torch.from_numpy(self.__np_data[:, 0]).view(-1, 1)
        self.len = self.__np_data.shape[0]

    def __getitem__(self, index):
        return self.X[index], self.y[index]

    def __len__(self):
        return self.len

class CNN_Feature2d(nn.Module):
    def __init__(self, device, out_channels, conv_kernel_size=5, pool_kernel_size=2, dense_size=64, dropout=0.5):
        super(CNN_Feature2d, self).__init__()
        self.device = device
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=out_channels, kernel_size=conv_kernel_size, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=pool_kernel_size)
        )
        self.dropout1 = nn.Dropout(dropout)
        self.conv2= nn.Sequential(
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=conv_kernel_size, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=pool_kernel_size)
        )
        self.dropout2 = nn.Dropout(dropout)
        self.conv3= nn.Sequential(
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=conv_kernel_size, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=pool_kernel_size)
        )
        self.dropout3 = nn.Dropout(dropout)
        self.conv4= nn.Sequential(
            nn.Conv2d(in_channels=out_channels, out_channels=2*out_channels, kernel_size=conv_kernel_size, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=pool_kernel_size)
        )
        self.dropout4 = nn.Dropout(dropout)
        self.fc1 = nn.Linear(9*out_channels, dense_size)
        self.fc2 = nn.Linear(dense_size, 1)
    
    def forward(self, X, is_train=False):
        out = X.view(-1, 3, 29, 29)        
        out = self.conv1(out)
        # print('Conv1: ', out.size())
        if is_train:
            out = self.dropout1(out)
        out = self.conv2(out)
        # print('Conv2: ', out.size())
        if is_train:
            out = self.dropout2(out)
        out = self.conv3(out)
        # print('Conv3: ', out.size())
        if is_train:
            out = self.dropout3(out)
        # out = self.conv4(out)
        # print('Conv4: ', out.size())
        # sys.exit(0)
        # if is_train:
        #     out = self.dropout4(out)
        # print(out.size())
        # sys.exit()
        out = out.view(out.size(0), -1)
        # print(out.size())
        # sys.exit()
        out = F.relu(self.fc1(out))        
        out = torch.sigmoid(self.fc2(out))
        return out
    
    def predict(self, loader):
        score = []
        for batch_X, batch_y in loader:            
            Xb = batch_X.to(self.device)
            yb = batch_y.to(self.device)
            y_ = self.forward(Xb)
            score.append(y_.cpu().data)
        return torch.cat(score).numpy()[:, 0]

# def pep1(file):
#     try:
#         AA = '-ARNDCQEGHILKMFPSTWYV'
#         if os.path.exists(file) == False:
#             return False, None
#
#         with open(file) as f:
#             record = f.read()
#
#         records = record.split('>')[1:]
#         # print(records)
#         myFasta = []
#         for fasta in records:
#             array = fasta.split('\n')
#             name, sequence = array[0].split()[0], ''.join(array[1:]).upper()
#             myFasta.append([name, sequence])
#
#         seqs = []
#         for fa in myFasta:
#             #替换非法字符：
#             name, sequence = fa[0], re.sub('[^ACDEFGHIKLMNPQRSTVWY]', '-', fa[1])
#             for i in range(len(sequence)):
#                 if sequence[i] == 'K':
#                     myStr = ''
#                     #组合成K位点左右15的序列；
#                     for j in range(i-14, i+15):
#                         if j in range(len(sequence)):
#                             myStr += sequence[j]
#                         else:
#                             myStr += '-'
#                     #处理数据集：
#                     seqs.append([name+"_"+str(j+1), myStr, 0, 'testing'])
#         return True, seqs
#     except Exception as e:
#         return False, None

def encoding(samples):
    try:
        with open('./AAindex/AAindex_normalized.txt') as f:
            records = f.readlines()[1:]
        AA_aaindex = 'ARNDCQEGHILKMFPSTWYV'
        AAindex = []
        AAindexName = []
        for i in records:
            AAindex.append(i.rstrip().split()[1:] if i.rstrip() != '' else None)
            AAindexName.append(i.rstrip().split()[0] if i.rstrip() != '' else None)

        #前29种物理化学性质：
        props = 'FINA910104:LEVM760101:JACR890101:ZIMJ680104:RADA880108:JANJ780101:CHOC760102:NADH010102:KYTJ820101:NAKH900110:GUYH850101:EISD860102:HUTJ700103:OLSK800101:JURD980101:FAUJ830101:OOBM770101:GARJ730101:ROSM880102:RICJ880113:KIDA850101:KLEP840101:FASG760103:WILM950103:WOLS870103:COWR900101:KRIW790101:AURR980116:NAKH920108'.split(':')
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


        index = {}
        for i in range(len(AA_aaindex)):
            index[AA_aaindex[i]] = i

        blosum62 = {
            'A': [4, -1, -2, -2, 0, -1, -1, 0, -2, -1, -1, -1, -1, -2, -1, 1, 0, -3, -2, 0, 0],  # A
            'R': [-1, 5, 0, -2, -3, 1, 0, -2, 0, -3, -2, 2, -1, -3, -2, -1, -1, -3, -2, -3, 0],  # R
            'N': [-2, 0, 6, 1, -3, 0, 0, 0, 1, -3, -3, 0, -2, -3, -2, 1, 0, -4, -2, -3, 0],  # N
            'D': [-2, -2, 1, 6, -3, 0, 2, -1, -1, -3, -4, -1, -3, -3, -1, 0, -1, -4, -3, -3, 0],  # D
            'C': [0, -3, -3, -3, 9, -3, -4, -3, -3, -1, -1, -3, -1, -2, -3, -1, -1, -2, -2, -1, 0],  # C
            'Q': [-1, 1, 0, 0, -3, 5, 2, -2, 0, -3, -2, 1, 0, -3, -1, 0, -1, -2, -1, -2, 0],  # Q
            'E': [-1, 0, 0, 2, -4, 2, 5, -2, 0, -3, -3, 1, -2, -3, -1, 0, -1, -3, -2, -2, 0],  # E
            'G': [0, -2, 0, -1, -3, -2, -2, 6, -2, -4, -4, -2, -3, -3, -2, 0, -2, -2, -3, -3, 0],  # G
            'H': [-2, 0, 1, -1, -3, 0, 0, -2, 8, -3, -3, -1, -2, -1, -2, -1, -2, -2, 2, -3, 0],  # H
            'I': [-1, -3, -3, -3, -1, -3, -3, -4, -3, 4, 2, -3, 1, 0, -3, -2, -1, -3, -1, 3, 0],  # I
            'L': [-1, -2, -3, -4, -1, -2, -3, -4, -3, 2, 4, -2, 2, 0, -3, -2, -1, -2, -1, 1, 0],  # L
            'K': [-1, 2, 0, -1, -3, 1, 1, -2, -1, -3, -2, 5, -1, -3, -1, 0, -1, -3, -2, -2, 0],  # K
            'M': [-1, -1, -2, -3, -1, 0, -2, -3, -2, 1, 2, -1, 5, 0, -2, -1, -1, -1, -1, 1, 0],  # M
            'F': [-2, -3, -3, -3, -2, -3, -3, -3, -1, 0, 0, -3, 0, 6, -4, -2, -2, 1, 3, -1, 0],  # F
            'P': [-1, -2, -2, -1, -3, -1, -1, -2, -2, -3, -3, -1, -2, -4, 7, -1, -1, -4, -3, -2, 0],  # P
            'S': [1, -1, 1, 0, -1, 0, 0, 0, -1, -2, -2, 0, -1, -2, -1, 4, 1, -3, -2, -2, 0],  # S
            'T': [0, -1, 0, -1, -1, -1, -1, -2, -2, -1, -1, -1, -1, -2, -1, 1, 5, -2, -2, 0, 0],  # T
            'W': [-3, -3, -4, -4, -2, -2, -3, -2, -2, -3, -2, -3, -1, 1, -4, -3, -2, 11, 2, -3, 0],  # W
            'Y': [-2, -2, -2, -3, -2, -1, -2, -3, 2, -1, -1, -2, -1, 3, -3, -2, -2, 2, 7, -1, 0],  # Y
            'V': [0, -3, -3, -3, -1, -2, -2, -3, -3, 3, 1, -2, 1, -1, -2, -2, 0, -3, -1, 4, 0],  # V
            'X': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # -
        }

        for key in blosum62:
            for j, value in enumerate(blosum62[key]):
                blosum62[key][j] = round((value+4)/15, 3) #四舍五入保留三位小数；

        encoding_aaindex = []
        for i in samples:
            name, sequence, label = i[0], i[1], i[2]
            code = [name, label]
            for aa in sequence:            
                if aa == '-':
                    for j in AAindex:
                        code.append(0)
                    continue
                for j in AAindex:                
                    code.append(j[index[aa]])
            encoding_aaindex.append(code)
        
        AA = 'ARNDCQEGHILKMFPSTWYV'
        encoding_binary = []
        for i in samples:
            name, sequence, label = i[0], i[1], i[2]
            code = [name, label]
            for aa in sequence:
                if aa == '-':
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
            name, sequence, label = i[0], i[1], i[2]
            code = [name, label]
            for aa in sequence:
                code = code + blosum62[aa]
                code += [0.267] * 9     #0.267是它的平均值；
            encoding_blosum.append(code)

        encoding_aaindex = np.array(encoding_aaindex)
        encoding_binary = np.array(encoding_binary)
        encoding_blosum = np.array(encoding_blosum)
        
        encodings = np.hstack((encoding_aaindex[:, 1:], encoding_binary[:, 2:], encoding_blosum[:, 2:]))
        return True, encodings.astype(np.float32)

    except Exception as e:
        return False, None

# def generateHtml(pd_result, file):
#     ftext = open('nhKCR_Pred.txt', 'w')
#     ftext.write('Samples\tPeptide\tScore\tPrediction\n')
#     for i in range(len(pd_result)):
#         ftext.write('%s\t%s\t%.4f\t' %(pd_result.iloc[i, 0], pd_result.iloc[i, 1], float(pd_result.iloc[i, 2])))
#         score = float(pd_result.iloc[i, 2])
#         if score >= 0.447:
#             ftext.write('Yes (high confidence)\n')
#         elif score >= 0.265:
#             ftext.write('Yes (medium confidence)\n')
#         elif score >= 0.146:
#             ftext.write('Yes (low confidence)\n')
#         else:
#             ftext.write('No\n')
#     ftext.close()


if __name__ == '__main__':    
    # workDir = './'
    # os.chdir(workDir)
    device = 'cpu'

    # error_msg = []
    # ok, samples = pep1(sys.argv[1])
    # ok, samples = pep1(input("LEVNNRIIEETLALKFENAAAGNKPEAVE"))
    pd_result = None
    ok=True

    if ok:
        samples=None
        ok_encoding, data_test = encoding(samples)

        if ok_encoding:            
            try:
                ind_set = DealDataset(data_test.astype(np.float32))

                ind = np.zeros((len(ind_set), 2))
                ind[:, 0] = data_test[:, 0].astype(np.float32)
                ind_loader = DataLoader(ind_set, batch_size=64, shuffle=False)

                model = CNN_Feature2d(device, 128, 5, 2, 64, 0.5)
                for i in range(5):
                    model.load_state_dict(torch.load('./models/Merged_%s.pkl' %(i+1), map_location=device))
                    ind[:, 1] += model.predict(ind_loader)
                ind[:, 1] /= 5
                
                tmp_sample = np.array(samples)            
                pd_result = pd.DataFrame(np.hstack((tmp_sample[:, 0:2], ind[:, 1].reshape((-1, 1)))), columns=['Samples', 'Peptide', 'Score'])

            except Exception as e:
                pd_result = None
            #
    #     else:
    #         error_msg.append('Feature encode error.')
    # else:
    #     error_msg.append('Prepare sample failed.')
    #
    # generateHtml(pd_result, 'result.php')