# author:Lenovo
# datetime:2023/3/7 12:46
# software: PyCharm
# project:pytorch项目


import numpy as np
from torch.utils.data import Dataset,DataLoader



Amino_acid_sequence='ACDEFGHIKLMNPQRSTVWY'
class MyDataset(Dataset):

    def __init__(self, filepath):
        """
        步骤二：实现 __init__ 函数，初始化数据集，将样本和标签映射到列表中
        """
        super(MyDataset, self).__init__()
        self.data_list = []
        with open(filepath,encoding='utf-8') as f:
            for line in f.readlines():
                x_data,label=list(line.strip('\n').split(','))
                self.data_list.append([x_data, label])

    def __getitem__(self, index):
        """
        步骤三：实现 __getitem__ 函数，定义指定 index 时如何获取数据，并返回单条数据（样本数据、对应的标签）
        """
        # 根据索引，从列表中取出一个图像
        x_data_sequence, label = self.data_list[index]
        # 飞桨训练时内部数据格式默认为float32，将图像数据格式转换为 float32
        # 对一行序列进行编码：
        code = []
        # 对一个氨基酸进行编码：
        one_code = []
        for seq in x_data_sequence:

            if seq == '_':
                zero_list = [0 for i in range(20)]
                # print(zero_list)
                # 前20位全部位0
                one_code.extend(zero_list)
                # 填充0.05
                one_code.extend([0.05] * 9)
            for amino_acid_index in Amino_acid_sequence:
                # 如果当前的氨基酸与遍历20种的氨基酸序对应上就进行赋值1的操作；反之赋值0
                if amino_acid_index == seq:
                    flag = 1
                else:
                    flag = 0
                one_code.append(flag)
            #一个氨基酸编码：
            one_code.extend([0.05] * 9)
            # print(one_code)

        # 一个序列的编码：29 * 29
        code.extend(one_code)
        x_data_sequence = np.array(code).reshape((29, 29))
        print(x_data_sequence.shape)

        #label标签：
        label = int(label)
        # 返回图像和对应标签
        return x_data_sequence, label

    def __len__(self):
        """
        步骤四：实现 __len__ 函数，返回数据集的样本总数
        """
        return len(self.data_list)

train_dataset=MyDataset('./data/Kcr_cv.csv')


for data in train_dataset:
    print(data[0],data[1])
