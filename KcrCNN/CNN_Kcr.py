# author:Lenovo
# datetime:2023/2/23 19:55
# software: PyCharm
# project:pytorch项目


import torch
from torch import nn
from torch.optim import SGD
import torch.nn.functional as F
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

#定义数据集迭代器：



#定义CNN网络模型


kcr_net=nn.Sequential(
    #输入通道是3
    nn.Conv2d(1,128,kernel_size=5,padding=2,stride=1),nn.ReLU(),
    nn.MaxPool2d(kernel_size=2,stride=1),#最大池化层；
    nn.Dropout(0.5),

    nn.Conv2d(128,128,kernel_size=5,padding=2,stride=1),nn.ReLU(),
    nn.MaxPool2d(kernel_size=2,stride=1),
    nn.Dropout(0.5),

    nn.Conv2d(128,128,kernel_size=5,padding=2,stride=1),nn.ReLU(),
    nn.MaxPool2d(kernel_size=2,stride=1),
    nn.Dropout(0.5),
    #全连接层；
    nn.Flatten(),
    nn.Linear(128 * 26 * 26,64),nn.ReLU(),#全连接神经元是 128 * 14 * 14

    nn.Linear(64,1),nn.Sigmoid(), #激活函数
)


X=torch.randn(size=(1,1,29,29))
# for layer in kcr_net:
#
#     X=layer(X)
#     print(layer.__class__.__name__,'out shape:\t',X.shape)

model=kcr_net(X)
print(model.detach().numpy()[0])


#定义5折交叉验证：


#模型参数准备：优化器：损失函数：




#定义训练函数：

def train(model):


    pass



#定义验证函数：


def test(model):


    pass










#评估模型

