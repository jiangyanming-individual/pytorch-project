# author:Lenovo
# datetime:2023/2/19 21:30
# software: PyCharm
# project:pytorch项目


import torch
from torch import nn
from d2l import torch as d2l

def pool2d(X,pool_size,mode='max'):

    p_h,p_w=pool_size
    Y=torch.zeros(X.shape[0]- p_h+1, X.shape[1]-p_w + 1)

    for i in range(Y.shape[0]):

        for j in range(Y.shape[1]):
            #平均池化层或者最大池化层：
            if mode == 'max':
                Y[i,j]=X[i:i+p_h,j:j+p_w].max()
            elif mode=='avg':
                Y[i,j]=X[i:i+p_h,j:j+p_w].mean()
    return Y

X = torch.tensor([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0], [6.0, 7.0, 8.0]])
print(pool2d(X,(2,2)))

print(pool2d(X,(2,2),mode='avg'))


X2=torch.arange(16,dtype=torch.float32).reshape((1,1,4,4))
print(X2)

pool2d_2=nn.MaxPool2d(3)

print(pool2d_2(X2))

pool2d_3=nn.MaxPool2d(3,padding=1,stride=2)
print(pool2d_3(X2))  # x.shape=4 * 4;   输出的大小是 4 / stride=2 =2

pool2d_4=nn.MaxPool2d(kernel_size=(2,3),padding=(0,1),stride=(2,3))
print(pool2d_4(X2))



#多通道：

X3=torch.cat((X2,X2+1),dim=1) #两个通道连接在一起，然后
print(X3)

pool2d_5=nn.MaxPool2d(3,padding=1,stride=2) # (3-1)/2=1 填充大小；
print(pool2d_5(X3))
print(pool2d_5(X3).shape)#torch.Size([1, 2, 2, 2])
# 输出也是大小为1，通道数为2个通道，输出的维度为2*2

