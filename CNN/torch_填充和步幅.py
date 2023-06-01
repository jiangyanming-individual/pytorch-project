# author:Lenovo
# datetime:2023/2/19 20:17
# software: PyCharm
# project:pytorch项目


import torch
from torch import nn
from d2l import torch as d2l

def comp_conv2d(conv2d,X):

    #shape 是属性不是函数；
    X=X.reshape((1,1) + X.shape) #(1,1)是大小和通道数；
    Y=conv2d(X)
    print(Y.shape) #torch.Size([1, 1, 8, 8])
    return Y.reshape(Y.shape[2:])

conv2d=nn.Conv2d(1,1,kernel_size=3,padding=1) #padding=1 说明行和列都添加了2行或者2列；
X=torch.rand((8,8)) #设置X的大小
print(comp_conv2d(conv2d,X).shape)

print("=====================================")
#重新设置padding的大小，使输入与输入的大小不变化；
conv2d2=nn.Conv2d(1,1,kernel_size=(5,3),padding=(2,1)) #ph=4，pw=2
print(comp_conv2d(conv2d2,X).shape)
























