# author:Lenovo
# datetime:2023/2/19 16:16
# software: PyCharm
# project:pytorch项目

import torch
from torch import nn
from d2l import torch as d2l



#互相关运算
def corr2d(X,K):

    h,w=K.shape
    Y=torch.zeros(X.shape[0] -h +1,X.shape[1] -w +1)

    for i in  range(Y.shape[0]):

        for j in range(Y.shape[1]):
            Y[i,j]=(X[i:i+h,j:j+w] * K).sum()
    return Y

X = torch.tensor([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0], [6.0, 7.0, 8.0]])
K = torch.tensor([[0.0, 1.0], [2.0, 3.0]])


# corr2d(X,K)
# print(corr2d(X,K))


#定义Conv2D层；
class Conv2D(nn.Module):

    #kenal_size是卷积和的大小；也是weight
    def __init__(self,kenal_size):
        # nn.Parameter绑定模型，使参数可以进行更新的操作
        self.weight=nn.Parameter(torch.rand(kenal_size))
        self.bais=nn.Parameter(torch.zeros(1))
    def forward(self):
        return corr2d(X,self.weight)+self.bais


X2=torch.ones((6,8))
X2[:,2:6] = 0
print(X2)

K2=torch.tensor([[1.0,-1.0]])

Y2=corr2d(X2,K2)
print(corr2d(X2,K2))
print(corr2d(X2.t(),K2))



#定义卷积和：
conv2d = nn.Conv2d(1,1, kernel_size=(1, 2), bias=False)

X2=X2.reshape((1,1,6,8))
Y2=Y2.reshape((1,1,6,7))

lr = 3e-2  # 学习率


for i in range(10):

    Y_hat=conv2d(X2)
    loss=(Y_hat - Y2) ** 2
    conv2d.zero_grad()
    loss.sum().backward() #反向传播；

    #更新weight：
    conv2d.weight.data[:]-= lr * conv2d.weight.grad

    if i % 2==0:
        print(f'epoch:{i+1},loss:{loss.sum():.3f}')


#打印conv2d的weight参数：
print(conv2d.weight.data.reshape((1,2)))

















