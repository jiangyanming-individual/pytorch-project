# author:Lenovo
# datetime:2023/2/7 21:25
# software: PyCharm
# project:pytorch项目
import torch
from torch.utils import data
import random
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

#生成数据集：
def generate_data(w,b,num_examples):

    #f(x)=X * w +b
    X=torch.normal(0,1,(num_examples,len(w)))
    print(X.shape)
    y=torch.matmul(X,w) +b
    print(y.shape)
    #加入噪音：
    y+=torch.normal(0,0.01,y.shape)
    return X,y.reshape((-1,1))  #返回一列的label的值；

#真实的数据 w,b ;
true_w=torch.tensor([2,-3.4])
true_b=4.2

features,labels=generate_data(true_w,true_b,1000)


#读取数据：

"""data_arrays :传入的data 和 target,batch_size:是每次传入多大的数据
is_train是否打乱数据"""
def load_array(data_arrays, batch_size, is_train=True):
    """构造一个PyTorch数据迭代器"""
    dataset = data.TensorDataset(*data_arrays) #传入一个元组的参数
    return data.DataLoader(dataset, batch_size, shuffle=is_train)

batch_size=10

data_iter=load_array((features,labels),batch_size,is_train=True)
# print(type(data_iter))
#iter()函数实际上就是调⽤了可迭代对象的 iter ⽅法。
# print(next(iter(data_iter))) #next 函数是调用下一个要遍历的对象；

#定义模型函数：

from torch import nn

Liner_net=nn.Sequential(
    nn.Linear(2,1), #线性模型加入；
)

#初始化 w ,b参数
"""Liner_net[0]是第一层"""
Liner_net[0].weight.data.normal_(0,0.01)
Liner_net[0].bias.data.fill_(0)

#定义损失函数：
"""
loss(x,y)=1/n∑(xi−yi)2
"""
loss=nn.MSELoss() #计算n个数据的均方误差求和；

#定义优化算法：
sgd=torch.optim.SGD(Liner_net.parameters(),lr=0.03) #SGD梯度下降；
#训练：

num_epoch=3
for epoch in range(num_epoch):

    #得到最佳的 w,b
    for X,y in data_iter:
        print(X,'\n',y)
        l=loss(Liner_net(X),y)
        sgd.zero_grad() #梯度清零：
        l.backward()
        sgd.step() #执行SGD;更新 w,b;

    #训练数据
    l=loss(Liner_net(features),labels)
    print(f"epoch:{epoch + 1},loss :{l:f}")


