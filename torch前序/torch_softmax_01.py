# author:Lenovo
# datetime:2023/2/11 22:07
# software: PyCharm
# project:pytorch项目


import torch
from torch import nn
from d2l import torch as d2l

batch_size=256
train_data,test_data=d2l.load_data_fashion_mnist(batch_size)


#初始化模型参数；
net=nn.Sequential(
    nn.Flatten(),
    nn.Linear(784,10)
)


#定义参数；
def init_weights(m):
    if type(m) ==nn.Linear:
        nn.init.normal(m.weight,std=0.01)
net.apply(init_weights)

#定义损失函数

loss=nn.CrossEntropyLoss(reduction='none')


#定义梯度优化：
trainer=torch.optim.SGD(net.parameters(),lr=0.1)

#训练：
num_epoch=10
d2l.train_ch3(net,train_data,test_data,loss,num_epoch,trainer)













