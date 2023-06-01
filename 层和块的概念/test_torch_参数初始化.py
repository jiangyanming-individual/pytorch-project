# author:Lenovo
# datetime:2023/2/12 20:54
# software: PyCharm
# project:pytorch项目


import torch
from torch import nn
import warnings
warnings.filterwarnings("ignore")

X=torch.rand(size=(2,4))

net=nn.Sequential(
    nn.Linear(4,8),
    nn.ReLU(),

    nn.Linear(8,1),
    nn.ReLU()
)

def init_constant(m):
    if type(m) == nn.Linear:
        nn.init.constant_(m.weight, 1)
        nn.init.zeros_(m.bias)
net.apply(init_constant)

print("weight:",net[0].weight.data[0], "bias:",net[0].bias.data[0])


def init_xavier(m):
    if type(m)== nn.Linear:
        nn.init.xavier_uniform(m.weight)

def init_42(m): #常量值weight进行初始化：
    if type(m) == nn.Linear:
        nn.init.constant_(m.weight,42)
net[0].apply(init_xavier)
net[2].apply(init_42)

print(net[0].weight.data[0])
print(net[0].weight.data)
print(net[2].weight.data)



""""共享层的参数都是一致的，一个改变所有的都改变"""

shared=nn.Linear(8,8)

net=nn.Sequential(
    nn.Linear(4,8),
    nn.ReLU(),
    shared,
    nn.ReLU(),
    shared,
    nn.Linear(8,1)
)
#判断weight里面的值是不是相等，如果相等的情况下，直接判断true或者false；
print(net[2].weight.data[0] == net[4].weight.data[0])
#改变第四层的参数；
net[4].weight.data[0,0]=100
print(net[2].weight.data[0]== net[4].weight.data[0])

