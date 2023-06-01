# author:Lenovo
# datetime:2023/2/12 20:45
# software: PyCharm
# project:pytorch项目


import torch
from torch import nn

def block1():
    return nn.Sequential(
        nn.Linear(4,8),
        nn.ReLU(),
        nn.Linear(8,4),
        nn.ReLU(),
    )

def block2():

    net=nn.Sequential()
    for i in range(4):
        net.add_module(f'block {i}:',block1())

    return net

resnet=nn.Sequential(block2(),nn.Linear(4,1))

#设置X的参数：
X=torch.rand(size=(2,4))

print(resnet(X))
print(resnet)

print(resnet[0][1][0].state_dict())
print(resnet[0][1][0].bias.data)



