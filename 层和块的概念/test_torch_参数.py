# author:Lenovo
# datetime:2023/2/12 20:13
# software: PyCharm
# project:pytorch项目


import torch
from torch import nn
from torch.nn import functional as F
net=nn.Sequential(
    nn.Linear(4,8),
    nn.ReLU(),

    nn.Linear(8,1)
)
X=torch.rand(size=(2,4))
print(net(X))

#得到第二层的参数：
print(net[2].state_dict())
print(net[2].bias)
print(net[2].bias.data)

#获得所有的参数:
print(*[(name,param.shape) for name,param in net.named_parameters()])
#获取一层的参数:
print(*[(name,param.shape) for name,param in net[0].named_parameters()])
