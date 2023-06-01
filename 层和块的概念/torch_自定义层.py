# author:Lenovo
# datetime:2023/2/15 15:13
# software: PyCharm
# project:pytorch项目


import torch
from torch import nn
from d2l import torch as d2l
import torch.nn.functional as F


class MyLinaer(nn.Module):

    def __init__(self,input_units,out_units):
        super(MyLinaer, self).__init__()
        self.weight=nn.Parameter(torch.randn(input_units,out_units))
        self.bais=nn.Parameter(torch.randn(out_units,))

    def forward(self,X):

        linear=torch.matmul(X,self.weight.data) + self.bais.data
        return F.relu(linear)

linear=MyLinaer(5,3)
# linear
print(linear.weight)
#进行forward()前向传播运算；
print(linear(torch.rand(2,5)))  #（2，5） * （5，3）


