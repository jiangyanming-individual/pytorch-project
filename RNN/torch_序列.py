# author:Lenovo
# datetime:2023/2/25 19:21
# software: PyCharm
# project:pytorch项目

import torch
from torch import nn
from d2l import torch as d2l



# d2l.load_array()
T=1000
time=torch.arange(1,T+1,dtype=torch.float32)
X=torch.sin(0.01 * time) + torch.normal(0,0.2,(T,)) #正态分布；

d2l.plot(time,[X],legend=['time','X'],xlim=[1,1000],figsize=(6,3))
