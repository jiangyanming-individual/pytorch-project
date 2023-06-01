# author:Lenovo
# datetime:2023/3/27 10:32
# software: PyCharm
# project:pytorch项目


import torch
import torch.nn.functional as F
from torch.nn import LSTM

rnn=torch.nn.LSTM(16,32,2)
x=torch.randn((4,23,16))


prev_h=torch.randn((2,4,32))
prev_c = torch.randn((2, 4, 32))
y, h, c = rnn(x, prev_h, prev_c)

print(y.shape)
print(h.shape)
print(c.shape)

#[4,23,32]
#[2,4,32]
#[2,4,32]





