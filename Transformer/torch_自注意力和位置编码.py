# author:Lenovo
# datetime:2023/4/15 18:00
# software: PyCharm
# project:pytorch项目


import math
import torch
from torch import nn
from d2l import torch as d2l


num_hiddens,num_heads=100,5

attention=d2l.MultiHeadAttention(num_hiddens,num_hiddens,num_hiddens,num_hiddens,num_heads,0.5)
batch_size,num_queries,valid_lens=2,4,torch.tensor([3,2])
X=torch.ones((batch_size,num_queries,num_hiddens))
res=attention(X,X,X,valid_lens).shape
print(res)

"""
位置编码：
"""

class PositionalEncoding(nn.Module):

    def __init__(self,num_hiddens,dropout,max_len=1000):
        super(PositionalEncoding, self).__init__()

        self.dropout=dropout
        self.P=torch.zeros((1,max_len,num_hiddens))
        X=torch.arange(max_len,dtype=torch.float32).reshape(-1,1)/torch.pow(10000,torch.arange(0,num_hiddens,2,dtype=torch.float32) /num_hiddens)

        self.P[:,:,0::2] =torch.sin(X)
        self.P[:,:,1::2]=torch.cos(X)


    def forward(self,X):

        X=X + self.P[:,:X.shape[1],:]
        return self.dropout(X)



