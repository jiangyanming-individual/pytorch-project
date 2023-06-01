# author:Lenovo
# datetime:2023/4/16 15:10
# software: PyCharm
# project:pytorch项目


import math
import torch
from torch import nn
from d2l import torch as d2l

class MultiHeadAttention(nn.Module):

    def __int__(self,key_size,query_size,value_size,num_hiddens,num_heads,dropout,bias=False,**kwargs):
        super(MultiHeadAttention, self).__int__()
        self.key_size=key_size
        self.query_size = query_size
        self.value_size = value_size
        self.num_hiddens=num_hiddens
        self.num_heads=num_heads
        self.dropout=dropout
        self.bias=bias


        self.attention=d2l.DotProductAttention(self.dropout)
        #全连接层：
        self.W_q=nn.Linear(self.query_size,num_hiddens,bias=self.bias)
        self.W_k = nn.Linear(self.key_size, num_hiddens,bias=self.bias)
        self.W_v = nn.Linear(self.value_size, num_hiddens,bias=self.bias)
        #最后的输出层：
        self.W_o = nn.Linear(self.num_hiddens, num_hiddens,bias=self.bias)


    def forward(self,queries,keys,values,valid_lens):
        # queries，keys，values的形状:
        # (batch_size，查询或者“键－值”对的个数，num_hiddens)
        queries=transpose_qkv(self.W_q(queries),self.num_heads)
        keys=transpose_qkv(self.W_k(keys),self.num_heads)
        values=transpose_qkv(self.W_v(values),self.num_heads)
        # 经过变换后，输出的queries，keys，values　的形状:
        # (batch_size*num_heads，查询或者“键－值”对的个数，
        # num_hiddens/num_heads)

        if valid_lens is not None:
            # 在轴0，将第一项（标量或者矢量）复制num_heads次，
            # 然后如此复制第二项，然后诸如此类。
            valid_lens=torch.repeat_interleave(
                valid_lens,repeats=self.num_heads,dim=0
            )
        # output的形状:(batch_size*num_heads，查询的个数，num_hiddens/num_heads)
        output=self.attention(queries,keys,values,valid_lens)
        print(output.shape)

        # 拼接：   output_concat的形状:(batch_size，查询的个数，num_hiddens)
        output_concat=transpose_output(output,self.num_heads)

        #最终的输出：
        return self.W_o(output_concat)


def transpose_qkv(X,num_heads):
    # 输入X的形状:(batch_size，查询或者“键－值”对的个数，num_hiddens)
    # 输出X的形状:(batch_size，查询或者“键－值”对的个数，num_heads，num_hiddens/num_heads)
    X=X.reshape(X.shape[0],X.shape[1],num_heads,-1)

    # 输出X的形状:(batch_size，num_heads，查询或者“键－值”对的个数,
    # num_hiddens/num_heads)
    X=X.permute(0,2,1,3)


    # 最终输出的形状:(batch_size*num_heads,查询或者“键－值”对的个数,
    # num_hiddens/num_heads)
    return X.reshape(-1,X.shape[2],X.shape[3])

def transpose_output(X,num_heads):

    """
    逆转transpose_qkv函数的操作
    :param X:
    :param num_heads:
    :return:
    """
    X=X.reshape(-1,num_heads,X.shape[1],X.shape[2])
    X=X.permute(0,2,1,3)

    return X.reshape(X.shape[0],X.shape[1],-1)


num_hiddens,num_heads=100,5
attention=MultiHeadAttention(num_hiddens,num_hiddens,num_hiddens,num_hiddens,num_heads,0.5)
attention.eval()


batch_size,num_queries=2,4
num_kvpairs, valid_lens =6, torch.tensor([3, 2])
X = torch.ones((batch_size, num_queries, num_hiddens))
Y = torch.ones((batch_size, num_kvpairs, num_hiddens))

res=attention(X,Y,Y,valid_lens).shape
print(res)