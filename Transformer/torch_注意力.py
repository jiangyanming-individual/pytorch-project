# author:Lenovo
# datetime:2023/4/15 18:00
# software: PyCharm
# project:pytorch项目


import math
import torch
from torch import nn
from d2l import torch as d2l
import numpy as np

x=torch.tensor([10.0,20.0,30.0])


#掩码信息： 掩码softmax的操作：
def masked_softmax(X,valid_lens):

    #valid_lens 有效词元长度：
    if valid_lens is None:
        return nn.functional.softmax(X,dim=-1) #dim=-1 和 dim =2 都是一行进行softmax

    else:
        shape=X.shape
        if valid_lens.dim() == 1:
            valid_lens=torch.repeat_interleave(valid_lens,shape[1])
        else:
            valid_lens=valid_lens.reshape(-1)
        X=d2l.sequence_mask(X.reshape(-1,shape[-1]),valid_lens,value=-1e6)

        return nn.functional.softmax(X.reshape(shape),dim=-1)


print(masked_softmax(torch.rand(2, 2, 4), torch.tensor([2, 3])))

#加注意力：

#valid_lens :有效长度
class Addattention(nn.Module):

    def __init__(self,key_size,query_size,num_hiddens,dropout,**kwargs):
        super(Addattention, self).__init__()

        self.W_k=nn.Linear(key_size,num_hiddens,bias=False)
        self.W_q=nn.Linear(query_size,num_hiddens,bias=False)
        self.W_v=nn.Linear(num_hiddens,1,bias=False) # kq *v

        self.dropout=nn.Dropout(dropout) #正则化；

    def forward(self,queries,keys,values,valid_lens):
        """
        将输入 * W 权重得到queries keys;
        :param queries:
        :param keys:
        :param values:
        :param valid_lens:
        :return:
        """

        queries,keys=self.W_q(queries),self.W_k(keys)

        # print(queries.shape) # 2 *1 *8
        # print(keys.shape) # 2 *10 *8
        # 在维度扩展后，
        # queries的形状：(batch_size，查询的个数，1，num_hidden) (2，1,1,8)
        # key的形状：(batch_size，1，“键－值”对的个数，num_hiddens) (2,1,10,8)
        # print(queries.unsqueeze(2).shape)
        # print(keys.unsqueeze(1).shape)
        features=queries.unsqueeze(2)+keys.unsqueeze(1) #在 dim=2 和dim =1分别增加一维；
        # print(features.shape) #torch.Size([2, 1, 10, 8])
        features=torch.tanh(features)

        # scores的形状：(batch_size，查询的个数，“键-值”对的个数)
        #(2,1,10)
        scores=self.W_v(features).squeeze(-1) #删除最后的那一个维度；
        """
        进行softmax操作：
        """
        self.attention_weights=masked_softmax(scores,valid_lens)
        # values的形状：(batch_size，“键－值”对的个数，值的维度) (2,10,40)

        #做两个矩阵的乘法：第一个矩阵的第三维和第二个矩阵的第二维是一样的要；
        return torch.bmm(self.dropout(self.attention_weights),values)
        """
         (2,1,10) * (2,10,4); ===> (2,1,4)
        """


'batch_size=2  键值对为 10对；'
queries,keys=torch.normal(0,1,(2,1,20)),torch.ones((2,10,2))
#repeat: 2 * 10 * 4大小；
values=torch.arange(40,dtype=torch.float32).reshape(1,10,4).repeat(2,1,1)
valid_lens=torch.tensor([2,6])

#初始化网络：
attention=Addattention(key_size=2,query_size=20,num_hiddens=8,dropout=0.1)
attention.eval()

res=attention(queries,keys,values,valid_lens)
print("res:",res.shape)
print(res)  #注意力汇聚输出的形状为（批量大小，查询的步数，值的维度）。



#缩放点积注意力
class DotProductAttention(nn.Module):

    def __init__(self,dropout,**kwargs):
        super(DotProductAttention, self).__init__()
        self.dropout=nn.Dropout(dropout)

    """
    d是 query 和 key有相同的长度： 比较常用的计算注意力分数：
    """

    # queries的形状：(batch_size，查询的个数，d)
    # keys的形状：(batch_size，“键－值”对的个数，d)
    # values的形状：(batch_size，“键－值”对的个数，值的维度)
    # valid_lens的形状:(batch_size，)或者(batch_size，查询的个数)
    def forward(self,queries,keys,values,valid_lens=None):

        d=queries.shape[-1]
        scores=torch.bmm(queries,keys.transpose(1,2)) /math.sqrt(d)
        self.attention_weights=masked_softmax(scores,valid_lens)

        return torch.bmm(self.dropout(self.attention_weights),values)


# queries的形状：(batch_size，查询的个数，d)
queries=torch.normal(0,1,(2,1,2)) #d=2
# keys的形状：(batch_size，“键－值”对的个数，d)
kes=torch.ones((2,10,2))
# values的形状：(batch_size，“键－值”对的个数，值的维度)
values=torch.arange(40,dtype=torch.float32).reshape(1,10,4).repeat(2,1,1)
attention=DotProductAttention(dropout=0.5)

attention.eval()
dot_res=attention(queries,keys,values,valid_lens)

print("dot_res:",dot_res)
print(dot_res.shape)