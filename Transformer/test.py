# author:Lenovo
# datetime:2023/6/11 13:03
# software: PyCharm
# project:pytorch项目

import torch
import torch.nn as nn

# hidden_state=torch.arange(40).resize(2,2,10)
# print(hidden_state)
#
# print(hidden_state[-1])
# print(hidden_state[-1].shape)

# x=torch.randn((10,3)) #(seq_len,batch_size,features_size)
# print(x)

x=[[1,3,4,5,6,8],[9,11,12,7,6,1]] #(2,6)
x=torch.tensor(x)
x=x.permute(1,0)
print(x.shape)

embedding=torch.nn.Embedding(num_embeddings=21,embedding_dim=100)

x=embedding(x)
print("x_embedding:",x.shape)
rnn=nn.RNN(input_size=100,hidden_size=20,num_layers=4) #4层RNN；

out,h=rnn(x)

print("out shape:",out.shape)
print("h shape",h.shape)