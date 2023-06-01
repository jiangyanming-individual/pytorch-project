# author:Lenovo
# datetime:2023/4/14 10:50
# software: PyCharm
# project:pytorch项目


import torch
from torch import nn
from torch.nn import functional as F
from d2l import torch as d2l

batch_size, num_steps = 32, 35
train_iter, vocab = d2l.load_data_time_machine(batch_size, num_steps)


num_hiddens=512
rnn_layer=nn.RNN(len(vocab),num_hiddens)

#初始化H的隐藏状态：
state=torch.zeros((1,batch_size,num_hiddens))
print(state.shape)#torch.Size([1, 32, 512])

"""
通过一个隐状态和一个输入，我们就可以用更新后的隐状态计算输出。rnn_layer缺少一个输出
输入X
"""
X=torch.rand(size=(num_steps,batch_size,len(vocab)))
Y,new_state=rnn_layer(X,state)
"""

W_xh=(vocab_size,num_hiddens)
W_hh=(num_hiddens,num_hiddens)
"""
print(Y.shape) #torch.Size([35, 32, 512])
print(new_state.shape) #torch.Size([1, 32, 512])


class RNNModel(nn.Module):
    def __init__(self,rnn_layer,vocab_size,**kwargs):

        super(RNNModel,self).__init__(**kwargs)
        self.rnn=rnn_layer
        self.vocab_size=vocab_size
        self.num_hiddens=self.rnn.hidden_size

        if not self.rnn.bidirectional:
            self.num_directions=1
            #输出：
            self.linear=nn.Linear(self.num_hiddens,self.vocab_size)
        else:
            #双向的RNN输出是num_hiddens * 2
            self.num_directions=2
            self.linear=nn.Linear(self.num_hiddens * 2,self.vocab_size)
    def forward(self,inputs,state):
        X=F.one_hot(inputs.T.long(),self.vocab_size)
        X=X.to(torch.float32)
        Y,state=self.rnn(X,state) #X,state是输入；

        # 全连接层首先将Y的形状改为(时间步数*批量大小,隐藏单元数)
        output=self.linear(Y.reshape((-1,Y.shape[-1])))# 它的输出形状是(时间步数*批量大小,词表大小)。
        return output,state
    
    def begin_state(self,device,batch_size=1):

        if not isinstance(self.rnn,nn.LSTM):
            #隐藏状态：(H)
            return torch.zeros(
                (self.num_directions * self.rnn.num_layers,batch_size,self.num_hiddens)
                ,device=device)

        else:
            # LSTM是隐藏状态：两个隐状态(H,C)
            return (
                torch.zeros(
                    (self.num_directions * self.rnn.num_layers,batch_size,self.num_hiddens),
                    device=device),
                torch.zeros((
                    self.num_directions * self.rnn.num_layers,batch_size,self.num_hiddens),
                    device=device
                )
            )
device='cpu'
net = RNNModel(rnn_layer, vocab_size=len(vocab))
net = net.to(device)
res=d2l.predict_ch8('time traveller', 10, net, vocab, device)
print(res)


num_epochs, lr = 500, 1
res2=d2l.train_ch8(net, train_iter, vocab, lr, num_epochs, device)
print(res2)