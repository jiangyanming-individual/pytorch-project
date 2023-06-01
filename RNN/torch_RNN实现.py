# author:Lenovo
# datetime:2023/4/13 15:31
# software: PyCharm
# project:pytorch项目

import math
import torch
from torch import nn
from torch.nn import functional as F
from d2l import torch as d2l

batch_size, num_steps = 32, 35
train_iter, vocab = d2l.load_data_time_machine(batch_size, num_steps)

# x=F.one_hot(torch.tensor([0, 2]), len(vocab))
# print(x.shape)
X= torch.arange(10).reshape((2, 5))
print(F.one_hot(X.T, 28).shape) #(时间步长，批量大小，词表的大小)


#得到所有的权重参数：
def get_params(vocab_size,num_hiddens,device):
    num_inputs=num_outputs=vocab_size

    def normal(shape):
        return torch.randn(size=shape,device=device) * 0.01

    #权重参数：
    W_xh=normal((num_inputs,num_hiddens)) #权重参数
    W_hh=normal((num_hiddens,num_hiddens))
    b_h=torch.zeros((num_hiddens),device=device)

    #输出参数：

    W_ho=normal((num_hiddens,num_outputs))
    b_o=torch.zeros(num_outputs,device=device)


    params=[W_xh,W_hh,b_h,W_ho,b_o]
    #梯度：
    for param in params:
        param.requires_grad_(True)
    return params

#初始的H (batch_size,num_hiddens)  X (batch_size,vocab_size)
def rnn_init_state(batch_size,num_hiddens,device):

    return (torch.zeros((batch_size,num_hiddens),device=device),)

#定义的rnn模型执行的过程
def rnn(inputs,state,params):
    # inputs的形状：(时间步数量 (一共多少个inputs)，批量大小，词表大小)
    W_xh,W_hh,b_h,W_ho,b_o=params
    H,=state #初始化H

    print(W_hh.shape)
    print(W_ho.shape)

    outputs=[]
    #X的大小 (batch_size,vocab_size(词表的小大))
    for X in inputs:
        H_t=torch.tanh(torch.matmul(X,W_xh)+ torch.matmul(H,W_hh)+b_h)
        Y_out=torch.matmul(H_t,W_ho) +b_o #输出；
        outputs.append(Y_out)
    return torch.cat(outputs,dim=0),(H,) #竖着拼接；

"""从零开始实现的循环神经网络模型"""
class RNNModelScratch:

    def __init__(self,vocab_size,num_hiddens,device,get_params,rnn_init_state,rnn_fc):
        self.vocab_size,self.num_hiddens=vocab_size,num_hiddens
        self.get_params=get_params(vocab_size,num_hiddens,device)
        self.rnn_init_state,self.rnn_fc=rnn_init_state,rnn_fc

    def __call__(self,X,state):
        #(时间步长，bacth_size,vocab_size)
        X = F.one_hot(X.T, self.vocab_size).type(torch.float32)
        return self.rnn_fc(X,state,self.get_params) #返回一层rnn的结果；(outputs,H隐藏单元)

    # 初始化 H (batch_size,num_hiddens)
    def begin_state(self,batch_size,device):
        return self.rnn_init_state(batch_size,self.num_hiddens,device)


num_hiddens=512

model=RNNModelScratch(len(vocab),num_hiddens=num_hiddens,device='cpu',get_params=get_params,rnn_init_state=rnn_init_state,rnn_fc=rnn)
state=model.begin_state(X.shape[0],device='cpu')  #X.shape[0]=2
# print(state)  #(2,512)
# print(X)
outputs,new_state=model(X,state) #输出结果：X=(2,5)


print(outputs.shape)
print(new_state[0].shape) #隐状态保持不变；

def predict_ch8(prefix,num_preds,net,vocab,device):


    state=net.begin_state(batch_size=1,device=device)
    outputs=[vocab[prefix[0]]]

    get_input=lambda :torch.tensor([outputs[-1]],device=device).reshape((1,1))

    for y in prefix[1:]:
        _,state=net(get_input(),state)
        outputs.append(vocab[y])

    for _ in range(num_preds):# 预测num_preds步
        y,state=net(get_input(),state)
        outputs.append(int(y.argmax(dim=1).reshape(-1)))

    return ''.join([vocab.idx_to_token[i] for i in outputs])


res=predict_ch8('time traveller ', 10, model, vocab,device='cpu')
print(res)

#梯度剪裁：


def grad_clipping(net,theta):

    if isinstance(net,nn.Module):
        params=[p for p in net.parameters() if p.requires_grad]
    else:
        params=net.params
    norm=torch.sqrt(sum(torch.sum((p.grad ** 2)) for p in params))

    if norm >theta:
        for param in params:
            param.grad[:]*=theta /norm





