# author:Lenovo
# datetime:2023/4/13 15:32
# software: PyCharm
# project:pytorch项目

import torch
from torch import nn
from d2l import torch as d2l

batch_size, num_steps = 32, 35
train_iter, vocab = d2l.load_data_time_machine(batch_size, num_steps)



#初始化权重参数：
def get_lstm_params(vocab_size,num_hiddens,device):

    num_inputs=num_outputs=vocab_size

    def normal(shape):
        return torch.randn(size=shape,device=device) * 0.01

    def three():
        return (
            #W_xh
            normal((num_inputs,num_hiddens)),
            #W_hh
            normal((num_hiddens,num_hiddens)),
            torch.zeros(num_hiddens,device=device)
        )

    W_xi,W_hi,b_i=three() #输入门
    W_xf,W_hf,b_f=three()#遗忘门
    W_xo,W_ho,b_o=three() #输出门
    W_xc,W_hc,b_c=three() #融合记忆单元

    #输出层参数：
    W_hq=normal((num_hiddens,num_outputs))
    b_q=torch.zeros(num_outputs,device=device)

    params=[ W_xi,W_hi,b_i,W_xf,W_hf,b_f,W_xo,W_ho,b_o,W_xc,W_hc,b_c,W_hq,b_q]
    for param in params:
        param.requires_grad_(True)
    return params


#初始的隐藏状态有2个 (H,Cell)
def init_lstm_state(batch_size,num_hiddens,device):
    return (
        torch.zeros((batch_size,num_hiddens),device=device),
        torch.zeros((batch_size,num_hiddens),device=device)
    )

def lstm(inputs,state,params):

    [ W_xi,W_hi,b_i,W_xf,W_hf,b_f,W_xo,W_ho,b_o,W_xc,W_hc,b_c,W_hq,b_q]=params
    (H,C)=state

    outputs=[]

    for X in inputs:
        I=torch.sigmoid((X @ W_xi) +(H @ W_hi) +b_i)
        F=torch.sigmoid((X@ W_xf) +(H @ W_hf) +b_f)
        O=torch.sigmoid((X @ W_xo) +(H @ W_ho) +b_o)
        C_tilda=torch.tanh((X @ W_xc) +(H @ W_hc) +b_c) #融合记忆信号
        C=F*C + I * C_tilda#记忆信号：
        H=O * torch.tanh(C) #融合输出；

        #真正的输出：
        Y=(H @ W_hq) +b_q
        outputs.append(Y)

    return torch.cat(outputs,dim=0),(H,C)


vocab_size,num_hiddens,device=len(vocab),256,'cpu'

num_epochs,lr=500,1
model=d2l.RNNModelScratch(len(vocab),num_hiddens,device,get_lstm_params,init_lstm_state,lstm)
d2l.train_ch8(model, train_iter, vocab, lr, num_epochs, device)

