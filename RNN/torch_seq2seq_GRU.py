# author:Lenovo
# datetime:2023/4/14 15:17
# software: PyCharm
# project:pytorch项目

import collections
import math
import torch
from torch import nn
from d2l import torch as d2l


#编码器：编码器没有真正的输出层===>全连接层

class Seq2seqEncoder(d2l.Encoder):
    def __init__(self,vocab_size,embed_size,num_hiddens,
                 num_layers,dropout=0,**kwargs):
        super(Seq2seqEncoder, self).__init__(**kwargs)
        #嵌入层：
        self.embedding=nn.Embedding(vocab_size,embed_size)
        self.rnn=nn.GRU(embed_size,num_hiddens,num_layers,dropout=dropout)

    def forward(self,X,*args):
        print(X.shape)  #torch.Size([4, 7])
        # 输出'X'的形状：(batch_size,num_steps,embed_size)
        X=self.embedding(X)
        # 在循环神经网络模型中，第一个轴对应于时间步
        print(X.shape) #torch.Size([4, 7, 8])
        X=X.permute(1,0,2)

        # 设置初始的状态；W_xh=(inputs,num_hiddens)
        #W_hh=(num_hiddens,num_hiddens)
        output,state=self.rnn(X)
        return output,state
        # output的形状:(num_steps,batch_size,num_hiddens)
        # state的形状:(num_layers,batch_size,num_hiddens)

#解码器：
encoder=Seq2seqEncoder(vocab_size=10,embed_size=8,num_hiddens=16,num_layers=2)
encoder.eval()

X=torch.zeros((4,7),dtype=torch.long)
output,state=encoder(X)
print(output.shape)#torch.Size([7, 4, 16])
print(state.shape) #toch_size(2,4,16)

#解码器：


class Seq2seqDecoder(d2l.Decoder):

    def __init__(self,vocab_size,embed_size,num_hiddens,num_layers,dropout=0,**kwargs):
        super(Seq2seqDecoder, self).__init__()

        self.embedding=nn.Embedding(vocab_size,embed_size)
        #encoder输出 + decoder的输入
        self.rnn=nn.GRU(embed_size+num_hiddens,num_hiddens,num_layers,dropout=dropout)

        #输出
        self.dense=nn.Linear(num_hiddens,vocab_size)

    def init_state(self, enc_outputs, *args):
        return enc_outputs[1] #encoder的输出state

    def forward(self,X,state):
        X=self.embedding(X).permute(1,0,2) #转换维度：(num_steps,batch_size,num_hiddens)
        print("X:",X.shape)

        # 广播context，使其具有与X相同的num_steps
        # state[-1] #取三中的一个二维 (4,16)

        #使用ecoder的状态输出：作decoder的输入：
        context=state[-1].repeat(X.shape[0],1,1) #每个维度复制一次
        X_and_context=torch.cat((X,context),dim=2) #第三列 累加；(8+16=24)
        print("X_and_context:",X_and_context.shape)

        output,state=self.rnn(X_and_context,state)
        # output(num_steps,batch_size,num_hiddens)
        #state(num_layer,batch_size,num_hiddens)
        #真正的输出：
        output=self.dense(output).permute(1,0,2)

        return output,state

decoder=Seq2seqDecoder(vocab_size=10,embed_size=8,num_hiddens=16,num_layers=2)
decoder.eval()

#encoder的输出：state ，作为decoder的state的输入；
state=decoder.init_state(encoder(X))
'decoder接受两个输入：X 和state;'
output,state=decoder(X,state)

print("decoder_output_shape",output.shape)
print("decoder_state_shape",state.shape)


