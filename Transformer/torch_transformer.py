# author:Lenovo
# datetime:2023/4/17 11:00
# software: PyCharm
# project:pytorch项目


import math
import pandas as pd
import torch
from torch import nn
from d2l import torch as d2l


"""
前馈网络：
基于位置的前馈网络对序列中的所有位置的表示进行变换时使用的是同一个多层感知机（MLP）
"""
class PositionWiseFFN(nn.Module):

    def __init__(self,ffn_num_inputs,ffn_num_hiddens,ffn_num_outputs):

        super(PositionWiseFFN, self).__init__()
        self.dense1=nn.Linear(ffn_num_inputs,ffn_num_hiddens)
        self.relu=nn.ReLU()
        self.dense2=nn.Linear(ffn_num_hiddens,ffn_num_outputs)

    def forward(self,X):
        return self.dense2(self.relu(self.dense1(X)))

# 两层的感知机转换成形状为（批量大小，时间步数，ffn_num_outputs）的输出张量

ffn=PositionWiseFFN(4,4,8) #(inputs=4 ,num_hidden=4,outputs=8)
ffn.eval()
res=ffn(torch.ones((2,3,4))) #===>(2,3,4) =>(2,3,4)=>(2,3,8)


"""
残差连接和层规范层：
"""

ln=nn.LayerNorm(2)
bn=nn.BatchNorm1d(2)

X=torch.tensor([[1,2],[2,3]],dtype=torch.float32)
print("layer norm:",ln(X),"\nbatch_norm:",bn(X))

#add 和 norm

class AddNorm(nn.Module):

    def __init__(self,normalized_shape,dropout,**kwargs):
        super(AddNorm, self).__init__()
        self.dropout=nn.Dropout(dropout)
        self.ln=nn.LayerNorm(normalized_shape) #输入大小

    def forward(self,X,Y):

        return self.ln(self.dropout(Y)+X) #add 和 norm

add_norm=AddNorm([3,4],0.5)
add_norm.eval()
add_norm_res=add_norm(torch.ones((2,3,4)),torch.ones((2,3,4))).shape
# print("add_norm_res:",add_norm_res)

#transformer编码块
class EncoderBlock(nn.Module):

    """
    两个规范层、一个多头注意力、一个位置感知层
    """
    def __init__(self,key_size,query_size,value_size,num_hiddens,norm_shape,ffn_inputs,
                 ffn_hiddens,num_heads,dropout,use_bias=False,**kwargs):

        super(EncoderBlock, self).__init__()
        self.attention=d2l.MultiHeadAttention(key_size,query_size,value_size,num_hiddens,
                                              num_heads,dropout,use_bias)
        self.addnorm1=AddNorm(norm_shape,dropout)
        self.ffn=PositionWiseFFN(ffn_inputs,ffn_hiddens,num_hiddens)
        self.addnorm2=AddNorm(norm_shape,dropout)
    def forward(self,X,valid_lens):

        Y=self.addnorm1(X,self.attention(X,X,X,valid_lens))
        return self.addnorm2(Y,self.ffn(Y)) #第二层加与规范化；


X=torch.ones((2,100,24))
valid_lens=torch.tensor([3,2])
encoder_blk=EncoderBlock(24,24,24,24,[100,24],24,48,8,0.5)
encoder_blk.eval()
print(encoder_blk(X,valid_lens).shape)
#transformer编码器：


class TransformerEncoder(d2l.Encoder):
    """Transformer编码器"""

    def __init__(self,vocab_size,key_size,query_size,value_size,
                 num_hiddens,norm_shape,ffn_num_inputs,ffn_num_hiddens,
                 num_heads,num_layers,dropout,use_bias=False,**kwargs):
        super(TransformerEncoder, self).__init__()

        self.num_hiddens=num_hiddens
        self.embedding=nn.Embedding(vocab_size,num_hiddens)
        #基于位置的编码：
        self.pos_encoding=d2l.PositionalEncoding(num_hiddens,dropout)

        self.blks=nn.Sequential()

        for i in range(num_layers):
            self.blks.add_module("block"+str(i),
                EncoderBlock(key_size,query_size,value_size,num_hiddens,
                             norm_shape,ffn_num_inputs,ffn_num_hiddens,num_heads,
                             dropout,use_bias)
                        )
    def forward(self,X,valid_lens,*args):

        X=self.pos_encoding(self.embedding(X) * math.sqrt(self.num_hiddens))

        self.attention_weights=[None] * len(self.blks)
        for i ,blk in enumerate(self.blks):
            X=blk(X,valid_lens)
            self.attention_weights[i]=blk.attention.attention.attention_weights
        return X


"""
初始化tansformer模型的参数：
"""
encoder=TransformerEncoder(200,24,24,24,24,[100,24],24,48,8,2,0.5)
encoder.eval()
print(encoder(torch.ones((2,100),dtype=torch.long),valid_lens=valid_lens).shape)
#transformer解码块


class DecoderBlock(nn.Module):

    def __init__(self,key_size,query_size,value_size,num_hiddens,
                 norm_shape,ffn_num_inputs,ffn_num_hiddens,num_heads,
                 dropout,i,**kwargs):
        super(DecoderBlock, self).__init__()
        self.i=i
        self.attention1=d2l.MultiHeadAttention(
            key_size,query_size,value_size,num_hiddens,num_heads,dropout
        )
        self.addnorm1=AddNorm(norm_shape,dropout)

        self.attention2=d2l.MultiHeadAttention(
            key_size,query_size,value_size,num_hiddens,num_heads,dropout
        )

        self.addnorm2=AddNorm(norm_shape,dropout)

        self.ffn=PositionWiseFFN(ffn_num_inputs,ffn_num_hiddens,num_hiddens)
        self.addnorm3=AddNorm(norm_shape,dropout)


    def forward(self,X,state):

        enc_outputs,enc_valid_lens=state[0],state[1]

        # 训练阶段，输出序列的所有词元都在同一时间处理，
        # 因此state[2][self.i]初始化为None。
        # 预测阶段，输出序列是通过词元一个接着一个解码的，
        # 因此state[2][self.i]包含着直到当前时间步第i个块解码的输出表示
        if state[2][self.i] is None:
            key_values=X
        else:
            key_values=torch.cat((state[2][self.i],X),axis=1)
        state[2][self.i]=key_values

        if self.training:

            batch_size,num_steps,_=X.shape
            # dec_valid_lens的开头:(batch_size,num_steps),
            dec_valid_lens=torch.arange(
                1,num_steps+1,device='cpu').repeat(batch_size,1)
        else:
            dec_valid_lens=None

        #自注意力： 输入query key value ,valid_lens
        X2=self.attention1(X,key_values,key_values,dec_valid_lens)
        Y=self.addnorm1(X,X2)
        Y2=self.attention2(Y,enc_outputs,enc_outputs,enc_valid_lens)
        Z=self.addnorm2(Y,Y2)


        #输出：解码器多了一层addnorm
        return self.addnorm3(Z,self.ffn(Z)),state

decoder_blk = DecoderBlock(24, 24, 24, 24, [100, 24], 24, 48, 8, 0.5, 0)
decoder_blk.eval()
X = torch.ones((2, 100, 24))
state = [encoder_blk(X, valid_lens), valid_lens, [None]]
print(decoder_blk(X,state)[0].shape)



#transformer解码器：
class TransformerDecoder(d2l.AttentionDecoder):


    def __init__(self,vocab_size,key_size,query_size,value_size,
                 num_hiddens,norm_shape,ffn_num_inputs,ffn_num_hiddens,
                 num_heads,num_layers,dropout,**kwargs
                 ):

        self.num_hiddens=num_hiddens
        self.num_layers=num_layers
        self.embedding=nn.Embedding(vocab_size,num_hiddens)
        self.pos_encoding=d2l.PositionalEncoding(num_hiddens,dropout)
        self.blks=nn.Sequential()

        for i in range(num_layers):
            self.blks.add_module("block"+str(i),
                DecoderBlock(key_size,query_size,value_size,num_hiddens,
                             norm_shape,ffn_num_inputs,ffn_num_hiddens,num_heads,dropout,i)
                        )
        """
        输出：
        """
        self.dense=nn.Linear(num_hiddens,vocab_size)
    def init_state(self,enc_outputs,enc_valid_lens,*args):
        return [enc_outputs,enc_valid_lens,[None] * self.num_layers]

    def forward(self,X,state):
        X=self.pos_encoding(self.embedding(X) * math.sqrt(self.num_hiddens))
        self._attention_weights=[[None]*len(self.blks) for _ in range(2)]

        for i ,blk in enumerate(self.blks):
            X,stat=blk(X,state)

            self._attention_weights[0][i]=blk.attention1.attention.attention_weights
            self._attention_weights[1][i]=blk.attention2.attention.attention_weights

        return self.dense(X),state

    @property #修饰只读的属性：
    def attention_weights(self):
        return self._attention_weights
