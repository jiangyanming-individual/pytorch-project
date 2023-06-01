# author:Lenovo
# datetime:2023/4/15 17:35
# software: PyCharm
# project:pytorch项目



"""
编码器输出的state是给解码器的一个输入
编码器还需要接受一个X的输入；
"""
from torch import nn


class Encoder(nn.Module):

    def __init__(self,**kwargs):

        super(Encoder, self).__init__()

    def forward(self):
        raise NotImplementedError #没有实现类的错误


class Decoder(nn.Module):

    def __init__(self):
        super(Decoder, self).__init__()

    def init_state(self,enc_outputs,*args):
        raise NotImplementedError

    def forward(self,X,state):

        raise NotImplementedError


class EncoderDecoder(nn.Module):

    def __init__(self,encoder,decoder,**kwargs):
        super(EncoderDecoder, self).__init__()
        self.encoder=encoder
        self.decoder=decoder
    def forward(self,enc_X,dec_X,*args):

        enc_outputs=self.encoder(enc_X) #输出编码器输出状态：
        dec_state=self.decoder.init_state(enc_outputs,*args) #初始化decoder的状态；
        return self.decoder(dec_X,dec_state) #解码器接受两个参数：dec_x的输入和dec_state解码器的状态；