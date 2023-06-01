# author:Lenovo
# datetime:2023/4/16 10:39
# software: PyCharm
# project:pytorch项目


import torch
from torch import nn
from d2l import torch as d2l

class AttentionDecoder(d2l.Decoder):
    """带有注意力机制解码器的基本接口"""
    def __init__(self, **kwargs):
        super(AttentionDecoder, self).__init__(**kwargs)

    @property
    def attention_weights(self):
        raise NotImplementedError


#带有注意力机制的RNN：
class Seq2SeqAttentionDecoder(AttentionDecoder):

    def __init__(self,vocab_size,embed_size,num_hiddens,num_layers,dropout=0,**kwargs):
        super(Seq2SeqAttentionDecoder, self).__init__()
        self.attention=d2l.AdditiveAttention(num_hiddens,num_hiddens,num_hiddens,dropout)
        self.embedding=nn.Embedding(vocab_size,embed_size)
        self.rnn=nn.GRU(embed_size+num_hiddens,num_hiddens,num_layers,dropout=dropout)

        #输出层：
        self.dense=nn.Linear(num_hiddens,vocab_size)


    def init_state(self, enc_outputs, enc_valid_lens,*args):#GRU的输出

        # outputs的形状为(batch_size，num_steps，num_hiddens).
        # hidden_state的形状为(num_layers，batch_size，num_hiddens)
        outputs,hidden_state=enc_outputs

        """
        输出：
        """
        return (outputs.permute(1,0,2),hidden_state,enc_valid_lens)

    def forward(self, X, state):
        """

        :param X:是decoder的输入
        :param state: encoder的输出；
        :return:
        """
        enc_outputs,hidden_state,enc_valid_lens=state
        #输出X的形状是：(num_steps,batch_size,embed_size)
        X=self.embedding(X).permute(1,0,2)

        outputs,self._attentions_weight=[],[]
        for x in X:

            #上一个时间步的state作为query
            # query的形状为(batch_size,1,num_hiddens)
            query=torch.unsqueeze(hidden_state[-1],dim=1)
            #encoder的输出作为 attention的key和value；
            context=self.attention(
                query,enc_outputs,enc_outputs,enc_valid_lens
            )

            #拼接：
            x=torch.cat((context,torch.unsqueeze(x,dim=1)),dim=-1)

            #输入到rnn层 输入X 和hidden_state
            out,hidden_state=self.rnn(x.permute(1,0,2),hidden_state)
            outputs.append(out)
            self._attentions_weight.append(self.attention.attention_weights)

        #真正的输出：竖着拼接
        outputs=self.dense(torch.cat(outputs,dim=0))

        return outputs.permute(1,0,2),[enc_outputs,hidden_state,enc_valid_lens]

    @property
    def attention_weights(self):
        return self._attention_weights

encoder=d2l.Seq2SeqEncoder(vocab_size=10,embed_size=8,num_hiddens=16,num_layers=2)
encoder.eval()

decoder=Seq2SeqAttentionDecoder(vocab_size=10,embed_size=8,num_hiddens=16,num_layers=2)
decoder.eval()

X = torch.zeros((4, 7), dtype=torch.long)  # (batch_size,num_steps)
state=decoder.init_state(encoder(X),None)
output,state=decoder(X,state)

print(output.shape)
print("len(state):",len(state))
print(state[0].shape) #enc_outputs
print("len(state[1]):",len(state[1]))#hidden_state
print(state[1][0].shape)
