# author:Lenovo
# datetime:2023/6/21 18:16
# software: PyCharm
# project:pytorch项目


import torch
import torch.nn as nn

class SelfAttention(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(SelfAttention, self).__init__()
        self.hidden_size = hidden_size

        self.query = nn.Linear(input_size, hidden_size)
        self.key = nn.Linear(input_size, hidden_size)
        self.value = nn.Linear(input_size, hidden_size)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, x):

        q = self.query(x)
        k = self.key(x)
        v = self.value(x)#[batch_size,seq_len,hidden_size]


        # print(k.transpose(1, 2).shape) [batch_size,hidden_size,seq_len]
        #注意力得分：使用点积的形式：[batch_size,seq_len,hidden_size] * [batch_size,hidden_size,seq_len]
        attention_scores = torch.bmm(q, k.transpose(1, 2)) / torch.sqrt(torch.tensor(self.hidden_size).float())
        # print("attention_scores shape:",attention_scores.shape)

        #经过softmax进行掩码：
        attention_weights = self.softmax(attention_scores)
        # print("attention_weights shape:",attention_weights.shape)

        #最后的输出 [batch_size,seq_len,seq_len] [batch_size,seq_len,hidden_size]
        output = torch.bmm(attention_weights, v)

        # print(output.shape) # [batch_size,seq_len,hidden_size]
        return output

# 测试自注意力机制的代码
input_size = 128
hidden_size = 64
seq_length = 10
batch_size = 32

x = torch.randn(batch_size, seq_length, input_size)
self_attention = SelfAttention(input_size, hidden_size)
output = self_attention(x)

print("Input shape:", x.shape)
print("Output shape:", output.shape)