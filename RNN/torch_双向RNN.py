# author:Lenovo
# datetime:2023/4/14 13:26
# software: PyCharm
# project:pytorch项目


import torch
from torch import nn
from d2l import torch as d2l


batch_size,num_steps,device=32,35,'cpu'
train_iter, vocab = d2l.load_data_time_machine(batch_size, num_steps)

vocab_size,num_hiddens,num_layer=len(vocab),256,2  #num_layer=2
num_inputs=vocab_size

#bidirectional 保证是双向的RNN
lstm_layer=nn.LSTM(num_inputs,num_hiddens,num_layer,bidirectional=True)
model=d2l.RNNModel(lstm_layer,len(vocab)) #传入RNN层；
model=model.to(device)

num_epochs, lr = 500, 1
d2l.train_ch8(model, train_iter, vocab, lr, num_epochs, device)



