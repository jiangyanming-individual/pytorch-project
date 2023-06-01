# author:Lenovo
# datetime:2023/2/12 15:19
# software: PyCharm
# project:pytorch项目


import torch
from torch import nn
from d2l import torch as d2l

#加载数据：
batch_size=256
train_data,test_data=d2l.load_data_fashion_mnist(batch_size)


#设置参数：
nums_inputs,nums_outputs,nums_hiddren=784,10,256

W1=nn.Parameter(
    torch.randn(nums_inputs,nums_hiddren,requires_grad=True) * 0.01
)
b1=nn.Parameter(torch.zeros(nums_hiddren,requires_grad=True))

W2=nn.Parameter(
    torch.randn(nums_hiddren,nums_outputs,requires_grad=True) * 0.01
)
b2=nn.Parameter(
    torch.zeros(nums_outputs,requires_grad=True)
)

params=[W1,b1,W2,b2]
#设置激活函数：
def relu(X):

    y=torch.zeros_like(X)
    return torch.max(X,y)


#设置模型：

def net(X):
    X=X.reshape(-1,nums_inputs)
    H=relu(X@W1 + b1) # 784 * (784 * 256) + 256;
    return (H@W2 + b2) # 256 * (256 * 10 ) + 10

#损失函数：
loss=nn.CrossEntropyLoss(reduction='none')


#训练：
num_epochs,lr=10,0.1
updater=torch.optim.SGD(params,lr=lr) #设置优化器；

d2l.train_ch3(net,train_data,test_data,loss,num_epochs,updater)





#简洁实现softmax

net2=nn.Sequential(
    nn.Flatten(),
    nn.Linear(784,256),
    nn.ReLU(),
    nn.Linear(256,10)
)

def init_weights(X):
    if type(X) == nn.Linear():
        nn.init.normal_(X.weight,mean=0,std=0.01)

net2.apply(init_weights)

loss=nn.CrossEntropyLoss(reduction='none')
trainer=torch.optim.SGD(net2.parameters(),lr=lr) #优化w,b的值；

















