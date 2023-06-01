# author:Lenovo
# datetime:2023/2/12 18:43
# software: PyCharm
# project:pytorch项目


import torch
from torch import nn
from d2l import torch as d2l

def dropout_layer(X,dropout):

    assert 0<=dropout<=1

    if dropout == 1:
        return torch.zeros_like(X)
    if dropout == 0:
        return X

    mask=(torch.rand(X.shape) > dropout).float()
    print(mask)
    return mask * X / 1- dropout

X=torch.arange(16,dtype=torch.float32).reshape(2,8)
print(X)
print(dropout_layer(X,0))
print(dropout_layer(X,1))
print(dropout_layer(X,0.5))


dropout1=0.2
dropout2=0.5

nums_inputs,nums_outputs,nums_hiddens1,nums_hiddens2=784,10,256,256

class Net(nn.Module):

    #初始化函数：
    def __init__(self,nums_inputs,nums_outputs,nums_hiddens1,nums_hiddens2,is_Training=True):

        super(Net,self).__init__()
        self.nums_inputs=nums_inputs
        self.is_Training=is_Training
        self.ln1=nn.Linear(nums_inputs,nums_hiddens1)
        self.ln2=nn.Linear(nums_hiddens1,nums_hiddens2)
        self.ln3=nn.Linear(nums_hiddens2,nums_outputs)
        self.Relu=nn.ReLU()


    def forward(self,X):

        H1=self.Relu(self.ln1(X.reshape((-1,self.nums_inputs)))) # 输入784 输入256

        if self.is_Training ==True:
            H1=dropout_layer(H1,dropout1)

        H2=self.Relu(self.ln2(H1)) #输入256 输出256

        if self.is_Training == True:
            H2=dropout_layer(H2,dropout2)

        out=self.ln3(H2)
        return out


net=Net(nums_inputs,nums_outputs,nums_hiddens1,nums_hiddens2,True)



"""简洁实现："""
net3=nn.Sequential(
    nn.Flatten(),
    nn.Linear(784,256),
    nn.ReLU(),
    nn.Dropout(dropout1),

    nn.Linear(256,256),
    nn.ReLU(),
    nn.Dropout(dropout2),

    nn.Linear(256,10)
)

def init_weight(m):
    if type(m) == nn.Linear:
       nn.init.normal_(m.weight,0,std=0.01)

net.apply(net3)






