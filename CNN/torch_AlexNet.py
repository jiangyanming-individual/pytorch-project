# author:Lenovo
# datetime:2023/2/22 10:55
# software: PyCharm
# project:pytorch项目


import torch
from torch import nn
from d2l import torch as d2l

net=nn.Sequential(
    nn.Conv2d(1,96,kernel_size=11,stride=4,padding=1),nn.ReLU(),#(1,96,54,54)
    nn.MaxPool2d(kernel_size=3,stride=2),#(1,96,26,26)

    nn.Conv2d(96,256,kernel_size=5,padding=2),nn.ReLU(),#(1,256,26,26)
    nn.MaxPool2d(kernel_size=3,stride=2),#(1,256,12,12) #池化层不用写输入输出的大小通道数。

    nn.Conv2d(256,384,kernel_size=3,padding=1),nn.ReLU(),#(1,384,12,12)
    nn.Conv2d(384,384,kernel_size=3,padding=1),nn.ReLU(),
    nn.Conv2d(384,256,kernel_size=3,padding=1),nn.ReLU(),
    nn.MaxPool2d(kernel_size=3,stride=2),#(1,256,5,5) 池化层的输入输出通道数是一样的
    #全连接层：
    nn.Flatten(),
    nn.Linear(6400,4096),nn.ReLU(), #输出是4096
    nn.Dropout(0.5),
    nn.Linear(4096,4096), nn.ReLU(),#输出是4096
    nn.Dropout(0.5),
    nn.Linear(4096,1000),#输出是1000
)


X=torch.randn(size=(1,1,224,224)) #通道数是为1的
for layer in net:
    X=layer(X)
    print(layer.__class__.__name__,'out_shape:\t',X.shape)



