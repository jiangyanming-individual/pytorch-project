# author:Lenovo
# datetime:2023/2/22 16:25
# software: PyCharm
# project:pytorch项目



import torch
from torch import nn
from d2l import torch as d2l


"""
使用ninnet的原因是要替代掉，全连接层，使得计算机的计算资源运用的不算那么大；
"""
#设置nin块
def nin_block(in_channels,out_channels,kernel_size,stride,padding):


    return nn.Sequential(
        nn.Conv2d(in_channels,out_channels,kernel_size,stride,padding),nn.ReLU(),
        nn.Conv2d(out_channels,out_channels,kernel_size=1),nn.ReLU(),
        nn.Conv2d(out_channels,out_channels,kernel_size=1),nn.ReLU(),
    )


#构建nin网络
net=nn.Sequential(

    nin_block(1,96,kernel_size=11,stride=4,padding=0),
    nn.MaxPool2d(kernel_size=3,stride=2),

    nin_block(96,256,kernel_size=5,stride=1,padding=2),
    nn.MaxPool2d(kernel_size=3,stride=2), #池化层，将元素的特征不断的缩小；

    nin_block(256,384,kernel_size=3,stride=1,padding=1),
    nn.MaxPool2d(kernel_size=3,stride=2),
    nn.Dropout(0.5),

    nin_block(384,10,kernel_size=3,stride=1,padding=1),
    nn.AdaptiveAvgPool2d((1,1)), #全局平均池化层；

    nn.Flatten(),
)


X=torch.rand(size=(1,1,224,224))

for layer in net :
    X=layer(X)

    print(layer.__class__.__name__,'out_shape:\t',X.shape)
