# author:Lenovo
# datetime:2023/2/22 15:17
# software: PyCharm
# project:pytorch项目

import torch
from torch import nn
from d2l import torch as d2l

#定义vgg块
def vgg_block(num_convs,in_channels,out_channels): #参数是多少卷积层，输入通道、输出通道；

    layer=[]
    for _ in range(num_convs):
        layer.append(nn.Conv2d(in_channels,out_channels,
                               kernel_size=3,padding=1))
        layer.append(nn.ReLU())
        in_channels=out_channels #将第一层的输入当作第二层的输入来算；
    layer.append(nn.MaxPool2d(kernel_size=2,stride=2))

    return nn.Sequential(*layer) #返回以恶搞net网络层；

#前面是多少卷积层，第二个参数是输入的输出的层数，类似于神经元的个数
conv_parameter = ((1, 64), (1, 128), (2, 256), (2, 512), (2, 512))

def vgg(conv_parameter):
    conv_blocks=[]
    in_channels=1#设置初始输入通道为1

    for (nums_conv,out_channels) in conv_parameter:
        #连接所有的卷积层
        conv_blocks.append(vgg_block(nums_conv,in_channels,out_channels))
        in_channels=out_channels #将上一层的输入当作下一层的输出；
    #返回一个vggNet
    return nn.Sequential(
        *conv_blocks,
        nn.Flatten(),#设置全连接层；
        nn.Linear(out_channels * 7 * 7,4096),nn.ReLU(),nn.Dropout(0.5),
        nn.Linear(4096,4096),nn.ReLU(),nn.Dropout(0.5),
        nn.Linear(4096,10),
    )

net=vgg(conv_parameter) #传入参数；

X = torch.randn(size=(1, 1, 224, 224))
for blk in net:
    X=blk(X)
    print(blk.__class__.__name__,'out_shape:\t',X.shape)



