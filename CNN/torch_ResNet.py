# author:Lenovo
# datetime:2023/2/22 16:50
# software: PyCharm
# project:pytorch项目


import torch
from torch import nn
from torch.nn import functional as F
from d2l import torch as d2l
import warnings
warnings.filterwarnings("ignore")


"""
残差网络在CNN中使用的比较的广泛；
"""
#自定义残差网络块；
class Residual(nn.Module):

    def __init__(self, in_channels, out_channels, use_1xconv=False, strides=1):

        super(Residual,self).__init__()
        # 第一层卷积层；
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=strides)
        # 第二层卷积层；
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1) #没有strides

        if use_1xconv:
            # 如果需要使用额外的卷积块；1 * 1的卷积大小
            self.conv3 = nn.Conv2d(in_channels, out_channels,
                                  kernel_size=1, stride=strides)
        else:
            self.conv3 = None
        self.bn1 = nn.BatchNorm2d(out_channels)  # 批量规范化；
        self.bn2 = nn.BatchNorm2d(out_channels)  # 批量规范化

    def forward(self, X):

        x=self.conv1(X)
        x=self.bn1(x)
        Y= F.relu(x)  # 在第一层后面加一个Relu激活函数\

        x=self.conv2(Y)
        Y=self.bn2(x)

        if self.conv3:
            # 如果需要额外的1 *1的卷积层加入话：
            x=self.conv3(X)
        Y+=x
        return F.relu(Y)  #最后加一个relu层；


block= Residual(3, 3)
X=torch.rand(size=(4,3,6,6)) #(n,channel,Hight,weight)
Y=block(X)

print(Y.shape)


#进行宽度和高度减半
blk=Residual(3,6,use_1xconv=True,strides=2) #需要步幅为2；
print(blk(X).shape)


b1=nn.Sequential(
    nn.Conv2d(1,64,kernel_size=7,stride=2,padding=3),#高宽减半；
    nn.BatchNorm2d(64),nn.ReLU(),
    nn.MaxPool2d(kernel_size=3,stride=2,padding=1),#池化层减半；
    )


def resnet_block(input_channels,output_channels,num_residuals,first_block=False):
    blk=[]
    for i in range(num_residuals):
        if i == 0 and not first_block:
            blk.append(Residual(input_channels,output_channels,use_1xconv=True,strides=2))

        else:
            blk.append(Residual(output_channels,output_channels)) #stride=1 不加1 *1卷积层输入输出通道不变；


    return blk


b2=nn.Sequential(*resnet_block(64,64,2,first_block=True)) #b2不使用 1 *1;
b3=nn.Sequential(*resnet_block(64,128,2))
b4=nn.Sequential(*resnet_block(128,256,2))
b5=nn.Sequential(*resnet_block(256,512,2))


#最后加入全连接，和平均池化；
net=nn.Sequential(b1,b2,b3,b4,b5,
                nn.AdaptiveAvgPool2d((1,1)),
                nn.Flatten(),
                nn.Linear(512,10),
              )


X = torch.rand(size=(1, 1, 224, 224))
for layer in net:
    X=layer(X)
    print(layer.__class__.__name__,'output shape:\t',X.shape)



lr, num_epochs, batch_size = 0.05, 10, 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size, resize=96)
d2l.train_ch6(net, train_iter, test_iter, num_epochs, lr,device='cpu')