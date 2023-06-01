# author:Lenovo
# datetime:2023/2/7 18:51
# software: PyCharm
# project:pytorch项目

import torch
import random
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")
def generate_data(w,b,num_examples):

    #f(x)=X * w +b
    X=torch.normal(0,1,(num_examples,len(w)))
    print(X.shape)
    y=torch.matmul(X,w) +b
    print(y.shape)
    #加入噪音：
    y+=torch.normal(0,0.01,y.shape)
    return X,y.reshape((-1,1))  #返回一列的label的值；

#真实的数据 w,b ;
true_w=torch.tensor([2,-3.4])
true_b=4.2

#训练集的大小：
X,y_label=generate_data(true_w,true_b,1000)
# print(X,'\n',y_label)
#
# print('X:',X[0],'\ny_label:',y_label[0])

plt.figure(figsize=(4,3),dpi=100)
plt.scatter(X[:,1].detach().numpy(),y_label.detach().numpy(),s=1)
# plt.show()

#读取数据：

def data_iter(batch_size,x,y):

    num_examples=len(x)
    indices=list(range(num_examples)) #形成列表；
    # print(indices)
    random.shuffle(indices) #随机打乱数据
    for i in range(0,num_examples,batch_size):
        #将列表元素转为tensor；
        batch_indices=torch.tensor(indices[i:(i+1) * batch_size])
        # print(batch_indices)

        # 生成器，每次生成batch_size大小的数据；
        yield x[batch_indices],y[batch_indices]

batch_size=10

#输出打印一下：
for X,y_label in data_iter(batch_size,X,y_label):
    print(X,'\n',y_label)
    break

#初始化参数模型：w ,b

w=torch.normal(0,0.01,size=(2,1),requires_grad=True)
b=torch.zeros(1,requires_grad=True)


#定义线性模型：
def Liner_model(X,w,b):
    return torch.matmul(X,w) + b

#定义损失函数：
def squared_loss(y_hat,y_label):
    """均方损失"""
    return (y_hat -y_label.reshape(y_hat.shape)) ** 2/ 2

#定义梯度下降算法：

def sgd(params,learning_rate,batch_size):

    """小批量随机梯度下降
        params 是[w,b]
    """
    with torch.no_grad():
        # param.grad：是对w,b求导数；
        for param in params:
            param -=learning_rate * param.grad/ batch_size
            param.grad.zero_() #梯度更新为0；

#训练模型：
learning_rate=0.03
num_epochs=3 #训练3次
Liner_net=Liner_model #模型函数 得到预测值；y_hat
loss=squared_loss

for epoch in range(num_epochs):

    for x,y in data_iter(batch_size,X,y_label):
        # loss_data是一个向量，X:[batch_size,2] w:[2,1]==>loss_data是[batch_size,1]
        loss_data=loss(Liner_net(x,w,b),y)
        # 反向传播求导；因为loss_data是向量，所以要把它相加变成标量以后再求导数,就是对w,b求导
        loss_data.sum().backward()
        print(f"w.grad:{w.grad},b.grad:{b.grad}")
        # print(loss_data.sum().backward())
        sgd([w,b],learning_rate,batch_size) #传入[w,b]然后求导，更新[w,b]

    #得到最佳的w,b 去训练集测试一下
    with torch.no_grad():
        train_loss=loss(Liner_net(X,w,b),y_label)
        print(f"epoch:{epoch + 1},loss:{float(train_loss.mean()):f}")

print(f'w的估计误差: {true_w - w.reshape(true_w.shape)}')
print(f'b的估计误差: {true_b - b}')






















