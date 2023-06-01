# author:Lenovo
# datetime:2023/2/20 15:14
# software: PyCharm
# project:pytorch项目


import torch
from torch import  nn
from d2l import torch as d2l

#定义LeNet模型
net=nn.Sequential(
    nn.Conv2d(1,6,kernel_size=5,padding=2),nn.Sigmoid(),# 28 * 28 6个通道数；(5-1)/2 =2 --->输出大小为为(1,6,28,28)
    nn.AvgPool2d(kernel_size=2,stride=2),#(1,6,14,14)

    nn.Conv2d(6,16,kernel_size=5),nn.Sigmoid(), #(1，16,10，10) 14-5+1=10
    nn.AvgPool2d(kernel_size=2,stride=2),# (1,16,5,5) 10/2=5

    nn.Flatten(),#全连接层；(1,16 * 5 * 5)
    nn.Linear(16 * 5 * 5,120),nn.Sigmoid(), #(1,120)
    nn.Linear(120,84),nn.Sigmoid(), #(1,84)
    nn.Linear(84,10), #(1,10)

)

X=torch.rand(size=(1,1,28,28)) #1通道
#查看模型参数
for layer in net:
    X=layer(X) #每一层的输出值；
    print(layer.__class__.__name__,'output shape:',X.shape)


#获取数据集 获取数据集的迭代工作：
batch_size=125
train_data,test_data=d2l.load_data_fashion_mnist(batch_size)
print(train_data)


#定义用Gpu计算模型再数据集上的表现
def evaluate_accuracy_gpu(net,data_iter,device=None):

    if isinstance(net,nn.Module):
        net.eval() #进行模型的评估：
        device=net(iter(net.parameters())).device
    metric=d2l.Accumulator(2)
    with torch.no_grad():
        for X,y in data_iter:
            if isinstance(X,list): #如果X是list类型的话，就遍历移动的奥gpu
                X=[x.to(device) for x in X]

            else:
                X=X.to(device)
            #将y的值也移动到gpu上；
            y=y.to(device)
            metric.add(d2l.accuracy(net(X),y),y.numel())

        #返回两者的正确率
        return metric[0] /metric[1]

#训练模型：
def train_chapter6(net,train_iter,test_iter,num_epochs,lr,device):

    def init_weights(m):

        if type(m) == nn.Linear() or type(m) == nn.Conv2d():
            nn.init.xavier_uniform_(m.weight) #初始化权重参数；
    net.apply(init_weights)
    print("training on:",device)
    net.to(device) #将网络移到gpu上
    optimizer=torch.optim.SGD(net.parameters(),lr=lr) #设置优化器 ，传递net参数和学习率
    loss=nn.CrossEntropyLoss()#设置损失函数模块，然后可以每次调用；
    animator=d2l.Animator(xlabel='epoch',xlim=[1,num_epochs],
                 legend=['train_loss','train_acc','test_acc'])

    timer,num_batches=d2l.Timer(),len(train_iter) #设置时间和校批量的大小；

    for epoch in range(num_epochs): #训练多少轮
        metric=d2l.Accumulator(3) #返回的是列表类型

        net.train() #将网络设为train模式
        for i,(X,y) in enumerate(train_iter): #使用枚举的方式；每次迭代训练集的大小；
            #开始计数
            timer.start()
            optimizer.zero_grad() #优化器梯度清0
            X,y=X.to(device),y.to(device) #将数据移到gpu上

            #得到预测zhi
            y_hat=net(X)
            #得到损失值：
            loss=loss(y_hat,y)
            loss.backward() #反向传播
            optimizer.step() #单次优化；
            with torch.no_grad():
                metric.add(loss * X.shape[0],d2l.accuracy(y_hat,y),X.shape[0])

            #停止计数：
            timer.stop()

            train_loss=metric[0] /metric[2] #返回的是列表
            train_acc=metric[1] /metric[2]
            #打印输出结果
            if(i+ 1) % (num_batches // 5) ==0 or i== num_batches -1:

                animator.add(epoch +(i+1) /num_batches,(train_loss,train_acc,None))
        #测试集在gpu上进行测试：
        test_acc=evaluate_accuracy_gpu(net,test_iter)

        #进行绘制图像：
        animator.add(epoch + 1,(None,None,test_acc))
    print(f'loss {train_loss:.3f}, train acc {train_acc:.3f}, '
          f'test acc {test_acc:.3f}')

    print(f'{metric[2] * num_epochs / timer.sum():.1f} examples/sec '
          f'on {str(device)}')

#设置参数：
lr,num_epochs=0.9,10
#进行模型的训练：
train_chapter6(net=net, train_iter=train_data, test_iter=test_data,num_epochs=num_epochs,lr=lr,device=d2l.try_gpu())





















d2l.Accumulator()


#数据可视化：



